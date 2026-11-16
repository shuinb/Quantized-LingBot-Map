from contextlib import contextmanager
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def copy_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def compact_kv_gather(old_key, old_value, special_key, special_value, new_key, new_value, output_key, output_value,
               old_batch, old_head, special_batch, special_head,
               new_batch, new_head, new_frame, new_token, new_channel,
               value_batch, value_head, value_frame, value_token, value_channel,
               old_frames, old_special, special, main_frames, anchors, evicted, keep_anchors, capacity,
               HEADS: tl.constexpr, CHANNELS: tl.constexpr, TOKENS: tl.constexpr,
               SPECIAL_PER_FRAME: tl.constexpr, BLOCK: tl.constexpr):
        positions = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        batch, head = tl.program_id(1) // HEADS, tl.program_id(1) % HEADS
        token, channel = positions // CHANNELS, positions % CHANNELS
        valid = token < special + main_frames * TOKENS
        from_special = token < old_special
        from_eviction = (token >= old_special) & (token < special)
        extra_special = token - old_special
        main_token = token - special
        main_frame = main_token // TOKENS
        main_frame = tl.where(keep_anchors, tl.where(main_frame < anchors, main_frame, main_frame + evicted), main_frame + evicted + anchors)
        frame = tl.where(from_eviction, anchors + extra_special // SPECIAL_PER_FRAME, main_frame)
        within_frame = tl.where(from_eviction, extra_special % SPECIAL_PER_FRAME, main_token % TOKENS)
        from_old = (~from_special) & (frame < old_frames)
        from_new = (~from_special) & (frame >= old_frames)
        old_offset = batch * old_batch + head * old_head + (frame * TOKENS + within_frame) * CHANNELS + channel
        special_offset = batch * special_batch + head * special_head + token * CHANNELS + channel
        new_frame_index = frame - old_frames
        key_offset = batch * new_batch + head * new_head + new_frame_index * new_frame + within_frame * new_token + channel * new_channel
        value_offset = batch * value_batch + head * value_head + new_frame_index * value_frame + within_frame * value_token + channel * value_channel
        old_key_data = tl.load(old_key + old_offset, valid & from_old, 0)
        old_value_data = tl.load(old_value + old_offset, valid & from_old, 0)
        special_key_data = tl.load(special_key + special_offset, valid & from_special, 0)
        special_value_data = tl.load(special_value + special_offset, valid & from_special, 0)
        new_key_data = tl.load(new_key + key_offset, valid & from_new, 0)
        new_value_data = tl.load(new_value + value_offset, valid & from_new, 0)
        key = tl.where(from_special, special_key_data, tl.where(from_old, old_key_data, new_key_data))
        value = tl.where(from_special, special_value_data, tl.where(from_old, old_value_data, new_value_data))
        destination = (batch * HEADS + head) * capacity * CHANNELS + positions
        tl.store(output_key + destination, key, valid)
        tl.store(output_value + destination, value, valid)

    return compact_kv_gather


class CompactCache:
    def __init__(self, reserve_frames=None, store_compute_dtype=False):
        if reserve_frames is not None and (type(reserve_frames) is not int or reserve_frames < 1):
            raise ValueError('Reserved cache frame count must be a positive integer')
        self.reserve_frames = reserve_frames
        self.store_compute_dtype = store_compute_dtype
        self.scratch = None
        self.stream = None
        self.updates = 0

    def reset(self):
        self.scratch = None
        self.stream = None

    def prepare(self, attention, cache, index, key, value, special_tokens):
        if torch.is_grad_enabled() or attention.training:
            raise RuntimeError('Compact KV cache is inference-only')
        if getattr(attention, '_kv_distribution_observer', None) is not None:
            raise RuntimeError('Compact cache does not support mutating KV distribution observers')
        if key.is_cuda:
            stream = (key.device, torch.cuda.current_stream(key.device).cuda_stream)
            if self.stream is not None and self.stream != stream:
                raise RuntimeError('Compact cache requires sequential execution on one CUDA stream')
            self.stream = stream
        batch, heads, current_frames, tokens, channels = key.shape
        old_key, old_value = cache[f'k_{index}'], cache[f'v_{index}']
        if self.store_compute_dtype:
            if not key.is_cuda or not torch.is_autocast_enabled('cuda'):
                raise ValueError('Compute-dtype KV storage requires CUDA autocast')
            dtype = torch.get_autocast_dtype('cuda')
            if any(tensor is not None and tensor.dtype != dtype for tensor in (old_key, old_value)):
                raise ValueError('Cannot change SDPA compute dtype with live KV history')
            # Cache maintenance only copies/gathers values. Move the existing
            # SDPA autocast conversion before storage instead of recasting the
            # entire history on every attention call.
            key, value = key.to(dtype), value.to(dtype)
        old_frames = old_key.shape[2] if old_key is not None else 0
        old_special_key, old_special_value = cache.get(f'k_{index}_special'), cache.get(f'v_{index}_special')
        old_special = old_special_key.shape[2] * old_special_key.shape[3] if old_special_key is not None else 0
        skip = cache.get('_skip_append', False)
        total_frames = old_frames + current_frames
        anchors = attention.kv_cache_scale_frames
        evicted = max(0, total_frames - anchors - attention.kv_cache_sliding_window) if tokens > 1 and not skip else 0
        keep_anchors = not evicted or attention.kv_cache_include_scale_frames
        kept_frames = total_frames - evicted - (anchors if evicted and not keep_anchors else 0)
        special_per_frame = min(tokens, 1 if attention.kv_cache_camera_only else special_tokens)
        added_special = evicted * special_per_frame if attention.kv_cache_cross_frame_special else 0
        special = old_special + added_special
        total_tokens = special + kept_frames * tokens
        main_capacity = min(max(anchors + attention.kv_cache_sliding_window, kept_frames) + 1,
                            ((kept_frames + 7) // 8) * 8 + 1)
        capacity = max(total_tokens, main_capacity * tokens + ((special + 511) // 512) * 512)
        if self.reserve_frames is not None and old_frames:
            reserved_main = min(self.reserve_frames, anchors + attention.kv_cache_sliding_window) if tokens > 1 else self.reserve_frames
            reserved_special = (max(self.reserve_frames - anchors - attention.kv_cache_sliding_window, 0) * special_per_frame
                                if tokens > 1 and attention.kv_cache_cross_frame_special else 0)
            capacity = max(capacity, (reserved_main + 1) * tokens + ((reserved_special + 511) // 512) * 512)
        buffers = self.scratch
        if buffers is None or buffers[0].shape[:2] != (batch, heads) or buffers[0].shape[-1] != channels or buffers[0].shape[2] < capacity or buffers[0].dtype != key.dtype or buffers[1].dtype != value.dtype or buffers[0].device != key.device:
            buffers = (torch.empty(batch, heads, capacity, channels, device=key.device, dtype=key.dtype),
                       torch.empty(batch, heads, capacity, channels, device=value.device, dtype=value.dtype))
        capacity = buffers[0].shape[2]
        if key.is_cuda:
            source = old_key if old_key is not None else buffers[0]
            source_value = old_value if old_value is not None else buffers[1]
            special_source = old_special_key if old_special_key is not None else source
            special_source_value = old_special_value if old_special_value is not None else source_value
            with torch.cuda.device(key.device):
                copy_kernel()[((total_tokens * channels + 511) // 512, batch * heads)](
                    source, source_value, special_source, special_source_value, key, value, *buffers,
                    *source.stride()[:2], *special_source.stride()[:2], *key.stride(), *value.stride(),
                    old_frames, old_special, special, kept_frames, anchors, evicted, keep_anchors, capacity,
                    heads, channels, tokens, special_per_frame, 512)
        else:
            for output, previous, previous_special, current in zip(buffers, (old_key, old_value), (old_special_key, old_special_value), (key, value)):
                if old_special:
                    output[:, :, :old_special].copy_(previous_special.flatten(2, 3))
                for special_index in range(added_special):
                    frame = anchors + special_index // special_per_frame
                    source = previous[:, :, frame] if frame < old_frames else current[:, :, frame - old_frames]
                    output[:, :, old_special + special_index].copy_(source[:, :, special_index % special_per_frame])
                for frame in range(kept_frames):
                    source_frame = frame + (evicted if frame >= anchors else 0) if keep_anchors else frame + evicted + anchors
                    source = previous[:, :, source_frame] if source_frame < old_frames else current[:, :, source_frame - old_frames]
                    output[:, :, special + frame * tokens:special + (frame + 1) * tokens].copy_(source)
        full_key, full_value = (buffer[:, :, :total_tokens] for buffer in buffers)
        if skip:
            self.scratch = buffers
        else:
            self.scratch = cache.get(f'_compact_buffers_{index}')
            cache[f'_compact_buffers_{index}'] = buffers
            cache[f'k_{index}'] = full_key[:, :, special:].view(batch, heads, kept_frames, tokens, channels)
            cache[f'v_{index}'] = full_value[:, :, special:].view(batch, heads, kept_frames, tokens, channels)
            if special:
                cache[f'k_{index}_special'] = full_key[:, :, :special].view(batch, heads, special // special_per_frame, special_per_frame, channels)
                cache[f'v_{index}_special'] = full_value[:, :, :special].view(batch, heads, special // special_per_frame, special_per_frame, channels)
        self.updates += 1
        return full_key, full_value


@contextmanager
def compact_streaming_cache(model, reserve_frames=None, store_compute_dtype=False):
    from lingbot_map.layers.attention import SDPAAttention
    aggregator = model.aggregator
    attentions = [block.attn for block in aggregator.global_blocks]
    if not attentions or any(not isinstance(attention, SDPAAttention) for attention in attentions):
        raise TypeError('Compact cache requires GCT SDPA attention')
    if any(hasattr(attention, '_compact_cache') for attention in attentions):
        raise RuntimeError('Compact cache already attached')
    if any(torch.is_tensor(value) for value in aggregator.kv_cache.values()):
        raise RuntimeError('Start compact cache at a sequence boundary, not with live KV history')
    manager = CompactCache(reserve_frames, store_compute_dtype=store_compute_dtype)
    original_clean = aggregator.clean_kv_cache
    previous_override = aggregator.__dict__.get('clean_kv_cache')

    def clean():
        original_clean()
        manager.reset()

    aggregator.clean_kv_cache = clean
    for attention in attentions:
        attention._compact_cache = manager
    try:
        yield manager
    finally:
        model.clean_kv_cache()
        for attention in attentions:
            del attention._compact_cache
        if previous_override is None:
            del aggregator.clean_kv_cache
        else:
            aggregator.clean_kv_cache = previous_override
