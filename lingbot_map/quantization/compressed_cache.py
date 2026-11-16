from contextlib import contextmanager
from dataclasses import dataclass

import torch

from lingbot_map.quantization.kv_codec import (
    PackedKV, allocate, encode_into, format_bits, kernels, reference_decode,
)


@dataclass
class LayerState:
    key: PackedKV
    value: PackedKV
    special_key: PackedKV
    special_value: PackedKV
    order: list
    special_count: int = 0


class CompressedCache:
    def __init__(self, format_name, reserve_frames=None, key_group_size=128,
                 deduplicate_special_scales=False, attention_backend='sdpa', fuse_sdpa_decode=False,
                 fuse_sage_value_decode=False, defer_full_reservation=False):
        format_bits(format_name)
        if reserve_frames is not None and (type(reserve_frames) is not int or reserve_frames < 1):
            raise ValueError('Reserved frames must be a positive integer')
        if key_group_size < 1 or key_group_size & (key_group_size - 1):
            raise ValueError('Key group size must be a positive power of two')
        self.format_name = format_name
        self.reserve_frames = reserve_frames
        self.key_group_size = key_group_size
        if attention_backend not in ('sdpa', 'direct', 'sage'):
            raise ValueError('Compressed attention must be sdpa, direct or sage')
        self.deduplicate_special_scales = deduplicate_special_scales
        self.attention_backend = attention_backend
        self.fuse_sdpa_decode = fuse_sdpa_decode
        self.defer_full_reservation = defer_full_reservation
        if fuse_sage_value_decode and attention_backend != 'sage':
            raise ValueError('Fused Sage value decode requires SageAttention')
        self.fuse_sage_value_decode = fuse_sage_value_decode
        self.updates = 0
        self.reset()

    def reset(self):
        self.states = {}
        self.scratch = None
        self.stream = None
        self._indices = {}

    def _index(self, indices, device):
        identity = (tuple(indices), device)
        if identity not in self._indices:
            self._indices[identity] = torch.tensor(indices, dtype=torch.int32, device=device)
        return self._indices[identity]

    def _allocate(self, key, value, capacity, special_capacity, special_per_frame):
        batch, heads, _, tokens, channels = key.shape
        shape = (batch, heads, capacity, tokens, channels)
        special_shape = (batch, heads, 1, max(special_capacity, 1), channels)
        special_group = special_per_frame if self.deduplicate_special_scales and special_per_frame <= self.key_group_size else 1
        return LayerState(allocate(shape, key.dtype, key.device, self.format_name, self.key_group_size),
                          allocate(shape, value.dtype, value.device, self.format_name, 0),
                          allocate(special_shape, key.dtype, key.device, self.format_name, special_group),
                          allocate(special_shape, value.dtype, value.device, self.format_name, 0), [])

    def _grow_special(self, state, required):
        if required <= state.special_key.shape[3]:
            return
        for name in ('special_key', 'special_value'):
            previous = getattr(state, name)
            shape = (*previous.shape[:3], max(required, previous.shape[3] * 2), previous.shape[4])
            larger = allocate(shape, previous.dtype, previous.codes.device, previous.format_name, previous.group_size)
            larger.codes[:, :, :, :state.special_count].copy_(previous.codes[:, :, :, :state.special_count])
            scale_count = ((state.special_count + previous.group_size - 1) // previous.group_size
                           if previous.group_size else state.special_count)
            larger.scales[:, :, :, :scale_count].copy_(previous.scales[:, :, :, :scale_count])
            setattr(state, name, larger)

    def _grow_main(self, state, capacity):
        """Copy existing codes/scales once; never decode or requantize history."""
        for name in ('key', 'value'):
            previous = getattr(state, name)
            shape = (*previous.shape[:2], capacity, *previous.shape[3:])
            larger = allocate(shape, previous.dtype, previous.codes.device, previous.format_name, previous.group_size)
            larger.codes[:, :, :previous.shape[2]].copy_(previous.codes)
            larger.scales[:, :, :previous.shape[2]].copy_(previous.scales)
            setattr(state, name, larger)

    def _preserve_special(self, state, slots, special_per_frame):
        self._grow_special(state, state.special_count + len(slots) * special_per_frame)
        for slot in slots:
            start = state.special_count
            if state.key.codes.is_cuda:
                batch, heads, capacity, tokens, channels = state.key.shape
                with torch.cuda.device(state.key.codes.device):
                    kernels()[2][((special_per_frame * channels + 127) // 128, batch * heads)](
                        state.key.codes, state.key.scales, state.value.codes, state.value.scales,
                        state.special_key.codes, state.special_key.scales, state.special_value.codes, state.special_value.scales,
                        slot, start, capacity, state.special_key.shape[3], tokens, channels,
                        self.key_group_size, special_per_frame, 8 // format_bits(self.format_name), 128,
                        state.special_key.group_size)
                state.special_count += special_per_frame
                continue
            for main, special in ((state.key, state.special_key), (state.value, state.special_value)):
                special.codes[:, :, 0, start:start + special_per_frame].copy_(main.codes[:, :, slot, :special_per_frame])
                if main.group_size:
                    indices = torch.arange(special_per_frame, device=main.codes.device) // main.group_size
                    scales = main.scales[:, :, slot].index_select(2, indices)
                else:
                    scales = main.scales[:, :, slot, :special_per_frame]
                if special.group_size > 1:
                    special.scales[:, :, 0, start // special.group_size:(start + special_per_frame) // special.group_size].copy_(scales[:, :, :1])
                else:
                    special.scales[:, :, 0, start:start + special_per_frame].copy_(scales)
            state.special_count += special_per_frame

    def _decode(self, state, key, value, extra_frames, compute_dtype=None, value_output_dtype=None):
        key_dtype = compute_dtype if compute_dtype is not None and key.dtype == torch.float32 else key.dtype
        value_dtype = value_output_dtype or value.dtype
        batch, heads, _, tokens, channels = key.shape
        special = state.special_count
        stored_tokens = special + len(state.order) * tokens
        total_tokens = stored_tokens + extra_frames * tokens
        capacity = state.key.shape[2] * tokens + state.special_key.shape[3] + max(extra_frames, 1) * tokens
        if (self.scratch is None or self.scratch[0].shape[:2] != (batch, heads)
                or self.scratch[0].shape[2] < capacity or self.scratch[0].shape[-1] != channels
                or self.scratch[0].dtype != key_dtype or self.scratch[1].dtype != value_dtype
                or self.scratch[0].device != key.device):
            self.scratch = (torch.empty(batch, heads, capacity, channels, device=key.device, dtype=key_dtype),
                            torch.empty(batch, heads, capacity, channels, dtype=value_dtype, device=key.device))
        order = self._index(state.order, key.device)
        for main, old_special, output, current in zip((state.key, state.value),
                                                    (state.special_key, state.special_value), self.scratch, (key, value)):
            if key.is_cuda and stored_tokens:
                with torch.cuda.device(key.device):
                    kernels()[1][((stored_tokens * channels + 511) // 512, batch * heads)](
                        main.codes, main.scales, old_special.codes, old_special.scales, order, output,
                        main.shape[2], old_special.shape[3], tokens, channels, main.group_size,
                        self.format_name, output.shape[2], old_special.group_size or 1,
                        special, len(state.order), 512,
                        main.dtype == torch.bfloat16 and output.dtype == torch.float16)
            elif not key.is_cuda:
                selected = PackedKV(main.codes.index_select(2, order.long()), main.scales.index_select(2, order.long()),
                                    (batch, heads, len(state.order), tokens, channels), main.dtype,
                                    self.format_name, main.group_size)
                output[:, :, special:stored_tokens].copy_(reference_decode(selected).flatten(2, 3))
                if special:
                    scale_count = ((special + old_special.group_size - 1) // old_special.group_size
                                   if old_special.group_size else special)
                    selected_special = PackedKV(old_special.codes[:, :, :, :special], old_special.scales[:, :, :, :scale_count],
                                                (batch, heads, 1, special, channels), old_special.dtype,
                                                self.format_name, old_special.group_size)
                    output[:, :, :special].copy_(reference_decode(selected_special).flatten(2, 3))
            if extra_frames:
                output[:, :, stored_tokens:total_tokens].copy_(current.flatten(2, 3))
        return tuple(output[:, :, :total_tokens] for output in self.scratch)

    def _prepare_state(self, attention, cache, index, key, value, special_tokens):
        if torch.is_grad_enabled() or attention.training:
            raise RuntimeError('Compressed KV cache is inference-only')
        if getattr(attention, '_kv_distribution_observer', None) is not None or getattr(attention, '_kv_write_quantizer', None) is not None:
            raise RuntimeError('Do not combine compressed KV with mutating observers or QDQ write hooks')
        if key.is_cuda:
            stream = (key.device, torch.cuda.current_stream(key.device).cuda_stream)
            if self.stream is not None and self.stream != stream:
                raise RuntimeError('Compressed cache requires sequential execution on one CUDA stream')
            self.stream = stream
        batch, heads, current_frames, tokens, channels = key.shape
        anchors, window = attention.kv_cache_scale_frames, attention.kv_cache_sliding_window
        if tokens > 1 and current_frames > anchors + window:
            raise ValueError('Joint prefill exceeds full-frame cache capacity')
        special_per_frame = min(tokens, 1 if attention.kv_cache_camera_only else special_tokens)
        state = self.states.get(index)
        if state is None:
            capacity = max(current_frames, anchors + window) if tokens > 1 else max(self.reserve_frames or 1024, current_frames)
            if self.defer_full_reservation and tokens > 1:
                capacity = current_frames
            special_capacity = max(0, (self.reserve_frames or 0) - anchors - window) * special_per_frame
            special_capacity = max(512, ((special_capacity + 511) // 512) * 512)
            state = self._allocate(key, value, capacity, special_capacity, special_per_frame)
            self.states[index] = state
        elif tokens > 1 and state.key.shape[2] < anchors + window:
            self._grow_main(state, anchors + window)
            # Non-keyframes may trigger the growth too. Replace borrowed cache
            # views immediately so the old compact buffers are not kept alive.
            cache[f'k_{index}'] = state.key.codes[:, :, :len(state.order)]
            cache[f'v_{index}'] = state.value.codes[:, :, :len(state.order)]
        if state.order and current_frames != 1:
            raise ValueError('Compressed streaming accepts joint initial prefill, then one frame per update')
        skip = bool(cache.get('_skip_append', False))
        if not skip:
            evicted = max(0, len(state.order) + current_frames - anchors - window) if tokens > 1 else 0
            if evicted and attention.kv_cache_cross_frame_special:
                self._preserve_special(state, state.order[anchors:anchors + evicted], special_per_frame)
            keep_anchors = not evicted or attention.kv_cache_include_scale_frames
            retained = ((state.order[:anchors] if keep_anchors else []) + state.order[anchors + evicted:]
                        if evicted else list(state.order))
            free = [slot for slot in range(state.key.shape[2]) if slot not in retained]
            if len(free) < current_frames:
                raise ValueError('Compressed cache capacity exceeded; increase --reserve-cache-frames')
            inserted = free[:current_frames]
            slots = self._index(inserted, key.device)
            encode_into(key, state.key, slots)
            encode_into(value, state.value, slots)
            state.order = retained + inserted
            cache[f'k_{index}'] = state.key.codes[:, :, :len(state.order)]
            cache[f'v_{index}'] = state.value.codes[:, :, :len(state.order)]
        self.updates += 1
        return state, current_frames if skip else 0

    def prepare(self, attention, cache, index, key, value, special_tokens):
        """Consume finite K/V; returned workspace views expire on the next prepare/reset."""
        state, extra_frames = self._prepare_state(attention, cache, index, key, value, special_tokens)
        return self._decode(state, key, value, extra_frames)

    def attention(self, attention, cache, index, query, key, value, special_tokens):
        if self.attention_backend in ('direct', 'sage'):
            dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else query.dtype
            if not query.is_cuda or dtype not in (torch.bfloat16, torch.float16):
                label = 'Direct compressed' if self.attention_backend == 'direct' else 'Sage'
                raise ValueError(f'{label} attention requires CUDA BF16/FP16 compute')
            if self.fuse_sage_value_decode and (value.dtype != dtype or query.shape[-1] not in (64, 128)):
                raise ValueError('Fused Sage value decode requires matching V compute dtype and head dimension 64/128')
        if self.attention_backend == 'sage':
            try:
                from sageattention import sageattn_qk_int8_pv_fp16_cuda
            except ImportError as error:
                raise RuntimeError('The optional sage backend requires SageAttention 2.2.0 with its CUDA extension') from error
        state, extra_frames = self._prepare_state(attention, cache, index, key, value, special_tokens)
        if self.attention_backend == 'direct':
            from .compressed_attention import compressed_attention
            return compressed_attention(query, state, self._index(state.order, key.device), key, value, extra_frames)
        # CUDA autocast otherwise casts the entire decoded FP32 key a second
        # time. Keep FP32 scale arithmetic, writing the final SDPA dtype once.
        compute_dtype = (torch.get_autocast_dtype('cuda')
                         if self.fuse_sdpa_decode and query.is_cuda and torch.is_autocast_enabled('cuda') else None)
        full_key, full_value = self._decode(state, key, value, extra_frames, compute_dtype,
                                           torch.float16 if self.fuse_sage_value_decode else None)
        if self.attention_backend == 'sage':
            if self.fuse_sage_value_decode:
                from .sage_attention import sage_attention_fp16_values
                return sage_attention_fp16_values(query.to(dtype), full_key.to(dtype), full_value)
            # Explicit numerical tradeoff: INT8 QK, FP16 PV with blockwise
            # FP32 accumulation. Retained history and FP8 codes are unchanged.
            # The upstream function sets the current device; restore it here.
            with torch.cuda.device(query.device):
                return sageattn_qk_int8_pv_fp16_cuda(
                    query.to(dtype), full_key.to(dtype), full_value.to(dtype),
                    tensor_layout='HND', is_causal=False, qk_quant_gran='per_warp',
                    pv_accum_dtype='fp16+fp32', smooth_k=True, smooth_v=False)
        return torch.nn.functional.scaled_dot_product_attention(query, full_key, full_value, dropout_p=0.)

    def memory_report(self):
        storages = [storage for state in self.states.values()
                    for storage in (state.key, state.value, state.special_key, state.special_value)]
        return {'compressed_storage': True, 'kv_format': self.format_name, 'key_group_size': self.key_group_size,
                'defer_full_reservation': self.defer_full_reservation,
                'fuse_sdpa_decode': self.fuse_sdpa_decode,
                'fuse_sage_value_decode': self.fuse_sage_value_decode,
                'scale_dtype': 'torch.float32', 'attention_compute': self.attention_backend,
                'deduplicate_special_scales': self.deduplicate_special_scales,
                'workspace_note': 'Direct attention uses temporary split partials, included in allocated peak;shared_scratch counts dense decode only',
                'storage_bytes': sum(storage.storage_bytes for storage in storages),
                'code_bytes': sum(storage.codes.numel() for storage in storages),
                'scale_bytes': sum(storage.scales.numel() * 4 for storage in storages),
                'shared_scratch_bytes': sum(tensor.numel() * tensor.element_size() for tensor in self.scratch)
                if self.scratch is not None else 0,
                'index_bytes': sum(tensor.numel() * tensor.element_size() for tensor in self._indices.values()),
                'layers': len(self.states),
                'full_frames': sorted({len(state.order) for state in self.states.values()}),
                'special_tokens': sorted({state.special_count for state in self.states.values()})}


@contextmanager
def compressed_streaming_cache(model, format_name, reserve_frames=None,
                               deduplicate_special_scales=False, attention_backend='sdpa', fuse_sdpa_decode=False,
                               fuse_sage_value_decode=False, defer_full_reservation=False):
    from lingbot_map.layers.attention import SDPAAttention
    aggregator = model.aggregator
    attentions = [block.attn for block in aggregator.global_blocks]
    if not attentions or any(not isinstance(attention, SDPAAttention) for attention in attentions):
        raise TypeError('Compressed cache requires GCT SDPA attention')
    if any(hasattr(attention, '_compact_cache') for attention in attentions):
        raise RuntimeError('Compact cache already attached')
    if any(torch.is_tensor(value) for value in aggregator.kv_cache.values()):
        raise RuntimeError('Start compressed cache at a sequence boundary')
    manager = CompressedCache(format_name, reserve_frames,
                              deduplicate_special_scales=deduplicate_special_scales, attention_backend=attention_backend,
                              fuse_sdpa_decode=fuse_sdpa_decode, fuse_sage_value_decode=fuse_sage_value_decode,
                              defer_full_reservation=defer_full_reservation)
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
