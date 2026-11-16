from functools import lru_cache

import torch

from .kv_codec import kernels


def attention_kernels():
    return _attention_kernels(torch.cuda.get_device_capability())


@lru_cache(None)
def _attention_kernels(capability):
    import triton
    import triton.language as tl

    decode = kernels()[3]

    @triton.jit
    def load_tile(codes, scales, special_codes, special_scales, order, extra,
                  positions, channels, head_batch, special_count, frame_count, total,
                  CAPACITY: tl.constexpr, SPECIAL_CAPACITY: tl.constexpr,
                  TOKENS: tl.constexpr, CHANNELS: tl.constexpr, GROUP: tl.constexpr,
                  SPECIAL_GROUP: tl.constexpr, KIND: tl.constexpr, EXTRA_TOKENS: tl.constexpr):
        valid = positions < total
        stored = special_count + frame_count * TOKENS
        special = positions < special_count
        current = positions >= stored
        main_position = tl.maximum(positions - special_count, 0)
        slot = tl.load(order + main_position // TOKENS, valid & ~special & ~current, 0)
        main_token = main_position % TOKENS
        pack: tl.constexpr = 2 if KIND == 'fp4' or KIND == 'int4' else 1
        main_offset = ((head_batch * CAPACITY + slot[:, None]) * TOKENS + main_token[:, None]) * (CHANNELS // pack) + channels[None, :] // pack
        special_offset = (head_batch * SPECIAL_CAPACITY + positions[:, None]) * (CHANNELS // pack) + channels[None, :] // pack
        main_code = tl.load(codes + main_offset, (valid & ~special & ~current)[:, None] & (channels < CHANNELS)[None, :], 0)
        special_code = tl.load(special_codes + special_offset, (valid & special)[:, None] & (channels < CHANNELS)[None, :], 0)
        code = tl.where(special[:, None], special_code, main_code)
        if pack == 2:
            code = (code >> ((channels[None, :] % 2) * 4)) & 15
        if GROUP:
            main_scale_offset = ((head_batch * CAPACITY + slot[:, None]) * tl.cdiv(TOKENS, GROUP) + main_token[:, None] // GROUP) * CHANNELS + channels[None, :]
            special_scale_offset = (head_batch * tl.cdiv(SPECIAL_CAPACITY, SPECIAL_GROUP) + positions[:, None] // SPECIAL_GROUP) * CHANNELS + channels[None, :]
            scale_mask = channels[None, :] < CHANNELS
        else:
            main_scale_offset = (head_batch * CAPACITY + slot[:, None]) * TOKENS + main_token[:, None]
            special_scale_offset = head_batch * SPECIAL_CAPACITY + positions[:, None]
            scale_mask = tl.full((1, channels.shape[0]), True, tl.int1)
        main_scale = tl.load(scales + main_scale_offset, (valid & ~special & ~current)[:, None] & scale_mask, 0.)
        special_scale = tl.load(special_scales + special_scale_offset, (valid & special)[:, None] & scale_mask, 0.)
        values = decode(code, KIND) * tl.where(special[:, None], special_scale, main_scale)
        values = values.to(extra.dtype.element_ty)
        current_offset = (head_batch * EXTRA_TOKENS + positions[:, None] - stored) * CHANNELS + channels[None, :]
        current_values = tl.load(extra + current_offset, (valid & current)[:, None] & (channels < CHANNELS)[None, :], 0.)
        return tl.where(current[:, None], current_values, values)

    @triton.jit
    def partial_kernel(query, key_codes, key_scales, value_codes, value_scales,
                       special_key_codes, special_key_scales, special_value_codes, special_value_scales,
                       order, extra_key, extra_value, partial, maxima, sums,
                       special_count, frame_count, total,
                       QUERIES: tl.constexpr, CHANNELS: tl.constexpr,
                       CAPACITY: tl.constexpr, SPECIAL_CAPACITY: tl.constexpr,
                       TOKENS: tl.constexpr, GROUP: tl.constexpr, SPECIAL_GROUP: tl.constexpr,
                       KIND: tl.constexpr, EXTRA_TOKENS: tl.constexpr, SPLITS: tl.constexpr,
                       SPLIT_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
        query_block, head_batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        channels = tl.arange(0, BLOCK_D)
        query_values = tl.load(query + (head_batch * QUERIES + rows[:, None]) * CHANNELS + channels[None, :],
                               (rows < QUERIES)[:, None] & (channels < CHANNELS)[None, :], 0)
        running_max = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        running_sum = tl.full((BLOCK_M,), 0., tl.float32)
        accumulator = tl.full((BLOCK_M, BLOCK_D), 0., tl.float32)
        start = split * SPLIT_SIZE
        end = tl.minimum(start + SPLIT_SIZE, total)
        for block_start in range(start, end, BLOCK_N):
            positions = block_start + tl.arange(0, BLOCK_N)
            key = load_tile(key_codes, key_scales, special_key_codes, special_key_scales, order, extra_key,
                            positions, channels, head_batch, special_count, frame_count, total,
                            CAPACITY, SPECIAL_CAPACITY, TOKENS, CHANNELS, GROUP, SPECIAL_GROUP, KIND, EXTRA_TOKENS)
            value = load_tile(value_codes, value_scales, special_value_codes, special_value_scales, order, extra_value,
                              positions, channels, head_batch, special_count, frame_count, total,
                              CAPACITY, SPECIAL_CAPACITY, TOKENS, CHANNELS, 0, 1, KIND, EXTRA_TOKENS)
            key = key.to(query.dtype.element_ty)
            value = value.to(query.dtype.element_ty)
            scores = tl.dot(query_values, tl.trans(key)) * (CHANNELS ** -.5)
            scores = tl.where((positions < end)[None, :], scores, -float('inf'))
            next_max = tl.maximum(running_max, tl.max(scores, 1))
            correction = tl.exp(running_max - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            running_sum = running_sum * correction + tl.sum(probabilities, 1)
            accumulator = accumulator * correction[:, None] + tl.dot(probabilities.to(query.dtype.element_ty), value)
            running_max = next_max
        scalar_offset = (head_batch * SPLITS + split) * QUERIES + rows
        tl.store(maxima + scalar_offset, running_max, rows < QUERIES)
        tl.store(sums + scalar_offset, running_sum, rows < QUERIES)
        tl.store(partial + scalar_offset[:, None] * CHANNELS + channels[None, :], accumulator,
                 (rows < QUERIES)[:, None] & (channels < CHANNELS)[None, :])

    @triton.jit
    def merge_kernel(partial, maxima, sums, output, QUERIES: tl.constexpr, CHANNELS: tl.constexpr,
                     SPLITS: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
        row, head_batch = tl.program_id(0), tl.program_id(1)
        splits = tl.arange(0, BLOCK_S)
        channels = tl.arange(0, BLOCK_D)
        scalar_offset = (head_batch * SPLITS + splits) * QUERIES + row
        maximum = tl.load(maxima + scalar_offset, splits < SPLITS, -float('inf'))
        normalizer = tl.load(sums + scalar_offset, splits < SPLITS, 0.)
        correction = tl.exp(maximum - tl.max(maximum, 0))
        values = tl.load(partial + scalar_offset[:, None] * CHANNELS + channels[None, :],
                         (splits < SPLITS)[:, None] & (channels < CHANNELS)[None, :], 0.)
        result = tl.sum(values * correction[:, None], 0) / tl.sum(normalizer * correction, 0)
        tl.store(output + (head_batch * QUERIES + row) * CHANNELS + channels, result, channels < CHANNELS)

    return partial_kernel, merge_kernel


def compressed_attention(query, state, order, extra_key, extra_value, extra_frames, split_size=4096,
                         block_m=32, num_stages=1):
    import triton

    dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else query.dtype
    if not query.is_cuda or dtype not in (torch.bfloat16, torch.float16):
        raise ValueError('Direct compressed attention requires CUDA BF16/FP16 compute')
    if (query.ndim != 4 or query.shape[:2] != state.key.shape[:2]
            or query.shape[-1] != state.key.shape[-1] or query.shape[2] < 1):
        raise ValueError('Query batch/head/channel dimensions must match the compressed cache')
    if (extra_key.shape != extra_value.shape or extra_key.shape[:2] != state.key.shape[:2]
            or extra_key.shape[3:] != state.key.shape[3:] or extra_frames not in (0, extra_key.shape[2])):
        raise ValueError('Invalid current-frame KV dimensions')
    if extra_key.dtype != state.key.dtype or extra_value.dtype != state.value.dtype:
        raise ValueError('Current KV dtype must match its stored decode dtype')
    if any(tensor.device != query.device for tensor in (state.key.codes, state.value.codes, order, extra_key, extra_value)):
        raise ValueError('Direct attention requires a single CUDA device')
    if split_size < 64 or split_size % 64:
        raise ValueError('Split size must be a positive multiple of 64')
    query = query.to(dtype).contiguous()
    batch, heads, queries, channels = query.shape
    _, _, capacity, tokens, _ = state.key.shape
    total = state.special_count + len(state.order) * tokens + extra_frames * tokens
    if total < 1:
        raise ValueError('Attention requires at least one visible token')
    splits = triton.cdiv(total, split_size)
    partial = torch.empty(batch * heads, splits, queries, channels, device=query.device, dtype=torch.float32)
    maxima = torch.empty(batch * heads, splits, queries, device=query.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    output = torch.empty_like(query)
    extra_key = extra_key.flatten(2, 3).contiguous()
    extra_value = extra_value.flatten(2, 3).contiguous()
    with torch.cuda.device(query.device):
        partial_kernel, merge_kernel = attention_kernels()
        partial_kernel[(triton.cdiv(queries, block_m), batch * heads, splits)](
            query, state.key.codes, state.key.scales, state.value.codes, state.value.scales,
            state.special_key.codes, state.special_key.scales, state.special_value.codes, state.special_value.scales,
            order, extra_key, extra_value, partial, maxima, sums, state.special_count, len(state.order), total,
            queries, channels, capacity, state.special_key.shape[3], tokens, state.key.group_size,
            state.special_key.group_size, state.key.format_name, extra_key.shape[2], splits, split_size,
            block_m, 64, max(16, triton.next_power_of_2(channels)), num_warps=4, num_stages=num_stages)
        merge_kernel[(queries, batch * heads)](
            partial, maxima, sums, output, queries, channels, splits, triton.next_power_of_2(splits),
            triton.next_power_of_2(channels), num_warps=4)
    return output
