from dataclasses import dataclass
from functools import lru_cache

import torch


FORMATS = ('fp8', 'fp4', 'int8', 'int4')


def format_bits(format_name):
    if format_name not in FORMATS:
        raise ValueError(f'Unknown KV format: {format_name}')
    return 4 if format_name.endswith('4') else 8


@dataclass
class PackedKV:
    codes: torch.Tensor
    scales: torch.Tensor
    shape: tuple
    dtype: torch.dtype
    format_name: str
    group_size: int

    @property
    def storage_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.codes, self.scales))


def allocate(shape, dtype, device, format_name, group_size):
    bits = format_bits(format_name)
    batch, heads, frames, tokens, channels = shape
    if channels % 2 or group_size < 0 or min(shape) < 1:
        raise ValueError('Require positive 5D shape, even channels, and nonnegative group size')
    scale_shape = ((batch, heads, frames, (tokens + group_size - 1) // group_size, channels)
                   if group_size else (batch, heads, frames, tokens, 1))
    return PackedKV(torch.empty(batch, heads, frames, tokens, channels * bits // 8,
                                dtype=torch.uint8, device=device),
                    torch.empty(scale_shape, dtype=torch.float32, device=device),
                    tuple(shape), dtype, format_name, group_size)


def fp4_codes(values):
    magnitude = values.abs()
    codes = torch.zeros_like(magnitude, dtype=torch.uint8)
    for level, threshold in enumerate((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.), 1):
        above = magnitude > threshold if level % 2 else magnitude >= threshold
        codes = torch.where(above, level, codes)
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def reference_encode(values, format_name, group_size):
    storage = allocate(values.shape, values.dtype, values.device, format_name, group_size)
    if not torch.isfinite(values).all():
        raise ValueError('Cannot encode nonfinite KV values')
    source = values.float()
    batch, heads, frames, tokens, channels = values.shape
    if group_size:
        padded = torch.nn.functional.pad(source, (0, 0, 0, (-tokens) % group_size))
        maxima = padded.reshape(batch, heads, frames, -1, group_size, channels).abs().amax(-2)
    else:
        maxima = source.abs().amax(-1, keepdim=True)
    maximum = {'fp8': 448., 'fp4': 6., 'int8': 127., 'int4': 7.}[format_name]
    scales = (maxima / maximum).clamp_min(1e-12)
    expanded = scales.repeat_interleave(group_size, dim=3)[:, :, :, :tokens] if group_size else scales
    normalized = source / expanded
    if format_name == 'fp8':
        codes = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    elif format_name == 'fp4':
        codes = fp4_codes(normalized)
    else:
        bound = int(maximum)
        codes = normalized.round().clamp(-bound - 1, bound).to(torch.int8).view(torch.uint8)
    if format_bits(format_name) == 4:
        codes = (codes[..., ::2] & 15) | ((codes[..., 1::2] & 15) << 4)
    storage.codes.copy_(codes)
    storage.scales.copy_(scales)
    return storage


def reference_decode(storage):
    codes = storage.codes
    if format_bits(storage.format_name) == 4:
        codes = torch.stack((codes & 15, codes >> 4), -1).flatten(-2)
    if storage.format_name == 'fp8':
        values = codes.contiguous().view(torch.float8_e4m3fn).float()
    elif storage.format_name == 'fp4':
        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=codes.device)
        values = table[(codes & 7).long()] * torch.where(codes >= 8, -1., 1.)
    elif storage.format_name == 'int4':
        values = torch.where(codes >= 8, codes.float() - 16, codes.float())
    else:
        values = codes.contiguous().view(torch.int8).float()
    scales = (storage.scales.repeat_interleave(storage.group_size, dim=3)[:, :, :, :storage.shape[3]]
              if storage.group_size else storage.scales)
    return (values * scales).to(storage.dtype)


def kernels():
    # Ampere has no E4M3 conversion instruction. Select by the active device,
    # rather than caching a device-specific closure for all GPUs in a process.
    return _kernels(torch.cuda.get_device_capability() >= (8, 9))


@lru_cache(None)
def _kernels(native_fp8):
    import triton as triton
    import triton.language as tl
    NATIVE_FP8 = tl.constexpr(native_fp8)

    @triton.jit
    def number_encode(values, KIND: tl.constexpr):
        if KIND == 'fp8':
            values = tl.minimum(tl.maximum(values, -448.), 448.)
            if NATIVE_FP8:
                return values.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            else:
                bits = values.to(tl.int32, bitcast=True)
                magnitude = bits & 0x7fffffff
                # Drop 20 mantissa bits, round nearest/even, rebase exponent.
                normal = ((magnitude + 0x7ffff + ((magnitude >> 20) & 1)) >> 20) - 960
                subnormal = tl.extra.cuda.libdevice.nearbyint(tl.abs(values) * 512.).to(tl.int32)
                code = tl.where(tl.abs(values) < .015625, subnormal, normal)
                return (code | ((bits >> 24) & 128)).to(tl.uint8)
        elif KIND == 'fp4':
            magnitude = tl.abs(values)
            result = tl.where(magnitude > .25, 1, 0)
            result = tl.where(magnitude >= .75, 2, result)
            result = tl.where(magnitude > 1.25, 3, result)
            result = tl.where(magnitude >= 1.75, 4, result)
            result = tl.where(magnitude > 2.5, 5, result)
            result = tl.where(magnitude >= 3.5, 6, result)
            result = tl.where(magnitude > 5., 7, result)
            sign = (values.to(tl.int32, bitcast=True) >> 28) & 8
            return (result | sign).to(tl.uint8)
        else:
            bound: tl.constexpr = 7 if KIND == 'int4' else 127
            return tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(values), -bound - 1), bound).to(tl.int8).to(tl.uint8, bitcast=True)

    @triton.jit
    def number_decode(codes, KIND: tl.constexpr):
        if KIND == 'fp8':
            if NATIVE_FP8:
                return codes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            else:
                magnitude = (codes & 127).to(tl.int32)
                normal_bits = (magnitude << 20) + (120 << 23)
                normal_bits = tl.where(magnitude == 127, 0x7fffffff, normal_bits)
                value = tl.where(magnitude < 8, magnitude.to(tl.float32) / 512.,
                                 normal_bits.to(tl.float32, bitcast=True))
                # Bitwise sign also preserves negative zero.
                bits = value.to(tl.int32, bitcast=True) | ((codes.to(tl.int32) & 128) << 24)
                return bits.to(tl.float32, bitcast=True)
        elif KIND == 'fp4':
            magnitude = codes & 7
            result = tl.where(magnitude == 0, 0., .5)
            result = tl.where(magnitude == 2, 1., result)
            result = tl.where(magnitude == 3, 1.5, result)
            result = tl.where(magnitude == 4, 2., result)
            result = tl.where(magnitude == 5, 3., result)
            result = tl.where(magnitude == 6, 4., result)
            result = tl.where(magnitude == 7, 6., result)
            return tl.where(codes & 8 != 0, -result, result)
        elif KIND == 'int4':
            return tl.where(codes >= 8, codes.to(tl.float32) - 16., codes.to(tl.float32))
        else:
            return codes.to(tl.int8, bitcast=True).to(tl.float32)

    @triton.jit
    def encode_kernel(source, codes, scales, slots, stride_batch: tl.constexpr, stride_head: tl.constexpr,
                      stride_frame: tl.constexpr, stride_token: tl.constexpr, stride_channel: tl.constexpr,
                      FRAMES: tl.constexpr, HEADS: tl.constexpr, TOKENS: tl.constexpr, CHANNELS: tl.constexpr,
                      CAPACITY: tl.constexpr, GROUP: tl.constexpr, KIND: tl.constexpr,
                      ROWS: tl.constexpr, COLS: tl.constexpr):
        group = tl.program_id(0)
        frame = tl.program_id(1) % FRAMES
        head_batch = tl.program_id(1) // FRAMES
        head = head_batch % HEADS
        batch = head_batch // HEADS
        token = group * ROWS + tl.arange(0, ROWS)
        channel = tl.arange(0, COLS)
        values = tl.load(source + batch * stride_batch + head * stride_head + frame * stride_frame
                         + token[:, None] * stride_token + channel[None, :] * stride_channel,
                         (token[:, None] < TOKENS) & (channel[None, :] < CHANNELS), 0).to(tl.float32)
        maximum: tl.constexpr = 448. if KIND == 'fp8' else 6. if KIND == 'fp4' else 127. if KIND == 'int8' else 7.
        slot = tl.load(slots + frame)
        if GROUP:
            scale = tl.maximum(tl.div_rn(tl.max(tl.abs(values), axis=0), maximum), 1e-12)
            scale_offset = ((head_batch * CAPACITY + slot) * tl.cdiv(TOKENS, GROUP) + group) * CHANNELS + channel
            tl.store(scales + scale_offset, scale, channel < CHANNELS)
            encoded = number_encode(tl.div_rn(values, scale[None, :]), KIND)
        else:
            scale = tl.maximum(tl.div_rn(tl.max(tl.abs(values), axis=1), maximum), 1e-12)
            tl.store(scales + (head_batch * CAPACITY + slot) * TOKENS + token, scale, token < TOKENS)
            encoded = number_encode(tl.div_rn(values, scale[:, None]), KIND)
        if KIND == 'fp4' or KIND == 'int4':
            low, high = tl.split(tl.reshape(encoded, (ROWS, COLS // 2, 2)))
            packed = (low & 15) | ((high & 15) << 4)
            packed_channel = tl.arange(0, COLS // 2)
            offset = ((head_batch * CAPACITY + slot) * TOKENS + token[:, None]) * (CHANNELS // 2) + packed_channel[None, :]
            tl.store(codes + offset, packed, (token[:, None] < TOKENS) & (packed_channel[None, :] < CHANNELS // 2))
        else:
            offset = ((head_batch * CAPACITY + slot) * TOKENS + token[:, None]) * CHANNELS + channel[None, :]
            tl.store(codes + offset, encoded, (token[:, None] < TOKENS) & (channel[None, :] < CHANNELS))

    @triton.jit
    def gather_kernel(codes, scales, special_codes, special_scales, order, output,
                      CAPACITY: tl.constexpr, SPECIAL_CAPACITY: tl.constexpr,
                      TOKENS: tl.constexpr, CHANNELS: tl.constexpr, GROUP: tl.constexpr,
                      KIND: tl.constexpr, OUTPUT_CAPACITY: tl.constexpr, SPECIAL_GROUP: tl.constexpr,
                      SPECIAL, FRAMES, BLOCK: tl.constexpr, ROUND_BF16: tl.constexpr = False):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        head_batch = tl.program_id(1)
        token = offset // CHANNELS
        channel = offset % CHANNELS
        valid = token < SPECIAL + FRAMES * TOKENS
        is_special = token < SPECIAL
        logical_frame = tl.maximum(token - SPECIAL, 0) // TOKENS
        slot = tl.load(order + logical_frame, valid & ~is_special, 0)
        main_token = tl.maximum(token - SPECIAL, 0) % TOKENS
        pack: tl.constexpr = 2 if KIND == 'fp4' or KIND == 'int4' else 1
        main_offset = ((head_batch * CAPACITY + slot) * TOKENS + main_token) * (CHANNELS // pack) + channel // pack
        special_offset = (head_batch * SPECIAL_CAPACITY + token) * (CHANNELS // pack) + channel // pack
        code = tl.load(codes + main_offset, valid & ~is_special, 0)
        special_code = tl.load(special_codes + special_offset, valid & is_special, 0)
        code = tl.where(is_special, special_code, code)
        if pack == 2:
            code = (code >> ((channel % 2) * 4)) & 15
        if GROUP:
            scale_offset = ((head_batch * CAPACITY + slot) * tl.cdiv(TOKENS, GROUP) + main_token // GROUP) * CHANNELS + channel
            special_scale_offset = (head_batch * tl.cdiv(SPECIAL_CAPACITY, SPECIAL_GROUP) + token // SPECIAL_GROUP) * CHANNELS + channel
        else:
            scale_offset = (head_batch * CAPACITY + slot) * TOKENS + main_token
            special_scale_offset = head_batch * SPECIAL_CAPACITY + token
        scale = tl.load(scales + scale_offset, valid & ~is_special, 0.)
        special_scale = tl.load(special_scales + special_scale_offset, valid & is_special, 0.)
        decoded = number_decode(code, KIND) * tl.where(is_special, special_scale, scale)
        if ROUND_BF16:
            # Preserve the original FP32 -> BF16 -> FP16 rounding sequence.
            decoded = decoded.to(tl.bfloat16).to(tl.float32)
        tl.store(output + head_batch * OUTPUT_CAPACITY * CHANNELS + offset, decoded, valid)

    @triton.jit
    def preserve_kernel(key_codes, key_scales, value_codes, value_scales,
                        special_key_codes, special_key_scales, special_value_codes, special_value_scales,
                        slot, start, CAPACITY: tl.constexpr, SPECIAL_CAPACITY: tl.constexpr,
                        TOKENS: tl.constexpr, CHANNELS: tl.constexpr, GROUP: tl.constexpr,
                        SPECIAL_TOKENS: tl.constexpr, PACK: tl.constexpr, BLOCK: tl.constexpr,
                        SPECIAL_GROUP: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        head_batch = tl.program_id(1)
        token, channel = offset // CHANNELS, offset % CHANNELS
        valid = token < SPECIAL_TOKENS
        packed_channels: tl.constexpr = CHANNELS // PACK
        code_valid = valid & (channel < packed_channels)
        code_source = ((head_batch * CAPACITY + slot) * TOKENS + token) * packed_channels + channel
        code_destination = (head_batch * SPECIAL_CAPACITY + start + token) * packed_channels + channel
        key_code = tl.load(key_codes + code_source, code_valid, 0)
        value_code = tl.load(value_codes + code_source, code_valid, 0)
        tl.store(special_key_codes + code_destination, key_code, code_valid)
        tl.store(special_value_codes + code_destination, value_code, code_valid)
        key_scale_source = ((head_batch * CAPACITY + slot) * tl.cdiv(TOKENS, GROUP) + token // GROUP) * CHANNELS + channel
        key_scale = tl.load(key_scales + key_scale_source, valid, 0.)
        scale_destination = (head_batch * tl.cdiv(SPECIAL_CAPACITY, SPECIAL_GROUP) + (start + token) // SPECIAL_GROUP) * CHANNELS + channel
        tl.store(special_key_scales + scale_destination, key_scale, valid & (token % SPECIAL_GROUP == 0))
        value_scale = tl.load(value_scales + (head_batch * CAPACITY + slot) * TOKENS + token, valid & (channel == 0), 0.)
        tl.store(special_value_scales + head_batch * SPECIAL_CAPACITY + start + token, value_scale, valid & (channel == 0))

    return encode_kernel, gather_kernel, preserve_kernel, number_decode


def encode_into(values, storage, slots):
    """Fast internal write: input must be finite, and slots must be valid unique indices."""
    if tuple(values.shape[:2]) != storage.shape[:2] or tuple(values.shape[3:]) != storage.shape[3:]:
        raise ValueError('KV codec source shape mismatch')
    if values.device != storage.codes.device or values.dtype != storage.dtype:
        raise ValueError('KV codec dtype/device mismatch')
    if not values.is_cuda:
        encoded = reference_encode(values, storage.format_name, storage.group_size)
        storage.codes[:, :, slots] = encoded.codes
        storage.scales[:, :, slots] = encoded.scales
        return
    import triton
    batch, heads, frames, tokens, channels = values.shape
    rows = storage.group_size or 16
    with torch.cuda.device(values.device):
        kernels()[0][(triton.cdiv(tokens, rows), batch * heads * frames)](
            values, storage.codes, storage.scales, slots, *values.stride(),
            frames, heads, tokens, channels, storage.shape[2], storage.group_size, storage.format_name,
            rows, triton.next_power_of_2(channels), num_warps=4)


def encode(values, format_name, group_size):
    if not torch.isfinite(values).all():
        raise ValueError('Cannot encode nonfinite KV values')
    storage = allocate(values.shape, values.dtype, values.device, format_name, group_size)
    slots = torch.arange(values.shape[2], device=values.device)
    encode_into(values, storage, slots)
    return storage
