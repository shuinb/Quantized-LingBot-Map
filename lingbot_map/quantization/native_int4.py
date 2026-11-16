from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .core import int4_quantize, minmax_scale


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name('csrc')
    cutlass = Path(os.environ.get('CUTLASS_PATH', Path(__file__).resolve().parents[2] / 'third_party/cutlass'))
    if not (cutlass / 'include/cutlass/gemm/device/gemm.h').is_file():
        raise FileNotFoundError('Set CUTLASS_PATH to NVIDIA CUTLASS v3.5.1; see README.md')
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0;8.6;8.9')
    return load(name='lingbot_cutlass_s4', sources=[str(source / 'int4_bindings.cpp'), str(source / 'int4_gemm.cu')],
                extra_include_paths=[str(cutlass / 'include')], extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', '--fmad=false'], verbose=False)


@torch.library.custom_op('lingbot_int4::linear', mutates_args=(), device_types='cuda')
def compiled_int4_linear(inputs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                         bias: Optional[torch.Tensor], smoothing: Optional[torch.Tensor],
                         signs: Optional[torch.Tensor], output_type: int,
                         activation_clip: Optional[torch.Tensor], fast_rotation: bool,
                         fused_epilogue: bool = False) -> torch.Tensor:
    return extension().linear(inputs, weight, weight_scale, bias, smoothing, signs, output_type,
                              activation_clip, fast_rotation, fused_epilogue)


@compiled_int4_linear.register_fake
def compiled_int4_linear_fake(inputs, weight, weight_scale, bias, smoothing, signs,
                              output_type, activation_clip, fast_rotation, fused_epilogue=False):
    dtype = (torch.float32, torch.float16, torch.bfloat16)[output_type]
    return inputs.new_empty((inputs.shape[0], weight.shape[0]), dtype=dtype)


def pack_int4(values):
    if values.dtype != torch.int8 or values.shape[-1] % 2:
        raise ValueError('Packing requires int8 codes and an even last dimension')
    if ((values < -8) | (values > 7)).any():
        raise ValueError('Signed INT4 code outside [-8, 7]')
    return ((values[..., ::2].to(torch.uint8) & 15) | ((values[..., 1::2].to(torch.uint8) & 15) << 4)).contiguous()


def unpack_int4(packed):
    if packed.dtype != torch.uint8:
        raise ValueError('Packed INT4 storage must be uint8')
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).to(torch.int8)
    return torch.where(codes >= 8, codes - 16, codes)


@lru_cache(maxsize=1)
@torch.compiler.assume_constant_result
def triton_kernels():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def quantize_kernel(inputs, signs, smoothing, clipping, packed, scales, CHANNELS: tl.constexpr,
                        ROTATION: tl.constexpr, SMOOTHING: tl.constexpr, CLIPPING: tl.constexpr,
                        ROWS: tl.constexpr, INT8_OUTPUT: tl.constexpr):
        row = tl.program_id(0)
        indices = tl.arange(0, CHANNELS)
        values = tl.load(inputs + row * CHANNELS + indices, row < ROWS, 0).to(tl.float32)
        if ROTATION:
            values *= tl.load(signs + indices)
            for stage in tl.static_range(0, tl.constexpr(CHANNELS.bit_length() - 1)):
                stride = 1 << stage
                partner = tl.gather(values, indices ^ stride, 0)
                values = tl.where((indices & stride) == 0, values + partner, partner - values)
            values *= CHANNELS ** -0.5
        if SMOOTHING:
            values = tl.div_rn(values, tl.load(smoothing + indices))
        if CLIPPING:
            lower = tl.minimum(tl.min(values, 0), 0.) * tl.load(clipping)
            upper = tl.maximum(tl.max(values, 0), 0.) * tl.load(clipping + 1)
            maximum = tl.maximum(-lower, upper)
        else:
            maximum = tl.max(tl.abs(values), 0)
        scale = tl.where(maximum == 0., 1., tl.maximum(maximum / 7., 1.1754943508222875e-38))
        invalid = tl.sum(((values != values) | (tl.abs(values) == float('inf'))).to(tl.int32), 0) > 0
        scale = tl.where(invalid, float('nan'), scale)
        codes = libdevice.nearbyint(tl.div_rn(values, scale))
        codes = tl.minimum(tl.maximum(codes, -8.), 7.).to(tl.int32)
        if INT8_OUTPUT:
            tl.store(packed + row * CHANNELS + indices, codes.to(tl.int8))
        else:
            codes &= 15
            byte_indices = tl.arange(0, CHANNELS // 2)
            low = tl.gather(codes, 2 * byte_indices, 0)
            high = tl.gather(codes, 2 * byte_indices + 1, 0)
            tl.store(packed + row * (CHANNELS // 2) + byte_indices, (low | (high << 4)).to(tl.uint8))
        tl.store(scales + row, scale)

    @triton.jit
    def epilogue_kernel(accumulator, activation_scale, weight_scale, bias, output, SIZE: tl.constexpr,
                        COLUMNS: tl.constexpr, BIAS: tl.constexpr, BLOCK: tl.constexpr):
        indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = indices < SIZE
        rows, columns = indices // COLUMNS, indices % COLUMNS
        values = tl.load(accumulator + indices, valid, 0).to(tl.float32)
        values = libdevice.mul_rn(values, tl.load(activation_scale + rows, valid, 0))
        values = libdevice.mul_rn(values, tl.load(weight_scale + columns, valid, 0))
        if BIAS:
            values = libdevice.add_rn(values, tl.load(bias + columns, valid, 0))
        tl.store(output + indices, values, valid)

    return quantize_kernel, epilogue_kernel


def quantize_activation(inputs, signs=None, smoothing=None, activation_clip=None, *, int8_output=False):
    if not inputs.is_cuda or inputs.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('Native activation quantization requires CUDA FP32/FP16/BF16')
    channels = inputs.shape[-1]
    if channels < 32 or channels > 4096 or channels & (channels - 1):
        raise ValueError('Fused activation kernel requires power-of-two K in [32, 4096]')
    inputs = inputs.contiguous().reshape(-1, channels)
    for transform in (signs, smoothing):
        if transform is not None and (transform.device != inputs.device or transform.shape != (channels,) or transform.dtype != torch.float32):
            raise ValueError('Transforms must be FP32 channel vectors on the input device')
    if activation_clip is not None and (activation_clip.device != inputs.device or activation_clip.shape != (2,)
                                        or activation_clip.dtype != torch.float32 or not activation_clip.is_contiguous()):
        raise ValueError('Activation clipping must be a contiguous FP32 [minimum, maximum] factor vector')
    rows = max(32, (inputs.shape[0] + 15) // 16 * 16) if int8_output else inputs.shape[0]
    packed = torch.empty((rows, channels if int8_output else channels // 2),
                         dtype=torch.int8 if int8_output else torch.uint8, device=inputs.device)
    scales = torch.empty(rows, dtype=torch.float32, device=inputs.device)
    quantize_kernel, _ = triton_kernels()
    with torch.cuda.device(inputs.device):
        quantize_kernel[(rows,)](inputs, signs, smoothing, activation_clip, packed, scales, channels,
                                           signs is not None, smoothing is not None, activation_clip is not None,
                                           inputs.shape[0], int8_output, num_warps=4,
                                           enable_fp_fusion=False)
    return packed, scales


def scaled_int4_mm(activation, weight, activation_scale, weight_scale, bias=None, dtype=torch.bfloat16):
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('Unsupported output dtype')
    if activation_scale.shape != (activation.shape[0],) or weight_scale.shape != (weight.shape[0],):
        raise ValueError('INT4 scale dimensions differ from GEMM dimensions')
    for value in (activation_scale, weight_scale, bias):
        if value is not None and (value.device != activation.device or value.dtype != torch.float32 or not value.is_contiguous()):
            raise ValueError('Scales and bias must be contiguous FP32 on the operand device')
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError('INT4 bias dimensions differ from GEMM dimensions')
    accumulator = extension().gemm(activation, weight)
    output = torch.empty(accumulator.shape, dtype=dtype, device=activation.device)
    _, epilogue = triton_kernels()
    with torch.cuda.device(activation.device):
        epilogue[((output.numel() + 255) // 256,)](accumulator, activation_scale, weight_scale, bias,
                                                output, output.numel(), output.shape[1], bias is not None, 256,
                                                enable_fp_fusion=False)
    return output


class PackedInt4Linear(nn.Module):
    def __init__(self, weight, weight_scale, bias=None, signs=None, smoothing=None, activation_clip=None):
        super().__init__()
        if weight.dtype != torch.uint8 or weight.ndim != 2:
            raise ValueError('Packed linear weights must be a uint8 matrix')
        self.in_features = weight.shape[1] * 2
        self.out_features = weight.shape[0]
        self.use_cpp_activation = False
        self.use_fast_rotation = False
        self.use_fused_epilogue = False
        if self.in_features < 32 or self.in_features > 4096 or self.in_features & (self.in_features - 1) or self.out_features % 4:
            raise ValueError('Packed Linear requires power-of-two K in [32,4096] and N divisible by 4')
        if not weight.is_contiguous():
            raise ValueError('Packed weights must be contiguous')
        for value, shape in ((weight_scale, (self.out_features,)), (bias, (self.out_features,)),
                             (signs, (self.in_features,)), (smoothing, (self.in_features,)), (activation_clip, (2,))):
            if value is not None and (value.shape != shape or value.dtype != torch.float32 or value.device != weight.device or not value.is_contiguous()):
                raise ValueError('Packed Linear scales/transforms/bias must be contiguous FP32 vectors on the weight device')
        self.register_buffer('packed_weight', weight)
        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('bias', bias)
        self.register_buffer('signs', signs)
        self.register_buffer('smoothing', smoothing)
        self.register_buffer('activation_clip', activation_clip)

    def _apply(self, function, recurse=True):
        protected = {name: getattr(self, name) for name in ('weight_scale', 'bias', 'signs', 'smoothing', 'activation_clip')}
        super()._apply(function, recurse)
        for name, value in protected.items():
            if value is not None:
                setattr(self, name, value.to(device=getattr(self, name).device, dtype=torch.float32))
        return self

    @classmethod
    @torch.no_grad()
    def from_fake(cls, layer):
        weight = layer.transform_weight()
        if layer.config.lwc:
            lower = weight.amin(-1, keepdim=True) * layer.clip_factor_w_min.sigmoid()
            upper = weight.amax(-1, keepdim=True) * layer.clip_factor_w_max.sigmoid()
            weight = weight.clamp(min=lower, max=upper)
        scale = minmax_scale(weight)
        activation_clip = (torch.cat((layer.clip_factor_a_min.sigmoid(), layer.clip_factor_a_max.sigmoid())).detach().float().clone()
                           if layer.config.lac else None)
        return cls(pack_int4(int4_quantize(weight, scale)), scale.flatten().contiguous(),
                   layer.bias.detach().float().clone() if layer.bias is not None else None,
                   layer.rotation.signs.detach().float().clone() if layer.rotation is not None else None,
                   layer.channel_scale.detach().float().clone() if layer.config.smoothing else None, activation_clip)

    def forward(self, inputs):
        if inputs.requires_grad and torch.is_grad_enabled():
            raise RuntimeError('Packed INT4 is inference-only')
        if inputs.shape[-1] != self.in_features or inputs.device != self.packed_weight.device:
            raise ValueError('Input channels/device differ from packed weights')
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else inputs.dtype
        if self.use_cpp_activation:
            output_type = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[dtype]
            operation = compiled_int4_linear if torch.compiler.is_compiling() else extension().linear
            output = operation(inputs.contiguous().reshape(-1, self.in_features), self.packed_weight,
                                        self.weight_scale, self.bias, self.smoothing, self.signs, output_type,
                                        self.activation_clip, self.use_fast_rotation, self.use_fused_epilogue)
            return output.reshape(*inputs.shape[:-1], self.out_features)
        activation, scale = quantize_activation(inputs, self.signs, self.smoothing, self.activation_clip)
        output = scaled_int4_mm(activation, self.packed_weight, scale, self.weight_scale, self.bias, dtype)
        return output.reshape(*inputs.shape[:-1], self.out_features)


def enable_fused_int4_epilogue(model):
    """Opt in to ordered FP32 row/column scaling inside the BF16 S4 GEMM."""
    count = 0
    for layer in model.modules():
        if isinstance(layer, PackedInt4Linear):
            layer.use_fused_epilogue = True
            count += 1
    if not count:
        raise ValueError('No native S4 Linear layers found')
    return count


def save_native_parameters(layers, path):
    torch.save({'backend': 'cutlass_sm80_s4_s4_s32',
                'layers': {name: {key: value.cpu() for key, value in layer.state_dict().items()}
                           for name, layer in layers.items()}}, path)


def load_native_parameters(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    return install_native_parameters(model, state)


def install_native_parameters(model, state, linear_class=PackedInt4Linear):
    if state.get('backend') != 'cutlass_sm80_s4_s4_s32' or not state.get('layers'):
        raise ValueError('Not a supported packed INT4 parameter file')
    replacements = []
    for name, tensors in state['layers'].items():
        original = model.get_submodule(name)
        if not isinstance(original, nn.Linear):
            raise ValueError(f'Expected an unwrapped Linear at {name}')
        if tuple(tensors['packed_weight'].shape) != (original.out_features, original.in_features // 2):
            raise ValueError(f'Packed weight shape differs at {name}')
        tensors = {key: value.to(original.weight.device) for key, value in tensors.items()}
        replacements.append((name, linear_class(tensors['packed_weight'], tensors['weight_scale'],
                                                    tensors.get('bias'), tensors.get('signs'), tensors.get('smoothing'),
                                                    tensors.get('activation_clip'))))
    for name, packed in replacements:
        parent_name, _, child_name = name.rpartition('.')
        setattr(model.get_submodule(parent_name), child_name, packed)
    return dict(replacements)


@contextmanager
def native_deployment(quantizer):
    restored = []
    deployed = {}
    try:
        for name, layer in quantizer.layers.items():
            device = layer.weight.device
            if device.type != 'cuda' or torch.cuda.get_device_capability(device)[0] != 8:
                raise RuntimeError('Native INT4 deployment currently requires SM80/86/89')
            packed = PackedInt4Linear.from_fake(layer)
            parent_name, _, child_name = name.rpartition('.')
            parent = quantizer.model.get_submodule(parent_name)
            layer.cpu()
            setattr(parent, child_name, packed)
            restored.append((parent, child_name, layer, device))
            deployed[name] = packed
        torch.cuda.empty_cache()
        yield deployed
    finally:
        for parent, child_name, layer, device in restored:
            setattr(parent, child_name, layer.to(device))
