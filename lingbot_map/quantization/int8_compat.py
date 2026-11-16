from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from .core import int4_quantize, minmax_scale
from .native_int4 import PackedInt4Linear, pack_int4, quantize_activation, triton_kernels, unpack_int4
from .transforms import hadamard_transform


def integer_mm(activation, weight):
    if activation.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError('Integer compatibility GEMM requires int8 containers for signed INT4 codes')
    if activation.ndim != 2 or weight.ndim != 2 or activation.shape[1] != weight.shape[1]:
        raise ValueError('Expected matching [M,K] activation and [N,K] weight matrices')
    if activation.device != weight.device:
        raise ValueError('Integer GEMM operands must share a device')
    if activation.device.type == 'cpu':
        return activation.to(torch.int32) @ weight.to(torch.int32).T
    if activation.device.type != 'cuda' or activation.shape[1] % 8:
        raise ValueError('CUDA compatibility GEMM requires K divisible by 8')
    rows, columns = activation.shape[0], weight.shape[0]
    padded_rows = max(32, (rows + 15) // 16 * 16)
    padded_columns = max(16, (columns + 7) // 8 * 8)
    padded_activation = (functional.pad(activation, (0, 0, 0, padded_rows - rows))
                         if padded_rows != rows else activation).contiguous()
    padded_weight = (functional.pad(weight, (0, 0, 0, padded_columns - columns))
                     if padded_columns != columns else weight).contiguous()
    accumulator = torch._int_mm(padded_activation, padded_weight.T)
    return accumulator[:rows, :columns].contiguous()


def cpu_quantize_activation(inputs, signs, smoothing, clipping):
    values = inputs.float().reshape(-1, inputs.shape[-1])
    if signs is not None:
        values = hadamard_transform(values * signs)
    if smoothing is not None:
        values = values / smoothing
    scale = minmax_scale(values)
    if clipping is not None:
        lower = values.amin(-1, keepdim=True).clamp(max=0) * clipping[0]
        upper = values.amax(-1, keepdim=True).clamp(min=0) * clipping[1]
        maximum = torch.maximum(lower.abs(), upper)
        scale = torch.where(maximum == 0, torch.ones_like(maximum),
                            (maximum / 7).clamp_min(torch.finfo(torch.float32).tiny))
    return pack_int4(int4_quantize(values, scale)), scale.flatten()


@lru_cache(maxsize=1)
@torch.compiler.assume_constant_result
def unpack_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def unpack_s4(packed, output, SIZE: tl.constexpr, BLOCK: tl.constexpr):
        indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        byte = tl.load(packed + indices // 2, indices < SIZE, 0).to(tl.int32)
        code = tl.where(indices % 2 == 0, byte & 15, byte >> 4)
        tl.store(output + indices, tl.where(code >= 8, code - 16, code).to(tl.int8), indices < SIZE)

    return unpack_s4


def unpack_int4_fused(packed):
    if not packed.is_cuda or packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise ValueError('Fused unpack requires contiguous CUDA packed INT4 weights')
    output = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), device=packed.device, dtype=torch.int8)
    with torch.cuda.device(packed.device):
        unpack_kernel()[((output.numel() + 1023) // 1024,)](packed, output, output.numel(), 1024)
    return output


class PackedW4A4Int8Linear(PackedInt4Linear):
    use_fused_int8 = False

    def forward(self, inputs):
        if inputs.requires_grad and torch.is_grad_enabled():
            raise RuntimeError('Packed W4A4 integer compatibility execution is inference-only')
        if inputs.shape[-1] != self.in_features or inputs.device != self.packed_weight.device:
            raise ValueError('Input channels/device differ from packed weights')
        dtype = (torch.get_autocast_dtype(inputs.device.type)
                 if torch.is_autocast_enabled(inputs.device.type) else inputs.dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('Expected floating-point input/output dtype')
        if inputs.numel() == 0:
            return inputs.new_empty((*inputs.shape[:-1], self.out_features), dtype=dtype)
        if inputs.is_cuda and self.use_fused_int8:
            activation, activation_scale = quantize_activation(inputs, self.signs, self.smoothing,
                                                              self.activation_clip, int8_output=True)
            accumulator = integer_mm(activation, unpack_int4_fused(self.packed_weight))
            accumulator = accumulator[:inputs.numel() // self.in_features]
        else:
            quantize = quantize_activation if inputs.is_cuda else cpu_quantize_activation
            activation, activation_scale = quantize(inputs, self.signs, self.smoothing, self.activation_clip)
            accumulator = integer_mm(unpack_int4(activation), unpack_int4(self.packed_weight))
        if inputs.is_cuda:
            output = torch.empty(accumulator.shape, dtype=dtype, device=inputs.device)
            _, epilogue = triton_kernels()
            with torch.cuda.device(inputs.device):
                epilogue[((output.numel() + 255) // 256,)](
                    accumulator, activation_scale, self.weight_scale, self.bias, output,
                    output.numel(), self.out_features, self.bias is not None, 256, enable_fp_fusion=False)
        else:
            output = accumulator.float() * activation_scale[:, None]
            output = output * self.weight_scale[None]
            if self.bias is not None:
                output = output + self.bias
            output = output.to(dtype)
        return output.reshape(*inputs.shape[:-1], self.out_features)


def install_w4a4_int8(model, deployment):
    artifact = torch.load(deployment, map_location='cpu', weights_only=True, mmap=True)
    if artifact.get('format') != 'lingbot_native_int4' or artifact.get('version') != 1 or not artifact.get('packed_layers'):
        raise ValueError('Require an existing packed W4A4 deployment checkpoint')
    state = artifact['state_dict']
    replacements = []
    for name in artifact['packed_layers']:
        original = model.get_submodule(name)
        if not isinstance(original, nn.Linear):
            raise ValueError(f'Expected an unwrapped Linear at {name}')
        tensors = {key.removeprefix(name + '.'): value.to(original.weight.device)
                   for key, value in state.items() if key.startswith(name + '.')}
        if tuple(tensors['packed_weight'].shape) != (original.out_features, original.in_features // 2):
            raise ValueError(f'Packed weight shape differs at {name}')
        replacement = PackedW4A4Int8Linear(tensors['packed_weight'], tensors['weight_scale'], tensors.get('bias'),
                                         tensors.get('signs'), tensors.get('smoothing'), tensors.get('activation_clip')).eval()
        replacements.append((name, replacement))
    for name, replacement in replacements:
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, replacement)
    return {'mode': 'w4a4_int8_compat', 'deployment': str(Path(deployment).resolve()),
            'quantized_layers': len(replacements), 'weight_bits': 4, 'activation_bits': 4,
            'storage': 'two signed INT4 codes per uint8 byte',
            'compute': 'unpack S4 into int8 containers, CUDA torch._int_mm S8xS8->S32; FP32 scales/bias; output follows autocast/input dtype',
            'native_int4_instructions': False, 'fake_quant': False,
            'packed_weight_bytes': sum(layer.packed_weight.numel() for _, layer in replacements),
            'rotation_layers': sum(layer.signs is not None for _, layer in replacements),
            'smoothing_layers': sum(layer.smoothing is not None for _, layer in replacements),
            'activation_clipping_layers': sum(layer.activation_clip is not None for _, layer in replacements),
            'unquantized': 'DINO, norms, attention/RoPE/softmax, geometry heads and BF16 KV',
            'layer_names': [name for name, _ in replacements]}
