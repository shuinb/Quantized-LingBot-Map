"""SageAttention 2.2.0 with its FP16 value operand prepared during KV decode."""
import torch


def sage_attention_fp16_values(query, key, value):
    """Use the same QK quantizer and PV kernel as the existing Sage path.

    Values must already equal the original decoded operand converted to FP16.
    Only HND, equal query/KV heads, 64/128 channels, noncausal attention,
    per-warp QK quantization, key smoothing and FP16+FP32 PV are supported.
    """
    if (query.ndim != 4 or key.ndim != 4 or value.shape != key.shape
            or query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]
            or query.shape[-1] not in (64, 128)):
        raise ValueError('Prepared Sage values require HND, equal heads and head dimension 64/128')
    if (not query.is_cuda or query.device != key.device or query.device != value.device
            or query.dtype not in (torch.bfloat16, torch.float16)
            or key.dtype != query.dtype or value.dtype != torch.float16
            or any(tensor.stride(-1) != 1 for tensor in (query, key, value))):
        raise ValueError('Prepared Sage values require matching CUDA Q/K and FP16 V, contiguous last dimension')
    from sageattention.quant import per_warp_int8
    from sageattention import sm80_compile
    dimension = query.shape[-1]
    with torch.cuda.device(query.device):
        mean = key.mean(dim=2, keepdim=True)
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(
            query, key, mean, tensor_layout='HND', BLKQ=128,
            WARPQ=16 if dimension == 128 else 32, BLKK=64)
        output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        sm80_compile.qk_int8_sv_f16_accum_f16_attn_inst_buf(
            q_int8, k_int8, value, output, q_scale, k_scale,
            1, 0, 2, dimension ** -0.5, 0)
    return output
