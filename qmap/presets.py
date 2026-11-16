"""Measured defaults; BF16 remains the original floating-point reference."""


def inference_defaults(method, backend):
    if method not in ('bf16', 'w4a4', 'dsfq'):
        raise ValueError(f'Unknown method: {method}')
    quantized = method != 'bf16'
    native = backend == 'native'
    return {
        'kv_format': 'fp4' if quantized else 'original',
        'store_kv_compute_dtype': not quantized,
        'camera_graph': True,
        'compile_stateless': quantized,
        'defer_kv_reserve': quantized,
        'memory_budget_gb': (5.5 if native else 5.2) if quantized else None,
        'prefill_mlp_chunk_tokens': 783 if quantized else 0,
        'prefill_frame_batch_size': 1 if quantized else 0,
        'fused_int4_epilogue': quantized and native,
    }


def validate_deployment(layers, method):
    """Avoid labeling an uncalibrated artifact as the calibrated comparison."""
    if not layers or any(layer.activation_clip is None for layer in layers):
        raise ValueError('Use a calibrated artifact exported by `qmap calibrate`.')
    if method == 'w4a4' and any(layer.signs is not None or layer.smoothing is not None for layer in layers):
        raise ValueError('--method w4a4 requires direct calibrated weights without rotation or smoothing')
    if method == 'dsfq' and any(layer.signs is None or layer.smoothing is None for layer in layers):
        raise ValueError('--method dsfq requires calibrated rotation and smoothing')
