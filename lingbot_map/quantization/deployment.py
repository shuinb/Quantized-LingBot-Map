from pathlib import Path

import torch

from .native_int4 import PackedInt4Linear, extension, install_native_parameters


def enable_bf16_camera(model):
    """Opt-in camera precision tradeoff; retain FP32 norms and pose constants.

    Store Linear operands in the same BF16 dtype used by CUDA autocast so a
    persistent FP32 copy is not retained alongside autocast's BF16 operands.
    Apply only to a freshly loaded inference model before creating caches.
    """
    enable_camera_compute_dtype(model, torch.bfloat16)


def enable_camera_compute_dtype(model, dtype):
    """Use FP16/BF16 Linear operands with matching camera autocast; keep FP32 norms."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError('Reduced camera Linear precision must be BF16 or FP16')
    if model.camera_head is None:
        raise ValueError('Camera precision conversion requires a camera head')
    for module in model.camera_head.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(dtype=dtype)
    model.camera_autocast_dtype = dtype


def construct_model(stream, attention_backend='sdpa'):
    from lingbot_map.models.gct_stream import GCTStream
    return GCTStream(img_size=518, pretrained_path=None, use_sdpa=attention_backend == 'sdpa', enable_3d_rope=True,
                     kv_cache_sliding_window=stream['kv_cache_sliding_window'],
                     kv_cache_scale_frames=stream['kv_cache_scale_frames'],
                     num_frame_for_scale=stream['num_scale_frames'],
                     camera_num_iterations=stream.get('camera_num_iterations', 4),
                     max_frame_num=stream.get('max_frame_num', 1024), use_gradient_checkpoint=False)


def export_deployment(model, stream_config, destination):
    layers = [name for name, module in model.named_modules() if isinstance(module, PackedInt4Linear)]
    if not layers:
        raise ValueError('Deployment requires at least one packed INT4 Linear')
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({'format': 'lingbot_native_int4', 'version': 1, 'stream_config': dict(stream_config),
                'packed_layers': layers, 'state_dict': state}, destination)
    return {'path': str(Path(destination).resolve()), 'file_bytes': Path(destination).stat().st_size,
            'packed_layers': len(layers), 'tensor_bytes': sum(tensor.numel() * tensor.element_size() for tensor in state.values()),
            'requires_original_checkpoint_at_runtime': False,
            'precision': 'packed W4A4 aggregator Linear; original-precision attention, heads and KV'}


def load_deployment(path, device='cpu', model_factory=None, stream_overrides=None, attention_backend='sdpa',
                    linear_backend='native', teacher_checkpoint=None):
    """Load packed weights; execute under no_grad and CUDA BF16 autocast."""
    artifact = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if artifact.get('format') != 'lingbot_native_int4' or artifact.get('version') != 1 or not artifact.get('packed_layers'):
        raise ValueError('Unsupported INT4 deployment checkpoint')
    if attention_backend not in ('sdpa', 'flashinfer'):
        raise ValueError(f'Unsupported attention backend: {attention_backend}')
    if linear_backend not in ('native', 'int8', 'int8-fused'):
        raise ValueError(f'Unsupported linear backend: {linear_backend}')
    device = torch.device(device)
    if teacher_checkpoint is None and linear_backend == 'native' and device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] != 8:
        raise RuntimeError('Native INT4 deployment currently requires SM80/86/89')
    stream = dict(artifact['stream_config'])
    if stream_overrides is not None and not isinstance(stream_overrides, dict):
        raise ValueError('Streaming overrides must be a JSON object')
    allowed = {'num_scale_frames', 'keyframe_interval', 'kv_cache_sliding_window',
               'kv_cache_scale_frames', 'camera_num_iterations', 'max_frame_num'}
    for name, value in (stream_overrides or {}).items():
        if name not in allowed or type(value) is not int or value < 1:
            raise ValueError(f'Unsupported streaming override: {name}={value!r}')
        stream[name] = value
    if model_factory is not None:
        model = model_factory(stream)
    elif attention_backend == 'sdpa':
        model = construct_model(stream)
    else:
        model = construct_model(stream, attention_backend=attention_backend)
    if teacher_checkpoint is not None:
        checkpoint = torch.load(teacher_checkpoint, map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(checkpoint.get('model', checkpoint), strict=True, assign=True)
        model.aggregator.to(dtype=torch.bfloat16)
        return model.eval().to(device), stream
    state = artifact['state_dict']
    layers = {name: {key.removeprefix(name + '.'): value for key, value in state.items() if key.startswith(name + '.')}
              for name in artifact['packed_layers']}
    parameters = {'backend': 'cutlass_sm80_s4_s4_s32', 'layers': layers}
    if linear_backend == 'native':
        install_native_parameters(model, parameters)
    else:
        from .int8_compat import PackedW4A4Int8Linear
        install_native_parameters(model, parameters, linear_class=PackedW4A4Int8Linear)
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().to(device)
    for module in model.modules():
        if isinstance(module, PackedInt4Linear):
            module.use_cpp_activation = linear_backend == 'native'
            module.use_fast_rotation = linear_backend == 'native'
            if linear_backend != 'native':
                module.use_fused_int8 = linear_backend == 'int8-fused'
    if device.type == 'cuda' and linear_backend == 'native':
        extension()
    return model, stream
