"""Shared geometry and model setup for quality-first windowed PTQ."""
from pathlib import Path

import torch


STREAM_KEYS = ('num_scale_frames', 'keyframe_interval', 'kv_cache_sliding_window',
               'kv_cache_scale_frames', 'camera_num_iterations', 'max_frame_num')


def window_ranges(frames, config):
    for name in (*STREAM_KEYS, 'window_size', 'overlap_keyframes'):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(frames) is not int or frames < 1:
        raise ValueError('frames must be a positive integer')
    anchors, interval = config['num_scale_frames'], config['keyframe_interval']
    if config['window_size'] < anchors:
        raise ValueError('window_size must include all scale frames')
    size = anchors + (config['window_size'] - anchors) * interval
    overlap = max(anchors, config['overlap_keyframes'] * interval)
    if overlap >= size:
        raise ValueError('Window overlap must be smaller than its input frame count')
    if size > config['max_frame_num']:
        raise ValueError('Window exceeds configured temporal RoPE range')
    result = []
    for start in range(0, frames, size - overlap):
        end = min(start + size, frames)
        result.append((start, end))
        if end == frames:
            break
    return result, overlap


def image_files(directory, start=0, frames=None):
    if start < 0 or (frames is not None and frames < 1):
        raise ValueError('Invalid input frame range')
    paths = sorted(path.resolve() for path in Path(directory).iterdir()
                   if path.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    selected = paths[start:] if frames is None else paths[start:start + frames]
    if not selected or (frames is not None and len(selected) != frames):
        raise ValueError(f'Not enough images in {directory}')
    return selected


def construct_quality_model(config):
    from lingbot_map.models.gct_stream_window import GCTStream
    return GCTStream(img_size=518, pretrained_path=None, use_sdpa=True,
                     enable_3d_rope=True, enable_point=False, use_gradient_checkpoint=False,
                     num_frame_for_scale=config['num_scale_frames'],
                     kv_cache_scale_frames=config['kv_cache_scale_frames'],
                     kv_cache_sliding_window=config['kv_cache_sliding_window'],
                     camera_num_iterations=config['camera_num_iterations'],
                     max_frame_num=config['max_frame_num'])


def load_quality_float(checkpoint, config, *, trunk_bf16=True):
    model = construct_quality_model(config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state.get('model', state), strict=True, assign=True)
    model.eval().requires_grad_(False)
    if trunk_bf16:
        model.aggregator.to(dtype=torch.bfloat16)
    return model


@torch.no_grad()
def pack_quality_model(quantizer):
    """Offline CPU/GPU export; never execute an integer kernel during packing."""
    from .native_int4 import PackedInt4Linear
    # Validate before replacing any module. Unobserved smoothing must not look
    # like a calibrated artifact just because its initial scale is one.
    for name, layer in quantizer.layers.items():
        if layer.config.smoothing and not layer.smoothing_initialized:
            raise ValueError(f'Channel smoothing has not been calibrated: {name}')
    for name, layer in quantizer.layers.items():
        packed = PackedInt4Linear.from_fake(layer)
        parent, _, child = name.rpartition('.')
        setattr(quantizer.model.get_submodule(parent), child, packed)
    return quantizer.model
