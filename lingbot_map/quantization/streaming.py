from contextlib import contextmanager
import time

import torch


def tensor_cache_summary(cache):
    if isinstance(cache, dict):
        return {str(key): tensor_cache_summary(value) for key, value in cache.items() if value is not None}
    if isinstance(cache, (list, tuple)):
        return [tensor_cache_summary(value) for value in cache]
    if isinstance(cache, torch.Tensor):
        return {'shape': list(cache.shape), 'dtype': str(cache.dtype), 'bytes': cache.numel() * cache.element_size()}
    if cache is None or isinstance(cache, (str, bool, int, float)):
        return cache
    return {'type': type(cache).__name__}


def cache_summary(model):
    summary = {}
    for name in ('aggregator', 'camera_head'):
        component = getattr(model, name, None)
        if component is None:
            continue
        entry = {'cache': tensor_cache_summary(getattr(component, 'kv_cache', None))}
        for attribute in ('total_frames_processed', 'frame_idx'):
            if hasattr(component, attribute):
                entry[attribute] = getattr(component, attribute)
        manager = getattr(component, 'kv_cache_manager', None)
        if manager is not None:
            entry['paged'] = {'type': type(manager).__name__, 'dtype': str(manager.dtype)}
            if hasattr(manager, 'get_cache_stats'):
                entry['paged']['occupancy'] = manager.get_cache_stats()
            for attribute in ('total_frames', 'num_frames', 'frame_count'):
                value = getattr(manager, attribute, None)
                if isinstance(value, (int, float)):
                    entry['paged'][attribute] = value
        summary[name] = entry
    return summary


@contextmanager
def trace_stream(model, quantizer=None, record_stride=1):
    if record_stride < 1:
        raise ValueError('Trace record_stride must be positive')
    records = []
    steps = 0
    recorded_current = False

    def after_aggregator(module, args, kwargs, output):
        nonlocal steps, recorded_current
        recorded_current = steps % record_stride == 0
        step = steps
        steps += 1
        if not recorded_current:
            return
        images = args[0] if args else kwargs['images']
        records.append({'step': step, 'frames_in_call': images.shape[1],
                        'state': cache_summary(model),
                        'layers': quantizer.report() if quantizer else {}})

    def after_camera(module, args, output):
        if records and recorded_current:
            records[-1]['state'] = cache_summary(model)

    handles = [model.aggregator.register_forward_hook(after_aggregator, with_kwargs=True)]
    if getattr(model, 'camera_head', None) is not None:
        handles.append(model.camera_head.register_forward_hook(after_camera))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def run_stream(model, images, scale_frames=1, keyframe_interval=1, independent_frames=False, return_images=True):
    if scale_frames < 1 or keyframe_interval < 1:
        raise ValueError('scale_frames and keyframe_interval must be >= 1')
    sequence_length = images.shape[-4]
    if sequence_length < 1:
        raise ValueError('Empty streaming sequence')
    options = {} if return_images else {'return_images': False}
    if not independent_frames:
        return model.inference_streaming(images, num_scale_frames=scale_frames,
                                         keyframe_interval=keyframe_interval, output_device=torch.device('cpu'), **options)
    outputs = []
    for index in range(sequence_length):
        frame = images[index:index + 1] if images.ndim == 4 else images[:, index:index + 1]
        outputs.append(model.inference_streaming(frame, num_scale_frames=1,
                                                keyframe_interval=1, output_device=torch.device('cpu'), **options))
    return {key: torch.cat([output[key] for output in outputs], dim=1) for key in outputs[0]}


@torch.no_grad()
def measure_stream(model, images, scale_frames=1, keyframe_interval=1, warmup=1, repeats=1, native_int4=False, return_images=True):
    if warmup < 0 or repeats < 1:
        raise ValueError('warmup must be >= 0 and repeats >= 1')
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    for _ in range(warmup):
        run_stream(model, images, scale_frames, keyframe_interval, return_images=return_images)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        run_stream(model, images, scale_frames, keyframe_interval, return_images=return_images)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    duration = sum(durations) / len(durations)
    frames = images.shape[-4]
    return {'seconds_per_sequence': duration, 'latency_ms_per_frame': duration * 1000 / frames,
            'sequence_durations_seconds': durations,
            'fps': frames / duration, 'repeats': repeats, 'warmup': warmup,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            'scope': 'original streaming entry, CPU output offload and cache lifecycle included; no analysis hooks',
            'native_int4': native_int4}
