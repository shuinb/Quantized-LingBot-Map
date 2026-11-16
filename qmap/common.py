"""Shared I/O, calibration placement and process memory helpers."""
import json
import subprocess
import numpy as np
import torch

class CPUCachePlacement:
    """Move only the current attention layer's original KV onto the GPU."""

    def __init__(self, attentions, device):
        self.device = device
        self.active = {}
        self.handles = []
        for attention in attentions:
            self.handles.append(attention.register_forward_pre_hook(self.before, with_kwargs=True))
            self.handles.append(attention.register_forward_hook(self.after, with_kwargs=True))

    def before(self, module, args, kwargs):
        cache = kwargs.get('kv_cache')
        if cache is None:
            return
        index = kwargs.get('global_idx', 0)
        keys = [f'{kind}_{index}{suffix}' for kind in ('k', 'v') for suffix in ('', '_special')]
        previous = {key: cache.get(key) for key in keys}
        self.active[module] = (cache, previous)
        for key, value in previous.items():
            if torch.is_tensor(value):
                cache[key] = value.to(self.device, non_blocking=True)

    def after(self, module, args, kwargs, output):
        state = self.active.pop(module, None)
        if state is None:
            return
        cache, previous = state
        for key, old in previous.items():
            if cache.get('_skip_append', False):
                cache[key] = old
            elif torch.is_tensor(cache.get(key)):
                value = cache[key]
                host = torch.empty_like(value, device='cpu', pin_memory=True)
                host.copy_(value)
                cache[key] = host


def depth_chunk_one(module, args, kwargs):
    # DPT has a supported frame-chunking API; temporal attention stays intact.
    return args, {**kwargs, 'frames_chunk_size': 1}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prediction_is_finite(prediction):
    """Check every floating output; use NumPy for CPU dtypes it supports."""
    for value in prediction.values():
        if not value.is_floating_point():
            continue
        if value.device.type == 'cpu' and value.dtype in (torch.float16, torch.float32, torch.float64):
            finite = bool(np.isfinite(value.detach().numpy()).all())
        else:
            finite = bool(torch.isfinite(value).all())
        if not finite:
            return False
    return True


def crop_geometry(original_hw, image_size=518, max_height=294, patch_size=14):
    height, width = original_hw
    new_height = round(height * image_size / width / patch_size) * patch_size
    first_crop = max((new_height - image_size) // 2, 0)
    intermediate_height = min(new_height, image_size)
    second_crop = max((intermediate_height - max_height) // 2, 0)
    return new_height, min(intermediate_height, max_height), first_crop + second_crop


def sampled_memory(pid):
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
    except subprocess.TimeoutExpired:
        return None
    for line in result.stdout.splitlines():
        fields = line.split(',')
        if len(fields) == 2 and fields[0].strip() == str(pid):
            try:
                return float(fields[1].strip())
            except ValueError:
                return None
    return None


def memory_sampler(gpu):
    try:
        import pynvml
    except ImportError:
        return sampled_memory, 'nvidia-smi fallback'
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    except pynvml.NVMLError:
        return sampled_memory, 'nvidia-smi fallback'

    def sample(pid):
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for process in processes:
                if process.pid == pid and isinstance(process.usedGpuMemory, int):
                    return process.usedGpuMemory / 2**20
        except pynvml.NVMLError:
            return None
        return None

    return sample, 'NVML physical-device process accounting'
