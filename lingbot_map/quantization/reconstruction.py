from contextlib import contextmanager

import torch
from torch import nn

from .core import FakeQuantLinear


def copy_tree(value, device='cpu', float_inputs=False):
    if isinstance(value, torch.Tensor):
        dtype = torch.float32 if float_inputs and value.is_floating_point() else value.dtype
        return value.detach().to(device=device, dtype=dtype).clone()
    if isinstance(value, dict):
        return {key: copy_tree(item, device, float_inputs) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_tree(item, device, float_inputs) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_tree(item, device, float_inputs) for item in value)
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(f"Cannot safely replay calibration state of type {type(value).__name__}; use SDPA dict cache")


class BlockReconstruction:
    def __init__(self, quantizer, max_samples=2, sample_stride=1):
        if max_samples < 1 or sample_stride < 1:
            raise ValueError("max_samples and sample_stride must be positive")
        self.quantizer = quantizer
        self.max_samples = max_samples
        self.sample_stride = sample_stride
        self.blocks = {}
        self.samples = {}
        for name in quantizer.layers:
            if name.startswith(('attn.', 'mlp.')):
                block_name = ''
            else:
                block_name = name.rsplit('.attn.', 1)[0] if '.attn.' in name else name.rsplit('.mlp.', 1)[0]
            block = quantizer.model.get_submodule(block_name) if block_name != name else quantizer.layers[name]
            self.blocks[block_name] = block
            self.samples[block_name] = []

    @contextmanager
    def capture(self, steps=None):
        selected_steps = None if steps is None else set(steps)
        if selected_steps is not None and (not selected_steps or min(selected_steps) < 0):
            raise ValueError('Explicit capture steps must be nonempty and nonnegative')
        handles = []
        pending = {}
        calls = {name: 0 for name in self.blocks}

        def before(name, module, args, kwargs):
            step = calls[name]
            calls[name] += 1
            selected = step % self.sample_stride == 0 if selected_steps is None else step in selected_steps
            if len(self.samples[name]) >= self.max_samples or not selected:
                return
            kwargs = dict(kwargs)
            cache = kwargs.get('kv_cache')
            if isinstance(cache, dict) and 'global_idx' in kwargs:
                index = kwargs['global_idx']
                kwargs['kv_cache'] = {key: value for key, value in cache.items()
                                      if key == '_skip_append' or key.startswith((f'k_{index}', f'v_{index}'))
                                      and key.split('_')[1] == str(index)}
            pending[name] = (copy_tree(args), copy_tree(kwargs))

        def after(name, module, args, kwargs, output):
            if name in pending:
                stored_args, stored_kwargs = pending.pop(name)
                if not isinstance(output, torch.Tensor):
                    raise TypeError("Block reconstruction requires a Tensor output")
                self.samples[name].append((stored_args, stored_kwargs, copy_tree(output)))

        for name, block in self.blocks.items():
            handles.append(block.register_forward_pre_hook(lambda module, args, kwargs, name=name: before(name, module, args, kwargs), with_kwargs=True))
            handles.append(block.register_forward_hook(lambda module, args, kwargs, output, name=name: after(name, module, args, kwargs, output), with_kwargs=True))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def optimize(self, epochs=15, lr=0.001):
        if epochs < 1 or lr <= 0:
            raise ValueError("Reconstruction requires epochs >= 1 and lr > 0")
        results = {}
        for name, block in self.blocks.items():
            samples = self.samples[name]
            if not samples:
                raise ValueError(f"No calibration samples captured for {name}")
            layers = [module for module in block.modules() if isinstance(module, FakeQuantLinear)]
            for layer in layers:
                layer.initialize_smoothing()
            smoothing_parameters = [layer.channel_scale for layer in layers if layer.config.smoothing]
            clipping_parameters = [parameter for layer in layers for parameter in layer.clipping_parameters()]
            parameters = smoothing_parameters + clipping_parameters
            if not parameters:
                continue
            device = next(block.parameters()).device
            parameter_states = [(parameter, parameter.requires_grad) for parameter in block.parameters()]
            parameter_dtypes = [(parameter, parameter.dtype) for parameter in block.parameters()]
            buffer_dtypes = [(module, key, value.dtype) for module in block.modules()
                             for key, value in module._buffers.items() if value is not None]
            layer_states = [(layer, layer.enabled, layer.calibrating, layer.training_quantization, layer.analyzing) for layer in layers]
            groups = []
            if smoothing_parameters:
                groups.append({'params': smoothing_parameters, 'lr': lr})
            if clipping_parameters:
                groups.append({'params': clipping_parameters, 'lr': lr * 10})
            optimizer = torch.optim.AdamW(groups, lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(samples), eta_min=lr * 1e-3)
            losses = []
            try:
                block.float()
                for parameter, _ in parameter_states:
                    parameter.requires_grad_(False)
                for parameter in parameters:
                    parameter.requires_grad_(True)
                targets = []
                for layer in layers:
                    layer.enabled = False
                    layer.calibrating = False
                    layer.analyzing = False
                with torch.no_grad(), torch.autocast(device.type, enabled=False):
                    for args, kwargs, _ in samples:
                        targets.append(block(*copy_tree(args, device, True), **copy_tree(kwargs, device, True)).detach().cpu())
                for layer in layers:
                    layer.enabled = True
                    layer.calibrating = False
                    layer.training_quantization = True
                    layer.analyzing = False
                with torch.enable_grad(), torch.autocast(device.type, enabled=False):
                    for epoch in range(epochs):
                        epoch_losses = []
                        for (args, kwargs, _), target in zip(samples, targets):
                            optimizer.zero_grad(set_to_none=True)
                            output = block(*copy_tree(args, device, True), **copy_tree(kwargs, device, True))
                            loss = nn.functional.mse_loss(output.float(), target.to(device).float())
                            if not torch.isfinite(loss):
                                raise ValueError(f"Non-finite reconstruction loss in {name}")
                            loss.backward()
                            optimizer.step()
                            with torch.no_grad():
                                for parameter in smoothing_parameters:
                                    parameter.clamp_(min=1e-5)
                            scheduler.step()
                            epoch_losses.append(loss.item())
                        losses.append(sum(epoch_losses) / len(epoch_losses))
            finally:
                for parameter, dtype in parameter_dtypes:
                    parameter.data = parameter.data.to(dtype)
                for module, key, dtype in buffer_dtypes:
                    if module._buffers[key] is not None:
                        module._buffers[key] = module._buffers[key].to(dtype)
                for parameter, state in parameter_states:
                    parameter.requires_grad_(state)
                    parameter.grad = None
                for layer, enabled, calibrating, training, analyzing in layer_states:
                    layer.enabled, layer.calibrating = enabled, calibrating
                    layer.training_quantization, layer.analyzing = training, analyzing
                    layer.invalidate()
            results[name] = {'samples': len(samples), 'epoch_mse': losses, 'lr': lr,
                             'epochs': epochs, 'objective': 'block_output_mse', 'optimizer': 'AdamW',
                             'weight_decay': 0.01, 'schedule': 'cosine',
                             'clipping_lr': lr * 10 if clipping_parameters else None,
                             'initial_scale': ('official_after_smoothfactor_init_ema_sqrt_over_original_weight_max_sqrt'
                                               if smoothing_parameters else 'no_channel_smoothing'),
                             'teacher': 'original_block_float32_on_captured_stream_state'}
        return results
