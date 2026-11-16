from contextlib import contextmanager
from fnmatch import fnmatchcase

import torch
from torch import nn
from torch.nn import functional as functional

from .config import QuantizationConfig
from .metrics import ErrorMetrics
from .transforms import RandomHadamard


def minmax_scale(tensor, dim=-1):
    if not tensor.is_floating_point() or tensor.shape[dim] == 0:
        raise ValueError("Quantization needs a nonempty floating-point channel dimension")
    maximum = tensor.float().abs().amax(dim=dim, keepdim=True)
    scale = (maximum / 7).clamp_min(torch.finfo(torch.float32).tiny)
    return torch.where(maximum == 0, torch.ones_like(scale), scale)


def int4_quantize(tensor, scale):
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Scale must be finite and positive")
    return torch.round(tensor.float() / scale).clamp(-8, 7).to(torch.int8)


def fake_quantize(tensor, scale=None, ste=False):
    scale = minmax_scale(tensor) if scale is None else scale
    if ste:
        normalized = tensor.float() / scale
        rounded = normalized + (normalized.round() - normalized).detach()
        return (rounded.clamp(-8, 7) * scale).to(tensor.dtype)
    return (int4_quantize(tensor, scale).float() * scale).to(tensor.dtype)


class ChannelObserver(nn.Module):
    def __init__(self, channels, momentum=0.95):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("ema", torch.zeros(channels, dtype=torch.float32))
        self.register_buffer("minimum", torch.full((channels,), float("inf")))
        self.register_buffer("maximum", torch.full((channels,), -float("inf")))
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def forward(self, tensor):
        values = tensor.detach().float()
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite calibration input")
        flattened = values.reshape(-1, values.shape[-1])
        current = values.abs() if values.ndim == 1 else values.abs().amax(dim=-2)
        current = current.reshape(-1, values.shape[-1]).mean(dim=0)
        if self.calls.item() == 0:
            self.ema.copy_(current)
        else:
            self.ema.mul_(self.momentum).add_(current, alpha=1 - self.momentum)
        self.minimum.copy_(torch.minimum(self.minimum, flattened.amin(dim=0)))
        self.maximum.copy_(torch.maximum(self.maximum, flattened.amax(dim=0)))
        self.calls.add_(1)

    def summary(self):
        if self.calls.item() == 0:
            return {"calls": 0, "abs_max": None, "ema_max": None}
        return {"calls": self.calls.item(), "abs_max": self.maximum.abs().maximum(self.minimum.abs()).max().item(),
                "ema_max": self.ema.max().item()}


class FakeQuantLinear(nn.Module):
    def __init__(self, linear, config, seed=None):
        super().__init__()
        self.linear = linear
        self.config = config
        self.enabled = True
        self.calibrating = False
        self.analyzing = False
        self.training_quantization = False
        self.transform_only = False
        self.cache_weight = True
        self.smoothing_initialized = False
        self.rotation = RandomHadamard(linear.in_features, config.seed if seed is None else seed,
                                       config.reference_root, linear.weight.device) if config.rotation else None
        self.channel_scale = nn.Parameter(torch.ones(linear.in_features, device=linear.weight.device), requires_grad=False)
        if config.lwc:
            self.clip_factor_w_max = nn.Parameter(torch.full((linear.out_features, 1), 4., device=linear.weight.device), requires_grad=False)
            self.clip_factor_w_min = nn.Parameter(torch.full((linear.out_features, 1), 4., device=linear.weight.device), requires_grad=False)
        if config.lac:
            self.clip_factor_a_max = nn.Parameter(torch.full((1,), 4., device=linear.weight.device), requires_grad=False)
            self.clip_factor_a_min = nn.Parameter(torch.full((1,), 4., device=linear.weight.device), requires_grad=False)
        self.observer = ChannelObserver(linear.in_features, config.momentum).to(linear.weight.device)
        self.errors = ErrorMetrics()
        self.last_error = ErrorMetrics()
        self.register_buffer("prepared_weight", None, persistent=False)
        self._weight_version = None

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

    def transform_input(self, tensor):
        tensor = self.rotation(tensor.float()) if self.rotation is not None else tensor.float()
        return tensor / self.channel_scale if self.config.smoothing else tensor

    def rotation_input(self, tensor):
        return self.rotation(tensor.float()) if self.rotation is not None else tensor.float()

    def transform_weight(self):
        weight = self.rotation(self.weight.float()) if self.rotation is not None else self.weight.float()
        return weight * self.channel_scale if self.config.smoothing else weight

    def _apply(self, function, recurse=True):
        super()._apply(function, recurse)
        self.channel_scale.data = self.channel_scale.data.float()
        for parameter in self.clipping_parameters():
            parameter.data = parameter.data.float()
        for name in ('ema', 'minimum', 'maximum'):
            setattr(self.observer, name, getattr(self.observer, name).float())
        self.invalidate()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        self.invalidate()
        super()._load_from_state_dict(*args, **kwargs)

    def version(self):
        return (self.weight._version, self.channel_scale._version,
                self.rotation.signs._version if self.rotation is not None else None,
                tuple(parameter._version for parameter in self.clipping_parameters()))

    def clipping_parameters(self):
        return [getattr(self, name) for name in ('clip_factor_w_max', 'clip_factor_w_min',
                                                'clip_factor_a_max', 'clip_factor_a_min') if hasattr(self, name)]

    def quantized_weight(self, ste=False):
        weight = self.transform_weight()
        if self.config.lwc:
            lower = weight.amin(-1, keepdim=True) * self.clip_factor_w_min.sigmoid()
            upper = weight.amax(-1, keepdim=True) * self.clip_factor_w_max.sigmoid()
            weight = weight.clamp(min=lower, max=upper)
        return fake_quantize(weight, ste=ste)

    def quantized_activation(self, tensor, ste=False):
        scale = None
        if self.config.lac:
            lower = tensor.amin(-1, keepdim=True).clamp(max=0) * self.clip_factor_a_min.sigmoid()
            upper = tensor.amax(-1, keepdim=True).clamp(min=0) * self.clip_factor_a_max.sigmoid()
            maximum = torch.maximum(lower.abs(), upper)
            scale = torch.where(maximum == 0, torch.ones_like(maximum),
                                (maximum / 7).clamp_min(torch.finfo(torch.float32).tiny))
        return fake_quantize(tensor, scale=scale, ste=ste)

    def invalidate(self):
        self.prepared_weight = None
        self._weight_version = None

    @torch.no_grad()
    def initialize_smoothing(self):
        if not self.config.smoothing or self.smoothing_initialized:
            return
        if self.observer.calls.item() == 0:
            raise ValueError('Channel smoothing requires observed calibration activations')
        minimum = torch.finfo(torch.float32).tiny
        activation_max = self.observer.ema.clamp_min(minimum)
        weight_max = self.weight.float().abs().amax(dim=0).clamp_min(minimum)
        self.channel_scale.copy_((activation_max.sqrt() / weight_max.sqrt()).clamp_min(1e-5))
        self.smoothing_initialized = True
        self.invalidate()

    @torch.no_grad()
    def prepare(self):
        self.prepared_weight = self.quantized_weight().to(self.weight.dtype)
        self._weight_version = self.version()

    def forward(self, tensor):
        if self.calibrating:
            self.observer(self.rotation_input(tensor))
            return self.linear(tensor)
        if not self.enabled:
            return self.linear(tensor)
        transformed = self.transform_input(tensor)
        if self.transform_only:
            return functional.linear(transformed.to(tensor.dtype), self.transform_weight().to(tensor.dtype),
                                     self.bias.to(tensor.dtype) if self.bias is not None else None)
        if self.training_quantization or not self.cache_weight:
            weight = self.quantized_weight(ste=self.training_quantization)
        else:
            if self.prepared_weight is None or self._weight_version != self.version():
                self.prepare()
            weight = self.prepared_weight
        activation = self.quantized_activation(transformed, ste=self.training_quantization).to(tensor.dtype)
        output = functional.linear(activation, weight.to(activation.dtype), self.bias.to(activation.dtype) if self.bias is not None else None)
        if self.analyzing:
            with torch.no_grad():
                reference = self.linear(tensor)
                self.errors.update(reference, output)
                self.last_error = ErrorMetrics()
                self.last_error.update(reference, output)
        return output


class Quantizer:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or QuantizationConfig()
        if any(isinstance(module, FakeQuantLinear) for module in model.modules()):
            raise ValueError("Model is already quantized")
        selected = [(name, module) for name, module in model.named_modules()
                    if name and isinstance(module, nn.Linear)
                    and any(fnmatchcase(name, pattern) for pattern in self.config.include)
                    and not any(fnmatchcase(name, pattern) for pattern in self.config.exclude)]
        if not selected:
            raise ValueError("No Linear matches quantization include/exclude patterns")
        self.layers = {}
        replacements = [(name, FakeQuantLinear(module, self.config, self.config.seed + index))
                        for index, (name, module) in enumerate(selected)]
        for name, replacement in replacements:
            parent_name, _, child_name = name.rpartition(".")
            setattr(model.get_submodule(parent_name), child_name, replacement)
            self.layers[name] = replacement

    def set_enabled(self, enabled):
        for layer in self.layers.values():
            layer.enabled = enabled

    @contextmanager
    def calibration(self):
        previous = [layer.calibrating for layer in self.layers.values()]
        for layer in self.layers.values():
            layer.calibrating = True
        try:
            yield self
        finally:
            for layer, state in zip(self.layers.values(), previous):
                layer.calibrating = state

    def prepare(self):
        for layer in self.layers.values():
            layer.prepare()

    def analyze(self, enabled=True, reset=True):
        for layer in self.layers.values():
            layer.analyzing = enabled
            if reset:
                layer.errors = ErrorMetrics()
                layer.last_error = ErrorMetrics()

    def reset_observers(self):
        for layer in self.layers.values():
            layer.observer = ChannelObserver(layer.in_features, self.config.momentum).to(layer.weight.device)

    def report(self):
        return {name: {"weight_shape": list(layer.weight.shape), "dtype": str(layer.weight.dtype),
                       "observer": layer.observer.summary(), "local_error": layer.errors.summary(),
                       "last_call_error": layer.last_error.summary()}
                for name, layer in self.layers.items()}

    def remove(self):
        for name, layer in self.layers.items():
            parent_name, _, child_name = name.rpartition(".")
            setattr(self.model.get_submodule(parent_name), child_name, layer.linear)

    def save_parameters(self, path):
        parameters = {}
        for name, layer in self.layers.items():
            parameters[name] = {'channel_scale': layer.channel_scale.detach().cpu()}
            for key in ('clip_factor_w_max', 'clip_factor_w_min', 'clip_factor_a_max', 'clip_factor_a_min'):
                if hasattr(layer, key):
                    parameters[name][key] = getattr(layer, key).detach().cpu()
            if layer.rotation is not None:
                parameters[name]['signs'] = layer.rotation.signs.detach().cpu()
        torch.save({'config': self.config.to_dict(), 'parameters': parameters}, path)

    @classmethod
    def from_parameters(cls, model, path, prepare=True):
        state = torch.load(path, map_location='cpu', weights_only=True)
        quantizer = cls(model, QuantizationConfig(**state['config']))
        if set(quantizer.layers) != set(state['parameters']):
            quantizer.remove()
            raise ValueError('Saved quantization parameters do not match model Linear paths')
        with torch.no_grad():
            for name, layer in quantizer.layers.items():
                layer.channel_scale.copy_(state['parameters'][name]['channel_scale'])
                layer.smoothing_initialized = True
                for key in ('clip_factor_w_max', 'clip_factor_w_min', 'clip_factor_a_max', 'clip_factor_a_min'):
                    if hasattr(layer, key):
                        getattr(layer, key).copy_(state['parameters'][name][key])
                if layer.rotation is not None:
                    layer.rotation.signs.copy_(state['parameters'][name]['signs'])
        if prepare:
            quantizer.prepare()
        return quantizer
