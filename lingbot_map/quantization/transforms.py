from functools import lru_cache
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn


def hadamard_transform(tensor):
    dimension = tensor.shape[-1]
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError(f"FWHT requires a power-of-two dimension, got {dimension}")
    transformed = tensor
    stride = 1
    while stride < dimension:
        grouped = transformed.reshape(*tensor.shape[:-1], -1, 2, stride)
        first, second = grouped.unbind(-2)
        transformed = torch.stack((first + second, first - second), dim=-2).reshape(tensor.shape)
        stride *= 2
    return transformed / math.sqrt(dimension)


@lru_cache(maxsize=4)
def official_hadamard(reference_root=None):
    if reference_root is None:
        raise ValueError('Non-power-of-two rotations require an explicit QuantVGGT reference_root; LingBot-Map uses power-of-two widths.')
    root = Path(reference_root)
    source = root / 'evaluation/quarot/quarot_utils.py'
    if not source.is_file():
        raise FileNotFoundError(f"Non-power-of-two Hadamard needs official QuantVGGT: {source}")
    spec = importlib.util.spec_from_file_location('quantvggt_reference_hadamard', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RandomHadamard(nn.Module):
    def __init__(self, dimension, seed=0, reference_root=None, device=None):
        super().__init__()
        if dimension <= 0:
            raise ValueError("Rotation dimension must be positive")
        self.dimension = dimension
        self.reference_root = reference_root
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.register_buffer('signs', (torch.randint(0, 2, (dimension,), generator=generator) * 2 - 1).float().to(device))
        self.power_of_two = dimension & (dimension - 1) == 0
        if not self.power_of_two:
            try:
                official_hadamard(reference_root).get_hadK(dimension)
            except AssertionError as error:
                raise ValueError(f"Official Hadamard does not support dimension {dimension}") from error

    def forward(self, tensor):
        if tensor.shape[-1] != self.dimension:
            raise ValueError("Rotation dimension differs from input")
        transformed = tensor * self.signs.to(tensor)
        if self.power_of_two:
            return hadamard_transform(transformed)
        return official_hadamard(self.reference_root).matmul_hadU(transformed.contiguous())
