from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class QuantizationConfig:
    mode: str = "naive"
    include: tuple[str, ...] = (
        "aggregator.frame_blocks.*.attn.qkv", "aggregator.frame_blocks.*.attn.proj",
        "aggregator.frame_blocks.*.mlp.fc1", "aggregator.frame_blocks.*.mlp.fc2",
        "aggregator.global_blocks.*.attn.qkv", "aggregator.global_blocks.*.attn.proj",
        "aggregator.global_blocks.*.mlp.fc1", "aggregator.global_blocks.*.mlp.fc2",
    )
    exclude: tuple[str, ...] = ()
    seed: int = 0
    momentum: float = 0.95
    reference_root: str | None = None
    lwc: bool = False
    lac: bool = False

    def __post_init__(self):
        if self.mode not in {"naive", "rotation", "smoothing", "dsfq"}:
            raise ValueError(f"Unknown quantization mode: {self.mode}")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.include = tuple(self.include)
        self.exclude = tuple(self.exclude)

    @property
    def rotation(self):
        return self.mode in {"rotation", "dsfq"}

    @property
    def smoothing(self):
        return self.mode in {"smoothing", "dsfq"}

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
