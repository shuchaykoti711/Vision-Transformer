from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data"
    batch_size: int = 64
    num_workers: int = 2
    image_size: int = 32


@dataclass
class ModelConfig:
    color_channels: int = 3
    patch_size: int = 4
    embedding_dimension: int = 384
    embedding_dropout: float = 0.1
    num_heads: int = 6
    num_layers: int = 7
    mlp_size: int = 1536
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    num_classes: int = 100


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4


@dataclass
class Config:
    """Top-level configuration built from a YAML file and CLI overrides."""

    seed: int = 42
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def num_patches(self) -> int:
        return (self.data.image_size // self.model.patch_size) ** 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            seed=raw.get("seed", 42),
            output_dir=raw.get("output_dir", "outputs"),
            checkpoint_dir=raw.get("checkpoint_dir", "checkpoints"),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    def apply_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Apply flat CLI overrides, routing each key to the section that owns it."""
        sections = [self, self.data, self.model, self.training]
        for key, value in overrides.items():
            if value is None:
                continue
            for section in sections:
                names = {f.name for f in fields(section)}
                if key in names:
                    setattr(section, key, value)
                    break
            else:
                raise KeyError(f"Unknown config override: {key}")
        return self

    def validate(self) -> None:
        if self.data.image_size % self.model.patch_size != 0:
            raise ValueError(
                f"image_size ({self.data.image_size}) must be divisible by "
                f"patch_size ({self.model.patch_size})"
            )
        if self.model.embedding_dimension % self.model.num_heads != 0:
            raise ValueError(
                f"embedding_dimension ({self.model.embedding_dimension}) must be "
                f"divisible by num_heads ({self.model.num_heads})"
            )
