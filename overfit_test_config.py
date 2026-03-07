"""
Configuration dataclass for overfit_test.py.

Follows the same pattern as staged_training_config.py: dataclass with YAML
serialization, CLI override support, and Optional model fields.
"""

from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class OverfitTestConfig:
    """Configuration for overfit test runs."""

    # Test parameters
    batch_size: int = 1
    max_subsets: int = 10
    max_iterations: int = 20000
    target_loss: float = 0.0001

    # Training parameters
    learning_rate: float = 3e-4
    plateau_patience: int = 3000
    plateau_threshold: float = 1e-7

    # Loss function overrides (None = use config.py defaults)
    focal_alpha: Optional[float] = 1.0
    focal_beta: Optional[float] = None

    # Weight decay override (None = use config.py default)
    weight_decay: Optional[float] = 0.0

    # Model overrides (None = use config.py defaults)
    model_type: Optional[str] = None
    vae_type: Optional[str] = None

    # Encoder architecture overrides (None = use config.py defaults)
    encoder_embed_dim: Optional[int] = None
    encoder_depth: Optional[int] = None
    encoder_num_heads: Optional[int] = None

    # Decoder architecture overrides (None = use config.py defaults)
    decoder_embed_dim: Optional[int] = 256
    decoder_depth: Optional[int] = None
    decoder_num_heads: Optional[int] = 8

    # Reproducibility
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "OverfitTestConfig":
        """Create from dictionary, ignoring unknown keys."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OverfitTestConfig":
        """Load configuration from YAML file, merging with defaults."""
        with open(yaml_path, "r") as f:
            try:
                overrides = yaml.safe_load(f) or {}
            except yaml.constructor.ConstructorError:
                f.seek(0)
                overrides = yaml.unsafe_load(f) or {}
        return cls.from_dict(overrides)
