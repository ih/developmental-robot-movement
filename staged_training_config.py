"""
Configuration for staged training script.

All defaults match the Gradio app's initial values, except where specified
(W&B enabled, loss-weighted sampling mode).
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class StagedTrainingConfig:
    """Configuration for staged training runs."""

    # Core training (app defaults)
    total_samples: int = 10000000  # App default (used when stage_samples_multiplier=0)
    batch_size: int = 12  # App default

    # Dynamic sample budget for staged training
    stage_samples_multiplier: int = 1000  # total_samples = num_valid_frames * multiplier
                                           # 0 = use fixed total_samples instead
    update_interval: int = 500  # App default
    window_size: int = 50  # App default
    num_best_models_to_keep: int = 1  # App default

    # Sampling (user override: Loss-weighted instead of Epoch-based)
    sampling_mode: str = "Loss-weighted"  # User specified
    loss_weight_temperature: float = 0.5  # App default
    loss_weight_refresh_interval: int = 50  # App default

    # Divergence stopping (app defaults)
    stop_on_divergence: bool = True  # App default
    divergence_gap: float = 0.002  # App default
    divergence_ratio: float = 1.3  # App default
    divergence_patience: int = 5  # App default
    divergence_min_updates: int = 5  # App default
    val_spike_threshold: float = 2.0  # App default
    val_spike_window: int = 15  # App default
    val_spike_frequency: float = 0.75  # App default

    # Validation plateau early stopping
    val_plateau_patience: int = 100  # Stop if val loss hasn't improved in N updates (0 = disabled)
    val_plateau_min_delta: float = 0.0001  # Minimum improvement to count as "better"

    # Learning rate (app defaults)
    custom_lr: float = 0.0001  # App default (0 = use config default)
    disable_lr_scaling: bool = True  # App default
    custom_warmup: int = -1  # App default (-1 = scaled default)
    lr_min_ratio: float = 0.001  # App default
    resume_warmup_ratio: float = 0.05  # App default
    plateau_factor: float = 0.8  # ReduceLROnPlateau reduction factor (new_lr = lr * factor)
    plateau_patience: int = 20  # ReduceLROnPlateau patience (0 = use divergence_patience * 2)

    # Resume mode settings (app defaults)
    preserve_optimizer: bool = False  # App default
    preserve_scheduler: bool = True  # App default
    samples_mode: str = "Train additional samples"  # App default

    # Visualization (app defaults)
    num_random_obs_to_visualize: int = 2  # App default
    selected_frame_offset: int = 3  # App default for frame selection

    # Stage settings
    runs_per_stage: int = 2  # CLI parameter
    clean_old_checkpoints: bool = True  # Clean old auto-saved checkpoints before starting

    # Run identification (set at runtime, saved for reference/reproducibility)
    run_id: Optional[str] = None  # Unique identifier for concurrent execution

    # W&B (user override: enabled)
    enable_wandb: bool = True  # User specified (app default is False)
    wandb_project: str = "developmental-robot-movement"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "StagedTrainingConfig":
        """Load configuration from YAML file, merging with defaults."""
        with open(yaml_path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        return cls(**overrides)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.__dict__.copy()
