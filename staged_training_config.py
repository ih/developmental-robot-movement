"""
Configuration for staged training script.

All defaults match the Gradio app's initial values, except where specified
(W&B enabled, loss-weighted sampling mode).
"""

from dataclasses import dataclass, field
from typing import Optional
import warnings
import yaml
from pathlib import Path


@dataclass
class LRSweepConfig:
    """Configuration for time-budgeted learning rate sweep.

    Note: Whether sweeps occur is controlled by PlateauSweepConfig.enabled:
    - plateau_sweep.enabled=True (default): Plateau-triggered sweeps during training
    - plateau_sweep.enabled=False: Upfront sweeps before each stage (legacy)

    This config only contains sweep parameters (LR range, phases, etc.).
    """

    # LR search space
    lr_min: float = 1e-7
    lr_max: float = 1e-2

    # Phase A: Broad exploration (many LRs, 1 seed, short budget)
    phase_a_num_candidates: int = 5
    phase_a_seeds: int = 1
    phase_a_time_budget_min: float = 3.0  # 3 minutes per trial
    phase_a_survivor_count: int = 2

    # Phase B: Deep validation (few LRs, multiple seeds, longer budget)
    phase_b_seeds: int = 3
    phase_b_time_budget_min: float = 10.0  # 10 minutes per trial

    # Ranking metric for selecting best LR
    ranking_metric: str = "median_best_val"  # "median_best_val", "mean_best_val", "min_best_val"

    # Early termination thresholds
    min_samples_before_timeout: int = 1000  # Minimum samples before allowing timeout
    min_evals_before_stop: int = 5  # Minimum eval intervals before early stop

    # Resume support
    save_sweep_state: bool = True  # Save intermediate state for resume

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
            'phase_a_num_candidates': self.phase_a_num_candidates,
            'phase_a_seeds': self.phase_a_seeds,
            'phase_a_time_budget_min': self.phase_a_time_budget_min,
            'phase_a_survivor_count': self.phase_a_survivor_count,
            'phase_b_seeds': self.phase_b_seeds,
            'phase_b_time_budget_min': self.phase_b_time_budget_min,
            'ranking_metric': self.ranking_metric,
            'min_samples_before_timeout': self.min_samples_before_timeout,
            'min_evals_before_stop': self.min_evals_before_stop,
            'save_sweep_state': self.save_sweep_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LRSweepConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PlateauSweepConfig:
    """Configuration for plateau-triggered LR sweeps.

    When validation loss plateaus during training, trigger an LR sweep using
    the current weights. Training continues with the winning LR and weights
    from the best trial.

    Sweeps use the full Phase A → Phase B multi-phase structure from LRSweepConfig.
    Only plateau detection and triggering parameters are configured here.
    """

    # Enable/disable plateau-triggered sweeps
    enabled: bool = True

    # Plateau detection parameters
    plateau_ema_alpha: float = 0.85  # EMA smoothing for validation loss (higher = more responsive)
    plateau_improvement_threshold: float = 0.0015  # 0.5% relative improvement required
    plateau_patience: int = 25  # Updates without improvement before triggering sweep

    # Cooldown after sweep (prevents immediate re-triggering)
    cooldown_updates: int = 5  # Updates to wait after sweep before detection resumes

    # Maximum consecutive sweeps within the same plateau before stopping
    # Resets to 0 if sweep finds improvement (val_loss drops below plateau baseline)
    max_sweeps_per_stage: int = 2

    # Minimum improvement required from a sweep to continue training
    # If sweep_improvement <= min_sweep_improvement, stop training
    # 0.0 = stop on any regression (sweep made loss worse or same)
    min_sweep_improvement: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'plateau_ema_alpha': self.plateau_ema_alpha,
            'plateau_improvement_threshold': self.plateau_improvement_threshold,
            'plateau_patience': self.plateau_patience,
            'cooldown_updates': self.cooldown_updates,
            'max_sweeps_per_stage': self.max_sweeps_per_stage,
            'min_sweep_improvement': self.min_sweep_improvement,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlateauSweepConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StagedTrainingConfig:
    """Configuration for staged training runs."""

    # Core training (app defaults)
    total_samples: int = 10000000  # App default (used when stage_samples_multiplier=0)
    batch_size: int = 8  # Matches config.py BATCH_SIZE

    # Dynamic sample budget for staged training
    stage_samples_multiplier: int = 100000000000  # total_samples = num_valid_frames * multiplier
                                           # 0 = use fixed total_samples instead
    update_interval: int = 250  # Samples between validation evaluations (triggers checkpointing, divergence/plateau checks)
    window_size: int = 100  # App default
    num_best_models_to_keep: int = 1  # App default

    # Sampling (user override: Loss-weighted instead of Epoch-based)
    sampling_mode: str = "Loss-weighted"  # User specified
    loss_weight_temperature: float = 0.5  # App default
    loss_weight_refresh_interval: int = 50  # App default

    # Divergence stopping (app defaults)
    stop_on_divergence: bool = True  # App default
    divergence_gap: float = 0.002  # App default
    divergence_ratio: float = 1.5  # App default
    divergence_patience: int = 50  # App default
    divergence_min_updates: int = 10  # App default
    val_spike_threshold: float = 2.0  # App default
    val_spike_window: int = 15  # App default
    val_spike_frequency: float = 0.75  # App default

    # Validation plateau early stopping
    val_plateau_patience: int = 250  # Stop if val loss hasn't improved in N updates (0 = disabled)
    val_plateau_min_delta: float = 0.0001  # Minimum improvement to count as "better"

    # Learning rate (app defaults)
    custom_lr: float = 0.0001  # App default (0 = use config default)
    disable_lr_scaling: bool = True  # App default
    custom_warmup: int = -1  # App default (-1 = scaled default)
    lr_min_ratio: float = 0.001  # App default
    resume_warmup_ratio: float = 0.05  # App default
    plateau_factor: float = 0.8  # ReduceLROnPlateau reduction factor (new_lr = lr * factor)
    plateau_patience: int = 15  # ReduceLROnPlateau patience (0 = use divergence_patience * 2)

    # Resume mode settings (app defaults)
    preserve_optimizer: bool = False  # App default
    preserve_scheduler: bool = True  # App default
    samples_mode: str = "Train additional samples"  # App default

    # Visualization (app defaults)
    num_random_obs_to_visualize: int = 2  # App default
    selected_frame_offset: int = 3  # App default for frame selection

    # Stage settings
    runs_per_stage: int = 5  # CLI parameter
    serial_runs: bool = True  # Run runs_per_stage serially instead of in parallel
    clean_old_checkpoints: bool = True  # Clean old auto-saved checkpoints before starting

    # Baseline comparison
    enable_baseline: bool = False  # Enable baseline (from-scratch) runs for comparison
    baseline_runs_per_stage: int = 1  # Number of baseline runs per stage

    # Run identification (set at runtime, saved for reference/reproducibility)
    run_id: Optional[str] = None  # Unique identifier for concurrent execution

    # W&B (user override: enabled)
    enable_wandb: bool = True  # User specified (app default is False)
    wandb_project: str = "developmental-robot-movement"

    # LR Sweep configuration (for upfront sweeps - legacy mode)
    lr_sweep: LRSweepConfig = field(default_factory=LRSweepConfig)

    # Plateau-triggered LR sweep configuration (new default mode)
    plateau_sweep: PlateauSweepConfig = field(default_factory=PlateauSweepConfig)

    # Initial LR sweep before each stage (runs regardless of plateau_sweep.enabled)
    initial_sweep_enabled: bool = True

    # Stage-level time budget in minutes (0 = unlimited)
    # Applies to main training only; sweeps use lr_sweep.phase_a/b_time_budget_min
    stage_time_budget_min: float = 180

    def __post_init__(self):
        """Validate configuration and emit deprecation warnings."""
        # Warn about deprecated ReduceLROnPlateau fields when plateau_sweep is enabled
        if self.plateau_sweep.enabled:
            if self.plateau_factor != 0.8 or self.plateau_patience != 20:
                warnings.warn(
                    "plateau_factor and plateau_patience (ReduceLROnPlateau) are ignored when "
                    "plateau_sweep.enabled=True. LR adaptation is handled by plateau-triggered sweeps.",
                    DeprecationWarning,
                    stacklevel=2
                )
            if self.val_plateau_patience != 100 or self.val_plateau_min_delta != 0.0001:
                warnings.warn(
                    "val_plateau_patience and val_plateau_min_delta are ignored when "
                    "plateau_sweep.enabled=True. Plateau detection now triggers LR sweeps "
                    "instead of early stopping. See plateau_sweep configuration.",
                    DeprecationWarning,
                    stacklevel=2
                )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "StagedTrainingConfig":
        """Load configuration from YAML file, merging with defaults."""
        with open(yaml_path, "r") as f:
            overrides = yaml.safe_load(f) or {}

        # Handle nested LRSweepConfig
        if 'lr_sweep' in overrides and isinstance(overrides['lr_sweep'], dict):
            overrides['lr_sweep'] = LRSweepConfig.from_dict(overrides['lr_sweep'])

        # Handle nested PlateauSweepConfig
        if 'plateau_sweep' in overrides and isinstance(overrides['plateau_sweep'], dict):
            overrides['plateau_sweep'] = PlateauSweepConfig.from_dict(overrides['plateau_sweep'])

        return cls(**overrides)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_dict(self) -> dict:
        """Convert to dictionary (serializable for multiprocessing)."""
        d = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "StagedTrainingConfig":
        """Create from dictionary."""
        data = data.copy()
        # Handle nested LRSweepConfig
        if 'lr_sweep' in data and isinstance(data['lr_sweep'], dict):
            data['lr_sweep'] = LRSweepConfig.from_dict(data['lr_sweep'])
        # Handle nested PlateauSweepConfig
        if 'plateau_sweep' in data and isinstance(data['plateau_sweep'], dict):
            data['plateau_sweep'] = PlateauSweepConfig.from_dict(data['plateau_sweep'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
