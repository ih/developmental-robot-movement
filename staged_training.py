"""
Staged Training Script

Trains progressively on staged splits of a session, using batch training
with loss-weighted sampling and divergence-based early stopping.

Usage:
    # Full staged training (all stages)
    python staged_training.py --root-session saved/sessions/so101/my_session
    python staged_training.py --root-session saved/sessions/so101/my_session --runs-per-stage 3
    python staged_training.py --root-session saved/sessions/so101/my_session --config my_config.yaml

    # Single stage only
    python staged_training.py --root-session saved/sessions/so101/my_session --stage 2

    # Direct session mode (no staged splits needed)
    python staged_training.py --train-session saved/sessions/so101/train --val-session saved/sessions/so101/val

    # Direct mode with explicit original session for evaluation
    python staged_training.py --train-session .../train --val-session .../val --original-session saved/sessions/so101/full_session

    # With starting checkpoint
    python staged_training.py --train-session ... --val-session ... --checkpoint saved/checkpoints/model.pth

Concurrent Execution:
    Multiple instances can run concurrently with isolated checkpoints and reports:

    # Terminal 1
    python staged_training.py --root-session saved/sessions/so101/my_session --run-id exp_a

    # Terminal 2 (simultaneously)
    python staged_training.py --root-session saved/sessions/so101/my_session --run-id exp_b

    If --run-id is not specified, a unique timestamp-based ID is auto-generated.
    Reports are saved to: saved/staged_training_reports/{session_name}/{run_id}/
    Checkpoints include run_id to prevent naming conflicts.
"""

import argparse
import base64
import gc
import io
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from staged_training_config import StagedTrainingConfig
from session_explorer_lib import (
    load_session_metadata,
    load_session_events,
    extract_observations,
    extract_actions,
    prebuild_all_canvases,
)
from autoencoder_concat_predictor_world_model import AutoencoderConcatPredictorWorldModel
from replay_robot import ReplayRobot
from recording_reader import RecordingReader
from recorded_policy import create_recorded_action_selector

# Import concat_world_model_explorer modules
from concat_world_model_explorer import state
from concat_world_model_explorer import training
from concat_world_model_explorer import evaluation
from concat_world_model_explorer import inference
from concat_world_model_explorer import visualization
from concat_world_model_explorer import checkpoint_manager
from concat_world_model_explorer.canvas_ops import build_canvas_from_frame
from models.autoencoder_concat_predictor import (
    canvas_to_tensor,
    compute_randomized_patch_mask_for_last_slot_gpu,
    compute_hybrid_loss_on_masked_patches,
)
from lr_sweep import (
    LRTrialResult,
    LRAggregatedResult,
    LRSweepPhaseResult,
    LRSweepStageResult,
    StageTiming,
    format_duration,
    run_lr_sweep_for_stage,
    run_main_training_parallel,
    run_plateau_triggered_sweep,
)


# Partial report state for interrupt/crash recovery.
# Populated by run_staged_training() so main() can generate a final report
# if the process is interrupted (Ctrl+C) or crashes mid-training.
_partial_report_state = None


def set_all_seeds(seed: int, deterministic_cudnn: bool = False):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(base_seed: int, stage_num: int, run_num: int, is_baseline: bool) -> int:
    """Derive a deterministic per-run seed from a base seed."""
    return hash((base_seed, stage_num, run_num, is_baseline)) % (2**32)


@dataclass
class StageResult:
    """Result from training a single stage/run."""
    stage_num: int
    run_num: int
    is_baseline: bool  # True for baseline runs (fresh weights), False for staged (carryover)
    best_checkpoint_path: str
    best_loss: float
    stop_reason: str
    elapsed_time: float
    total_samples_trained: int
    cumulative_metrics: dict
    final_train_loss: float
    final_val_loss: Optional[float]
    # Plateau sweep tracking
    sweep_history: list = field(default_factory=list)  # List of sweep results
    total_sweeps_triggered: int = 0
    initial_lr: float = 0.0
    final_lr: float = 0.0


@dataclass
class StageComparison:
    """Comparison metrics between staged and baseline for a single stage."""
    stage_num: int
    staged_orig_loss: float
    baseline_orig_loss: float
    improvement_absolute: float  # baseline_loss - staged_loss (positive = staged better)
    improvement_ratio: float  # baseline_loss / staged_loss (>1 = staged better)
    staged_samples_trained: int
    baseline_samples_trained: int


def compute_stage_comparison(
    staged_result: StageResult,
    baseline_result: StageResult,
    staged_orig_loss: float,
    baseline_orig_loss: float,
) -> StageComparison:
    """Compute comparison metrics between staged and baseline results."""
    improvement_absolute = baseline_orig_loss - staged_orig_loss
    improvement_ratio = baseline_orig_loss / staged_orig_loss if staged_orig_loss > 0 else float('inf')

    return StageComparison(
        stage_num=staged_result.stage_num,
        staged_orig_loss=staged_orig_loss,
        baseline_orig_loss=baseline_orig_loss,
        improvement_absolute=improvement_absolute,
        improvement_ratio=improvement_ratio,
        staged_samples_trained=staged_result.total_samples_trained,
        baseline_samples_trained=baseline_result.total_samples_trained,
    )


def create_staged_vs_baseline_plot(
    staged_evals: list,
    baseline_evals: list,
) -> plt.Figure:
    """Create dual-line plot comparing staged vs baseline progression."""
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = [s for s, _, _, _ in staged_evals]
    staged_losses = [loss for _, _, loss, _ in staged_evals]
    baseline_losses = [loss for _, _, loss, _ in baseline_evals]

    ax.plot(stages, staged_losses, 'g-o', linewidth=2, markersize=8,
            label='Staged (weight carryover)')
    ax.plot(stages, baseline_losses, 'b--s', linewidth=2, markersize=8,
            label='Baseline (fresh each stage)')

    ax.set_xlabel('Stage')
    ax.set_ylabel('Hybrid Loss (Original Session)')
    ax.set_title('Stage Progression: Staged vs Baseline Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def aggregate_sweep_data(
    completed_stages: list[tuple[int, StageResult, Optional[StageResult]]],
) -> dict:
    """
    Aggregate plateau sweep data across all stages for timeline visualization.

    Args:
        completed_stages: List of (stage_num, staged_result, baseline_result) tuples

    Returns:
        Dictionary with aggregated sweep statistics and per-stage data with
        cumulative sample offsets for timeline plotting.
    """
    total_sweeps = 0
    total_sweep_time_sec = 0.0
    per_stage = []
    cumulative_samples = 0

    for stage_num, staged, _ in completed_stages:
        sweep_history = staged.sweep_history if hasattr(staged, 'sweep_history') else []
        sweep_count = len(sweep_history)
        total_sweeps += sweep_count

        stage_sweep_time = sum(s.get('sweep_duration_sec', 0) for s in sweep_history)
        total_sweep_time_sec += stage_sweep_time

        per_stage.append({
            'stage_num': stage_num,
            'sweep_count': sweep_count,
            'sweep_history': sweep_history,
            'initial_lr': staged.initial_lr if hasattr(staged, 'initial_lr') else 0.0,
            'final_lr': staged.final_lr if hasattr(staged, 'final_lr') else 0.0,
            'cumulative_samples_start': cumulative_samples,
            'total_samples_trained': staged.total_samples_trained,
            'cumulative_samples_end': cumulative_samples + staged.total_samples_trained,
        })

        cumulative_samples += staged.total_samples_trained

    avg_sweep_duration = total_sweep_time_sec / total_sweeps if total_sweeps > 0 else 0.0

    return {
        'total_sweeps': total_sweeps,
        'total_sweep_time_sec': total_sweep_time_sec,
        'avg_sweep_duration_sec': avg_sweep_duration,
        'stages_with_sweeps': sum(1 for s in per_stage if s['sweep_count'] > 0),
        'per_stage': per_stage,
        'total_samples': cumulative_samples,
    }


def create_plateau_sweep_timeline_plot(
    completed_stages: list[tuple[int, StageResult, Optional[StageResult]]],
) -> Optional[plt.Figure]:
    """
    Create timeline plot showing LR changes and plateau sweep events across all stages.

    X-axis: Cumulative samples trained across all stages
    Y-axis: Learning rate (log scale)
    Markers: Vertical dashed lines at sweep trigger points, solid lines at stage boundaries

    Args:
        completed_stages: List of (stage_num, staged_result, baseline_result) tuples

    Returns:
        Matplotlib figure or None if no data to plot
    """
    sweep_data = aggregate_sweep_data(completed_stages)

    if not sweep_data['per_stage']:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette for stages
    colors = plt.cm.tab10.colors
    stage_colors = {s['stage_num']: colors[i % len(colors)] for i, s in enumerate(sweep_data['per_stage'])}

    # Track all LR points for y-axis limits
    all_lrs = []

    for stage_info in sweep_data['per_stage']:
        stage_num = stage_info['stage_num']
        color = stage_colors[stage_num]
        start_samples = stage_info['cumulative_samples_start']
        end_samples = stage_info['cumulative_samples_end']
        initial_lr = stage_info['initial_lr']
        sweep_history = stage_info['sweep_history']

        if initial_lr <= 0:
            continue

        all_lrs.append(initial_lr)

        # Build LR segments for this stage
        lr_points = [(start_samples, initial_lr)]
        current_lr = initial_lr

        for sweep in sorted(sweep_history, key=lambda s: s.get('triggered_at_samples', 0)):
            trigger_samples = start_samples + sweep.get('triggered_at_samples', 0)
            selected_lr = sweep.get('selected_lr', current_lr)

            # Add point just before sweep
            lr_points.append((trigger_samples, current_lr))
            # Add point after sweep with new LR
            lr_points.append((trigger_samples, selected_lr))

            all_lrs.append(selected_lr)
            current_lr = selected_lr

            # Draw vertical dashed line for sweep trigger
            ax.axvline(x=trigger_samples, color=color, linestyle='--', alpha=0.6, linewidth=1.5)

            # Annotate sweep
            sweep_num = sweep.get('sweep_num', '?')
            ax.annotate(
                f'S{stage_num}#{sweep_num}',
                xy=(trigger_samples, selected_lr),
                xytext=(5, 10), textcoords='offset points',
                fontsize=8, color=color, alpha=0.8,
            )

        # Add final point at stage end
        lr_points.append((end_samples, current_lr))

        # Plot LR line for this stage
        x_vals = [p[0] for p in lr_points]
        y_vals = [p[1] for p in lr_points]
        ax.plot(x_vals, y_vals, color=color, linewidth=2, marker='o', markersize=4,
                label=f'Stage {stage_num}')

        # Draw stage boundary (solid vertical line)
        if start_samples > 0:
            ax.axvline(x=start_samples, color='gray', linestyle='-', alpha=0.3, linewidth=2)

    # Configure axes
    ax.set_xlabel('Cumulative Samples Trained', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Timeline with Plateau Sweeps', fontsize=12, fontweight='bold')
    ax.set_yscale('log')

    # Set y-axis limits with padding
    if all_lrs:
        min_lr = min(all_lrs) * 0.5
        max_lr = max(all_lrs) * 2
        ax.set_ylim(min_lr, max_lr)

    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def discover_staged_splits(root_session_path: str) -> list[tuple[int, str, str]]:
    """
    Discover all staged train/validate session pairs for a root session.

    Args:
        root_session_path: Path to the root session directory

    Returns:
        List of (stage_num, train_path, validate_path) tuples, sorted by stage number
    """
    root_path = Path(root_session_path).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Root session not found: {root_path}")

    session_name = root_path.name
    parent_dir = root_path.parent

    # Pattern: {session_name}_stage{N}_train_{size}
    train_pattern = re.compile(rf"^{re.escape(session_name)}_stage(\d+)_train_\d+$")

    stages = []
    for item in parent_dir.iterdir():
        if not item.is_dir():
            continue

        match = train_pattern.match(item.name)
        if match:
            stage_num = int(match.group(1))
            train_path = str(item)

            # Find corresponding validate session
            validate_pattern = f"{session_name}_stage{stage_num}_validate_"
            validate_paths = list(parent_dir.glob(f"{validate_pattern}*"))

            if validate_paths:
                validate_path = str(validate_paths[0])
                stages.append((stage_num, train_path, validate_path))
            else:
                print(f"Warning: No validation session found for stage {stage_num}")

    # Sort by stage number
    stages.sort(key=lambda x: x[0])
    return stages


def load_session_for_training(session_path: str) -> dict:
    """
    Load a session and pre-build all canvases for training.

    Args:
        session_path: Path to session directory

    Returns:
        Session state dictionary with canvas_cache
    """
    session_path = Path(session_path).resolve()

    metadata = load_session_metadata(str(session_path))
    events = load_session_events(str(session_path))

    if not metadata or not events:
        raise ValueError(f"Could not load session from {session_path}")

    # Extract observations and actions (properly unpacking data dicts)
    observations = extract_observations(events, str(session_path))
    actions = extract_actions(events)

    # Pre-build all canvases
    canvas_cache, detected_frame_size = prebuild_all_canvases(
        str(session_path),
        observations,
        actions,
        config.AutoencoderConcatPredictorWorldModelConfig,
    )

    return {
        "session_name": metadata.get("session_name", session_path.name),
        "session_dir": str(session_path),
        "metadata": metadata,
        "events": events,
        "observations": observations,
        "actions": actions,
        "canvas_cache": canvas_cache,
        "detected_frame_size": detected_frame_size,
        "action_space": metadata.get("action_space", []),
    }


def setup_world_model(
    train_session: dict,
    checkpoint_path: Optional[str] = None,
    cfg: Optional[StagedTrainingConfig] = None,
) -> None:
    """
    Set up or restore world model in global state.

    Args:
        train_session: Training session state dict
        checkpoint_path: Optional checkpoint to load
        cfg: Training config
    """
    # Update global state with session info
    state.session_state = train_session

    # Create replay robot and action selector
    reader = RecordingReader(train_session["session_dir"])
    action_selector = create_recorded_action_selector(reader)

    # Determine checkpoint directory based on session path
    session_dir = train_session["session_dir"]
    checkpoint_dir = state.get_checkpoint_dir_for_session(session_dir)

    # Determine frame_size: prefer checkpoint's frame_size if loading a checkpoint
    frame_size = train_session.get("detected_frame_size")
    checkpoint = None

    if checkpoint_path:
        # Pre-load checkpoint to get frame_size before creating model
        checkpoint = torch.load(checkpoint_path, map_location=state.device, weights_only=False)
        checkpoint_frame_size = checkpoint.get('config', {}).get('frame_size')
        if checkpoint_frame_size:
            checkpoint_frame_size = tuple(checkpoint_frame_size)
            if checkpoint_frame_size != frame_size:
                print(f"Using checkpoint frame_size {checkpoint_frame_size} (session has {frame_size})")
            frame_size = checkpoint_frame_size

    # Create world model with correct frame_size
    robot = ReplayRobot(reader, train_session["action_space"])
    state.world_model = AutoencoderConcatPredictorWorldModel(
        robot_interface=robot,
        action_selector=action_selector,
        device=state.device,
        frame_size=frame_size,
    )

    # Load checkpoint state if provided
    if checkpoint_path and checkpoint is not None:
        state.world_model.autoencoder.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer and scheduler if available
        if "optimizer_state_dict" in checkpoint:
            state.world_model.ae_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            # Check scheduler type compatibility before loading
            checkpoint_scheduler_type = checkpoint.get("scheduler_type", "unknown")
            current_scheduler_type = type(state.world_model.ae_scheduler).__name__
            if checkpoint_scheduler_type == current_scheduler_type:
                state.world_model.ae_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                print(f"Skipping scheduler state: checkpoint has {checkpoint_scheduler_type}, "
                      f"model has {current_scheduler_type}")

        # Update loaded checkpoint metadata
        state.loaded_checkpoint_metadata = {
            "samples_seen": checkpoint.get("samples_seen", 0),
            "loss": checkpoint.get("loss", 0),
            "original_peak_lr": checkpoint.get("original_peak_lr"),
            "checkpoint_name": Path(checkpoint_path).name,
        }
        print(f"Loaded checkpoint: {Path(checkpoint_path).name}")
    else:
        state.loaded_checkpoint_metadata = {}

    # Free checkpoint dict to release GPU memory
    if checkpoint is not None:
        del checkpoint


def run_stage_training(
    stage_num: int,
    run_num: int,
    train_session: dict,
    val_session: dict,
    checkpoint_path: Optional[str],
    cfg: StagedTrainingConfig,
    output_dir: Path,
    is_baseline: bool = False,
    run_id: str = "",
    time_budget_min: float = 0,
) -> StageResult:
    """
    Run training for a single stage.

    Args:
        stage_num: Stage number
        run_num: Run number within stage
        train_session: Training session state
        val_session: Validation session state
        checkpoint_path: Starting checkpoint (None for fresh)
        cfg: Training configuration
        output_dir: Output directory for this run
        is_baseline: True for baseline runs (fresh weights), False for staged
        run_id: Unique identifier for this training run (used in W&B run name)
        time_budget_min: Time budget in minutes (0 = unlimited)

    Returns:
        StageResult with training outcomes
    """
    baseline_label = " (BASELINE)" if is_baseline else ""
    print(f"\n{'='*60}")
    print(f"Stage {stage_num} Run {run_num}{baseline_label}")
    print(f"Training: {train_session['session_name']}")
    print(f"Validation: {val_session['session_name']}")
    print(f"{'='*60}\n")

    # Setup world model
    setup_world_model(train_session, checkpoint_path, cfg)

    # Setup validation session in state
    state.validation_session_state = val_session

    # Determine resume mode
    resume_mode = checkpoint_path is not None
    starting_samples = state.loaded_checkpoint_metadata.get("samples_seen", 0) if resume_mode else 0

    # W&B run name (includes run_id and baseline indicator)
    baseline_suffix = "_baseline" if is_baseline else ""
    run_id_part = f"_{run_id}" if run_id else ""
    wandb_run_name = f"{train_session['session_name']}{run_id_part}{baseline_suffix}_run{run_num}"

    # Track training metrics
    start_time = time.time()
    # Reset cumulative_metrics in state (will be populated by run_world_model_batch)
    state.cumulative_metrics = {
        "samples_seen": [],
        "loss_at_sample": [],
        "val_loss_at_sample": [],
        "lr_at_sample": [],
    }
    best_loss = float("inf")
    best_checkpoint_path = None
    stop_reason = "max_samples"
    final_train_loss = 0
    final_val_loss = None
    total_samples_trained = 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate dynamic sample budget based on training set size
    num_train_observations = len(train_session.get("observations", []))
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    num_valid_frames = num_train_observations - (min_frames_needed - 1)

    if cfg.stage_samples_multiplier > 0:
        effective_total_samples = num_valid_frames * cfg.stage_samples_multiplier
        print(f"Dynamic sample budget: {num_valid_frames} valid frames x {cfg.stage_samples_multiplier} = {effective_total_samples:,} samples")
    else:
        effective_total_samples = cfg.total_samples
        print(f"Fixed sample budget: {effective_total_samples:,} samples")

    # Plateau sweep tracking
    sweep_history = []
    sweep_count = 0  # Total sweeps triggered (for reporting)
    consecutive_sweep_count = 0  # Consecutive sweeps within same plateau (resets on improvement)
    val_loss_at_plateau_start = None  # Baseline for detecting improvement between sweeps
    initial_lr = cfg.custom_lr
    current_lr = cfg.custom_lr
    current_checkpoint = checkpoint_path
    post_sweep = False  # Track if we're resuming after a sweep (affects samples_mode)

    # Determine if plateau sweep mode is enabled
    plateau_sweep_enabled = hasattr(cfg, 'plateau_sweep') and cfg.plateau_sweep.enabled

    # Run batch training generator (with sweep loop if plateau_sweep_enabled)
    training_complete = False
    carried_checkpoints = []  # Carry auto_saved_checkpoints across generator restarts
    try:
        while not training_complete:
            generator = training.run_world_model_batch(
                total_samples=effective_total_samples,
                batch_size=cfg.batch_size,
                current_observation_idx=cfg.selected_frame_offset,
                update_interval=cfg.update_interval,
                window_size=cfg.window_size,
                num_random_obs=cfg.num_random_obs_to_visualize,
                num_best_models_to_keep=cfg.num_best_models_to_keep,
                enable_wandb=cfg.enable_wandb,
                wandb_run_name=wandb_run_name,
                # Resume mode
                resume_mode=(current_checkpoint is not None),
                # After a sweep, use "Train to total samples" to avoid over-training.
                # BUT: in divergence mode, total_samples gets overridden to chunk size,
                # so "Train to total samples" would exit immediately if starting_samples > chunk_size.
                # Keep "Train additional samples" in divergence mode to let chunking work correctly.
                samples_mode="Train to total samples" if (post_sweep and not cfg.stop_on_divergence) else cfg.samples_mode,
                starting_samples=starting_samples,
                preserve_optimizer=cfg.preserve_optimizer,
                preserve_scheduler=cfg.preserve_scheduler,
                # Learning rate
                custom_lr=current_lr,
                disable_lr_scaling=cfg.disable_lr_scaling,
                custom_warmup=cfg.custom_warmup,
                lr_min_ratio=cfg.lr_min_ratio,
                resume_warmup_ratio=cfg.resume_warmup_ratio,
                # Sampling
                sampling_mode=cfg.sampling_mode,
                # Loss-weighted
                loss_weight_temperature=cfg.loss_weight_temperature,
                loss_weight_refresh_interval=cfg.loss_weight_refresh_interval,
                # Divergence
                stop_on_divergence=cfg.stop_on_divergence,
                divergence_gap=cfg.divergence_gap,
                divergence_ratio=cfg.divergence_ratio,
                divergence_patience=cfg.divergence_patience,
                divergence_min_updates=cfg.divergence_min_updates,
                val_spike_threshold=cfg.val_spike_threshold,
                val_spike_window=cfg.val_spike_window,
                val_spike_frequency=cfg.val_spike_frequency,
                # Validation plateau early stopping (legacy - ignored when plateau_sweep_enabled)
                val_plateau_patience=cfg.val_plateau_patience if not plateau_sweep_enabled else 0,
                val_plateau_min_delta=cfg.val_plateau_min_delta,
                # ReduceLROnPlateau (legacy - ignored when plateau_sweep_enabled)
                plateau_factor=cfg.plateau_factor,
                plateau_patience=cfg.plateau_patience,
                # Baseline comparison parameters (for W&B logging)
                is_baseline_run=is_baseline,
                enable_baseline=cfg.enable_baseline,
                baseline_runs_per_stage=cfg.baseline_runs_per_stage,
                # Time budget for LR sweep
                time_budget_min=time_budget_min,
                min_samples_for_timeout=cfg.lr_sweep.min_samples_before_timeout if hasattr(cfg, 'lr_sweep') else 1000,
                # Plateau sweep parameters
                plateau_sweep_enabled=plateau_sweep_enabled,
                plateau_sweep_ema_alpha=cfg.plateau_sweep.plateau_ema_alpha if plateau_sweep_enabled else 0.9,
                plateau_sweep_improvement_threshold=cfg.plateau_sweep.plateau_improvement_threshold if plateau_sweep_enabled else 0.005,
                plateau_sweep_patience=cfg.plateau_sweep.plateau_patience if plateau_sweep_enabled else 10,
                plateau_sweep_cooldown_updates=cfg.plateau_sweep.cooldown_updates if plateau_sweep_enabled else 10,
                plateau_sweep_max_sweeps=cfg.plateau_sweep.max_sweeps_per_stage if plateau_sweep_enabled else 3,
                plateau_sweep_count=consecutive_sweep_count,
                prior_auto_saved_checkpoints=carried_checkpoints,
            )

            # Consume generator and collect results
            sweep_triggered = False
            for result in generator:
                if result is None:
                    continue

                # Check for sweep request sentinel
                if isinstance(result, tuple) and len(result) >= 5 and result[0] == "SWEEP_REQUESTED":
                    _, samples_at_sweep, checkpoint_for_sweep, new_consecutive_count, val_loss_at_trigger = result
                    sweep_triggered = True
                    sweep_count += 1  # Total sweeps (for reporting)

                    # Determine if this is a new plateau (improvement) or same plateau (consecutive)
                    improvement_threshold = cfg.plateau_sweep.plateau_improvement_threshold if plateau_sweep_enabled else 0.0005
                    if val_loss_at_plateau_start is None:
                        # First sweep - establish baseline
                        val_loss_at_plateau_start = val_loss_at_trigger
                        consecutive_sweep_count = 1
                        print(f"\n[STAGED TRAINING] Sweep #{sweep_count} (consecutive: {consecutive_sweep_count}) - first plateau at val_loss={val_loss_at_trigger:.6f}")
                    elif val_loss_at_trigger < val_loss_at_plateau_start * (1 - improvement_threshold):
                        # Improvement detected - this is a new plateau, reset consecutive count
                        print(f"\n[STAGED TRAINING] Sweep #{sweep_count} - improvement detected: {val_loss_at_trigger:.6f} < {val_loss_at_plateau_start:.6f} * {1 - improvement_threshold:.4f}")
                        val_loss_at_plateau_start = val_loss_at_trigger
                        consecutive_sweep_count = 1
                        print(f"[STAGED TRAINING] New plateau - reset consecutive count to {consecutive_sweep_count}")
                    else:
                        # No improvement - same plateau, increment consecutive count
                        consecutive_sweep_count = new_consecutive_count
                        print(f"\n[STAGED TRAINING] Sweep #{sweep_count} (consecutive: {consecutive_sweep_count}) - same plateau, val_loss={val_loss_at_trigger:.6f} vs baseline={val_loss_at_plateau_start:.6f}")

                    print(f"[STAGED TRAINING] Checkpoint for sweep: {checkpoint_for_sweep}")

                    # Free main process model to give GPU memory to sweep workers
                    # Model state is preserved in checkpoint_for_sweep
                    if state.world_model is not None:
                        del state.world_model
                        state.world_model = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Snapshot existing checkpoints so we can clean up sweep-created ones afterward
                    checkpoint_dir = state.get_checkpoint_dir_for_session(train_session["session_dir"])
                    pre_sweep_files = set(Path(checkpoint_dir).glob("best_model_auto_*.pth"))

                    # Run plateau-triggered sweep (uses phase budgets from lr_sweep config)
                    sweep_start_time = time.time()
                    new_lr, new_checkpoint, sweep_best_val = run_plateau_triggered_sweep(
                        sweep_number=sweep_count,
                        train_session_path=train_session["session_dir"],
                        val_session_path=val_session["session_dir"],
                        current_checkpoint=checkpoint_for_sweep,
                        cfg_dict=cfg.to_dict(),
                        output_dir=output_dir,
                        run_id=run_id,
                    )
                    sweep_elapsed = time.time() - sweep_start_time

                    # Force GPU memory cleanup after sweep
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Clean up sweep trial checkpoints (workers save to shared dir)
                    post_sweep_files = set(Path(checkpoint_dir).glob("best_model_auto_*.pth"))
                    sweep_created = post_sweep_files - pre_sweep_files
                    winning_path = Path(new_checkpoint) if new_checkpoint else None
                    deleted_count = 0
                    for f in sweep_created:
                        if winning_path and f.resolve() == winning_path.resolve():
                            continue
                        try:
                            f.unlink()
                            deleted_count += 1
                        except Exception as e:
                            print(f"[SWEEP CLEANUP] Failed to delete {f.name}: {e}")
                    if deleted_count > 0:
                        print(f"[SWEEP CLEANUP] Deleted {deleted_count} trial checkpoint(s), kept winner: {winning_path.name if winning_path else 'none'}")

                    # Reload model -- use winning checkpoint if available, else fall back to pre-sweep checkpoint
                    # Model was freed for GPU memory, so it must be restored regardless of sweep outcome
                    restore_checkpoint = new_checkpoint or checkpoint_for_sweep
                    if restore_checkpoint:
                        setup_world_model(train_session, restore_checkpoint, cfg)
                        state.validation_session_state = val_session

                    # Calculate improvement from sweep
                    # Handle None case (if phase_b.winner_best_val was None)
                    if sweep_best_val is None:
                        sweep_best_val = val_loss_at_trigger  # Assume no improvement
                    sweep_improvement = val_loss_at_trigger - sweep_best_val
                    improvement_pct = (sweep_improvement / val_loss_at_trigger) * 100 if val_loss_at_trigger > 0 else 0.0

                    # Record sweep in history
                    sweep_history.append({
                        'sweep_num': sweep_count,
                        'consecutive_num': consecutive_sweep_count,
                        'triggered_at_samples': samples_at_sweep,
                        'triggered_at_time': time.time() - start_time - sweep_elapsed,
                        'val_loss_at_trigger': val_loss_at_trigger,
                        'val_loss_after': sweep_best_val,
                        'improvement': sweep_improvement,
                        'improvement_pct': improvement_pct,
                        'selected_lr': new_lr,
                        'checkpoint_path': new_checkpoint,
                        'sweep_duration_sec': sweep_elapsed,
                    })

                    # Check if sweep produced improvement
                    min_improvement = cfg.plateau_sweep.min_sweep_improvement
                    if sweep_improvement <= min_improvement:
                        stop_reason = f"sweep_no_improvement (val: {val_loss_at_trigger:.6f} -> {sweep_best_val:.6f}, delta: {sweep_improvement:.6f})"
                        print(f"\n[STAGED TRAINING] Sweep produced no improvement: {sweep_improvement:.6f} <= {min_improvement}")
                        print(f"[STAGED TRAINING] Stopping training: {stop_reason}")
                        training_complete = True
                        break

                    print(f"[STAGED TRAINING] Sweep improved val_loss: {val_loss_at_trigger:.6f} -> {sweep_best_val:.6f} ({improvement_pct:+.2f}%)")

                    # Update for next iteration
                    current_lr = new_lr
                    current_checkpoint = new_checkpoint
                    # Use samples_at_sweep (not checkpoint's samples_seen) to avoid over-training
                    # The sweep trains extra samples internally, but those shouldn't count toward stage budget
                    # With post_sweep=True, we use "Train to total samples" mode to reach effective_total_samples
                    starting_samples = samples_at_sweep
                    post_sweep = True

                    print(f"[STAGED TRAINING] Continuing training with LR={new_lr:.2e}, starting_samples={starting_samples}, consecutive={consecutive_sweep_count}/{cfg.plateau_sweep.max_sweeps_per_stage}")
                    # Clean up old generator to free DataLoader pinned memory buffers
                    del generator
                    gc.collect()
                    break  # Exit generator loop to restart with new LR/checkpoint

                # Normal result - unpack tuple (matches generate_batch_training_update output)
                if isinstance(result, tuple) and len(result) == 9:
                    (status_msg, loss_fig, loss_recent_fig, lr_fig, weights_fig,
                     eval_loss_fig, eval_dist_fig, obs_status, obs_fig) = result

                    # Parse status for metrics
                    if isinstance(status_msg, str) and "samples" in status_msg.lower():
                        # Extract samples count from status
                        match = re.search(r"(\d+)\s*/\s*(\d+|\?+)", status_msg)
                        if match:
                            samples_seen = int(match.group(1))

                    # Check for stop reason in status (use specific phrases to avoid false positives)
                    if isinstance(status_msg, str):
                        status_lower = status_msg.lower()
                        if "nan" in status_lower or "inf loss" in status_lower:
                            stop_reason = "nan_loss"
                        elif "divergence detected" in status_lower:
                            stop_reason = "divergence"
                        elif "validation plateau" in status_lower:
                            stop_reason = "val_plateau"
                        elif "max sweeps reached" in status_lower:
                            stop_reason = f"max_sweeps ({cfg.plateau_sweep.max_sweeps_per_stage})"
                        elif "time budget" in status_lower:
                            stop_reason = "time_budget"
                        elif "training complete" in status_lower or "training finished" in status_lower:
                            stop_reason = "completed"

                    # Close figures to free memory
                    for fig in [loss_fig, loss_recent_fig, lr_fig, weights_fig,
                                eval_loss_fig, eval_dist_fig, obs_fig]:
                        if fig is not None:
                            plt.close(fig)

            # Carry forward auto_saved_checkpoints for next generator restart
            if state.cumulative_metrics and state.cumulative_metrics.get("auto_saved_checkpoints"):
                carried_checkpoints = state.cumulative_metrics["auto_saved_checkpoints"]

            # If no sweep was triggered, training is complete
            if not sweep_triggered:
                training_complete = True

    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        stop_reason = f"error: {str(e)}"

    elapsed_time = time.time() - start_time

    # Make stop_reason more informative for sample budget or time budget completion
    if stop_reason == "completed":
        if cfg.stage_samples_multiplier > 0:
            stop_reason = f"sample_budget ({num_valid_frames} frames x {cfg.stage_samples_multiplier} = {effective_total_samples:,})"
        else:
            stop_reason = f"sample_budget ({effective_total_samples:,} fixed)"
    elif stop_reason == "time_budget":
        stop_reason = f"time_budget ({time_budget_min:.1f} min limit)"

    # Extract total_samples_trained and final losses from cumulative_metrics
    if state.cumulative_metrics and state.cumulative_metrics.get("samples_seen"):
        total_samples_trained = state.cumulative_metrics["samples_seen"][-1] - starting_samples
    else:
        total_samples_trained = 0

    if state.cumulative_metrics:
        train_losses = state.cumulative_metrics.get("loss_at_sample", [])
        val_losses = state.cumulative_metrics.get("val_loss_at_sample", [])
        if train_losses:
            final_train_loss = train_losses[-1]
        if val_losses:
            final_val_loss = val_losses[-1]

    # Find best checkpoint from this run (use tracked checkpoints, not glob to avoid old checkpoint pollution)
    checkpoint_dir = state.get_checkpoint_dir_for_session(train_session["session_dir"])
    session_name = train_session["session_name"]

    # Use auto_saved_checkpoints from cumulative_metrics (only checkpoints from this run)
    auto_saved_checkpoints = []
    if state.cumulative_metrics and state.cumulative_metrics.get("auto_saved_checkpoints"):
        auto_saved_checkpoints = state.cumulative_metrics["auto_saved_checkpoints"]

    if auto_saved_checkpoints:
        # Sort by loss (ascending) - list contains (loss, filepath) tuples
        auto_saved_checkpoints.sort(key=lambda x: x[0])
        best_loss, best_checkpoint_path = auto_saved_checkpoints[0]
        print(f"Best checkpoint from this run: {Path(best_checkpoint_path).name} (loss: {best_loss:.6f})")
    else:
        # Save current model as checkpoint
        checkpoint_name = f"stage{stage_num}_run{run_num}_final.pth"
        best_checkpoint_path = str(Path(checkpoint_dir) / checkpoint_name)
        torch.save({
            "model_state_dict": state.world_model.autoencoder.state_dict(),
            "optimizer_state_dict": state.world_model.ae_optimizer.state_dict(),
            "scheduler_state_dict": state.world_model.ae_scheduler.state_dict(),
            "scheduler_type": type(state.world_model.ae_scheduler).__name__,
            "model_type": config.AutoencoderConcatPredictorWorldModelConfig.MODEL_TYPE,
            "samples_seen": starting_samples + total_samples_trained,
            "loss": best_loss if best_loss != float("inf") else 0,
            "config": {
                "frame_size": state.world_model.frame_size,  # Critical for loading with correct dimensions
                "separator_width": config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
                "canvas_history_size": config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE,
            },
        }, best_checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_name}")

    return StageResult(
        stage_num=stage_num,
        run_num=run_num,
        is_baseline=is_baseline,
        best_checkpoint_path=best_checkpoint_path,
        best_loss=best_loss,
        stop_reason=stop_reason,
        elapsed_time=elapsed_time,
        total_samples_trained=total_samples_trained,
        cumulative_metrics=state.cumulative_metrics,  # Get from state (populated by run_world_model_batch)
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        # Plateau sweep tracking
        sweep_history=sweep_history,
        total_sweeps_triggered=sweep_count,
        initial_lr=initial_lr,
        final_lr=current_lr,
    )


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def save_fig_to_file(fig, path: Path) -> None:
    """Save matplotlib figure to file."""
    if fig is None:
        return
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def get_config_changes_from_git() -> dict[str, tuple]:
    """
    Compare current StagedTrainingConfig defaults against the last git commit.

    Returns dict of {param_name: (old_value, new_value)} for changed defaults.
    Uses dotted names for nested configs (e.g., 'plateau_sweep.plateau_patience').
    Returns empty dict if git or import fails.
    """
    import subprocess
    import importlib
    import importlib.util
    import tempfile
    import sys

    try:
        result = subprocess.run(
            ["git", "show", "HEAD:staged_training_config.py"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return {}

        # Write to temp file and import
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "_prev_config.py")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        # Add temp dir to sys.path temporarily
        sys.path.insert(0, temp_dir)
        try:
            # Import the previous version
            spec = importlib.util.spec_from_file_location("_prev_config", temp_path)
            prev_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(prev_module)
            prev_cfg = prev_module.StagedTrainingConfig()
        finally:
            sys.path.remove(temp_dir)
            # Clean up temp dir (may contain __pycache__)
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                pass

        # Compare with current defaults
        current_cfg = StagedTrainingConfig()
        changes = {}

        def compare_configs(old_obj, new_obj, prefix=""):
            for field_name in new_obj.__dataclass_fields__:
                old_val = getattr(old_obj, field_name, None)
                new_val = getattr(new_obj, field_name, None)
                full_name = f"{prefix}{field_name}" if not prefix else f"{prefix}.{field_name}"
                if prefix == "":
                    full_name = field_name

                # Recurse into nested dataclasses
                if hasattr(new_val, '__dataclass_fields__') and hasattr(old_val, '__dataclass_fields__'):
                    compare_configs(old_val, new_val, full_name)
                elif old_val != new_val:
                    changes[full_name] = (old_val, new_val)

        compare_configs(prev_cfg, current_cfg)
        return changes

    except Exception:
        return {}


def generate_config_html(cfg: StagedTrainingConfig) -> str:
    """
    Generate HTML showing config changes from git and all parameters.

    Returns HTML with:
    - Visible table of defaults changed since last commit
    - Collapsible section with all parameter values
    """
    # Get git changes
    changes = get_config_changes_from_git()

    # Build changed params table
    if changes:
        change_rows = ""
        for param, (old_val, new_val) in sorted(changes.items()):
            change_rows += f"""
            <tr>
                <td><code>{param}</code></td>
                <td>{old_val}</td>
                <td><strong>{new_val}</strong></td>
            </tr>"""

        changes_html = f"""
        <h3>Config Defaults Changed Since Last Commit</h3>
        <table>
            <tr><th>Parameter</th><th>Previous</th><th>Current</th></tr>
            {change_rows}
        </table>"""
    else:
        changes_html = "<p><em>No config defaults changed since last commit.</em></p>"

    # Build all params table
    def flatten_config(obj, prefix=""):
        rows = []
        for field_name in obj.__dataclass_fields__:
            val = getattr(obj, field_name, None)
            full_name = f"{prefix}.{field_name}" if prefix else field_name
            if hasattr(val, '__dataclass_fields__'):
                rows.extend(flatten_config(val, full_name))
            else:
                rows.append((full_name, val))
        return rows

    all_params = flatten_config(cfg)
    all_rows = ""
    for param, val in all_params:
        all_rows += f"""
            <tr><td><code>{param}</code></td><td>{val}</td></tr>"""

    all_params_html = f"""
        <details>
            <summary style="cursor: pointer; color: #1976D2; font-weight: bold; margin-top: 15px;">All Staged Training Parameters ({len(all_params)} parameters)</summary>
            <table style="margin-top: 10px;">
                <tr><th>Parameter</th><th>Value</th></tr>
                {all_rows}
            </table>
        </details>"""

    # Build world model architecture config table
    from config import AutoencoderConcatPredictorWorldModelConfig as WMConfig
    import config as cfg_module
    wm_rows = ""
    for attr in sorted(vars(WMConfig)):
        if attr.startswith('_'):
            continue
        val = getattr(WMConfig, attr)
        wm_rows += f"""
            <tr><td><code>{attr}</code></td><td>{val}</td></tr>"""
    # Add module-level mask ratio configs
    for attr in ['MASK_RATIO_MIN', 'MASK_RATIO_MAX', 'TRAIN_MASK_RATIO_MIN', 'TRAIN_MASK_RATIO_MAX']:
        val = getattr(cfg_module, attr, None)
        if val is not None:
            wm_rows += f"""
            <tr><td><code>{attr}</code></td><td>{val}</td></tr>"""

    wm_config_html = f"""
        <details>
            <summary style="cursor: pointer; color: #1976D2; font-weight: bold; margin-top: 15px;">World Model Architecture (config.py)</summary>
            <table style="margin-top: 10px;">
                <tr><th>Parameter</th><th>Value</th></tr>
                {wm_rows}
            </table>
        </details>"""

    return f"""
    <section id="configuration">
        <h2>Configuration</h2>
        {changes_html}
        {all_params_html}
        {wm_config_html}
    </section>
"""


def create_full_training_loss_plot(
    completed_stages: list[tuple[int, "StageResult", Optional["StageResult"]]],
) -> tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    """
    Create loss plots across the full training run over all stages.

    Returns (full_figure, zoomed_figure):
    - full_figure: Complete loss timeline with stage boundaries
    - zoomed_figure: Zoomed view starting after initial drop to show plateau/learning
    """
    all_samples = []
    all_train_loss = []
    all_val_loss = []
    stage_boundaries = []  # (sample_offset, stage_num)
    cumulative_offset = 0

    for stage_num, staged, _ in completed_stages:
        metrics = staged.cumulative_metrics
        if not metrics:
            continue

        samples = metrics.get("samples_seen", [])
        train_loss = metrics.get("loss_at_sample", [])
        val_loss = metrics.get("val_loss_at_sample", [])

        if not samples or not train_loss:
            continue

        stage_boundaries.append((cumulative_offset, stage_num))

        for i, s in enumerate(samples):
            all_samples.append(s + cumulative_offset)
            if i < len(train_loss):
                all_train_loss.append(train_loss[i])
            if i < len(val_loss):
                all_val_loss.append(val_loss[i])

        # Offset for next stage
        if samples:
            cumulative_offset += samples[-1]

    if not all_samples:
        return None, None

    # Figure 1: Full timeline
    fig_full, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(all_samples[:len(all_train_loss)], all_train_loss, 'b-', alpha=0.7, linewidth=1, label='Train Loss')
    if all_val_loss:
        ax1.plot(all_samples[:len(all_val_loss)], all_val_loss, 'r-', alpha=0.7, linewidth=1, label='Val Loss')
    for offset, stage_num in stage_boundaries:
        ax1.axvline(x=offset, color='gray', linestyle='--', alpha=0.5)
        ax1.text(offset, ax1.get_ylim()[1] * 0.95, f' S{stage_num}', fontsize=8, color='gray', va='top')
    ax1.set_xlabel('Cumulative Samples')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Across Full Training Run')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Figure 2: Zoomed view - skip initial drop
    # Heuristic: start from where loss drops below 2x the final loss or median, whichever is lower
    reference_loss = all_train_loss[-1] if all_train_loss else 0
    median_loss = float(np.median(all_train_loss)) if all_train_loss else 0
    threshold = min(2.0 * reference_loss, median_loss)

    zoom_start_idx = 0
    for i, loss in enumerate(all_train_loss):
        if loss <= threshold:
            zoom_start_idx = max(0, i - 5)  # Include a few points before threshold
            break

    # Only create zoomed view if we're actually skipping something
    fig_zoomed = None
    if zoom_start_idx > len(all_train_loss) * 0.05:  # At least 5% of data is skipped
        fig_zoomed, ax2 = plt.subplots(figsize=(12, 6))
        zoomed_samples = all_samples[zoom_start_idx:len(all_train_loss)]
        zoomed_train = all_train_loss[zoom_start_idx:]
        ax2.plot(zoomed_samples, zoomed_train, 'b-', alpha=0.7, linewidth=1, label='Train Loss')
        if all_val_loss and zoom_start_idx < len(all_val_loss):
            zoomed_val = all_val_loss[zoom_start_idx:]
            ax2.plot(all_samples[zoom_start_idx:zoom_start_idx + len(zoomed_val)], zoomed_val, 'r-', alpha=0.7, linewidth=1, label='Val Loss')
        for offset, stage_num in stage_boundaries:
            if offset >= all_samples[zoom_start_idx]:
                ax2.axvline(x=offset, color='gray', linestyle='--', alpha=0.5)
                ax2.text(offset, ax2.get_ylim()[1] * 0.95, f' S{stage_num}', fontsize=8, color='gray', va='top')
        ax2.set_xlabel('Cumulative Samples')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Detail (Post Initial Drop)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    return fig_full, fig_zoomed


def generate_multi_run_stats_html(
    all_stage_runs: dict[int, list["StageResult"]],
    completed_stages: list[tuple[int, "StageResult", Optional["StageResult"]]],
) -> str:
    """
    Generate HTML section with statistics across all runs per stage.

    Only generates content if any stage has more than 1 run.
    """
    # Check if any stage has multiple runs
    has_multi_runs = any(len(runs) > 1 for runs in all_stage_runs.values())
    if not has_multi_runs:
        return ""

    # Build best run lookup
    best_runs = {}
    for stage_num, staged, _ in completed_stages:
        best_runs[stage_num] = staged

    sections = []
    total_runs = 0
    all_best_losses = []

    for stage_num in sorted(all_stage_runs.keys()):
        runs = all_stage_runs[stage_num]
        if not runs:
            continue

        best = best_runs.get(stage_num)
        valid_losses = [r.best_loss for r in runs if r.best_loss < float('inf')]
        total_runs += len(runs)
        all_best_losses.extend(valid_losses)

        # Per-run table
        run_rows = ""
        for r in sorted(runs, key=lambda x: x.run_num):
            is_best = best and r.best_checkpoint_path == best.best_checkpoint_path
            selected = "&#10003;" if is_best else ""
            row_style = ' style="background: #e8f5e9; font-weight: bold;"' if is_best else ''
            run_rows += f"""
                <tr{row_style}>
                    <td>{r.run_num}</td>
                    <td>{format_loss_safe(r.best_loss)}</td>
                    <td>{r.stop_reason}</td>
                    <td>{r.total_samples_trained:,}</td>
                    <td>{format_duration(r.elapsed_time)}</td>
                    <td>{selected}</td>
                </tr>"""

        # Aggregate stats
        if valid_losses:
            mean_loss = np.mean(valid_losses)
            std_loss = np.std(valid_losses)
            min_loss = np.min(valid_losses)
            max_loss = np.max(valid_losses)
            run_rows += f"""
                <tr style="background: #e0e0e0; font-weight: bold; font-style: italic;">
                    <td colspan="2">Mean: {mean_loss:.6f} &plusmn; {std_loss:.6f}</td>
                    <td colspan="2">Min: {min_loss:.6f} / Max: {max_loss:.6f}</td>
                    <td colspan="2">Range: {max_loss - min_loss:.6f}</td>
                </tr>"""

        sections.append(f"""
        <h3>Stage {stage_num} ({len(runs)} runs)</h3>
        <table>
            <tr>
                <th>Run</th>
                <th>Best Loss</th>
                <th>Stop Reason</th>
                <th>Samples</th>
                <th>Time</th>
                <th>Selected</th>
            </tr>
            {run_rows}
        </table>""")

    # Overall summary
    overall_html = ""
    if all_best_losses:
        overall_html = f"""
        <div class="metric">
            <strong>Total Runs:</strong> {total_runs}<br>
            <strong>Average Best Loss:</strong> {np.mean(all_best_losses):.6f} &plusmn; {np.std(all_best_losses):.6f}<br>
            <strong>Best Overall:</strong> {np.min(all_best_losses):.6f}<br>
            <strong>Worst Overall:</strong> {np.max(all_best_losses):.6f}
        </div>"""

    return f"""
    <section id="multi-run-stats">
        <h2>Multi-Run Statistics</h2>
        {overall_html}
        {"".join(sections)}
    </section>
"""


def get_available_actions(session: dict) -> list[int]:
    """Get list of available action indices from session."""
    action_space = session.get("action_space", [])
    if not action_space:
        # Default to 2 actions (toroidal dot)
        return [0, 1]

    # Extract action indices
    actions = []
    for i, action in enumerate(action_space):
        if isinstance(action, dict) and "action" in action:
            actions.append(action["action"])
        else:
            actions.append(i)
    return sorted(set(actions))


def generate_counterfactual_images(
    session: dict,
    frame_idx: int,
    output_dir: Path,
    prefix: str = "frame",
) -> list[tuple[int, str]]:
    """
    Generate counterfactual inference images for all actions.

    Returns list of (action, base64_image) tuples.
    """
    available_actions = get_available_actions(session)
    results = []

    for action in available_actions:
        try:
            result = inference.run_counterfactual_inference(frame_idx, action)
            if result and len(result) >= 5:
                # result[4] is fig_diff_heatmap
                _, fig_true, fig_cf, fig_true_inf, fig_cf_inf, fig_diff, _ = result

                # Save composite (true inference shows the prediction)
                if fig_true_inf is not None:
                    img_path = output_dir / f"{prefix}_{frame_idx}_action{action}.png"
                    save_fig_to_file(fig_true_inf, img_path)
                    results.append((action, fig_to_base64(fig_cf_inf) if fig_cf_inf else ""))

                # Close other figures
                for f in [fig_true, fig_cf, fig_true_inf, fig_diff]:
                    if f is not None:
                        plt.close(f)
        except Exception as e:
            print(f"  Error generating counterfactual for action {action}: {e}")

    return results


def evaluate_session_with_checkpoint(
    checkpoint_path: str,
    session: dict,
) -> tuple[Optional[plt.Figure], Optional[plt.Figure], dict]:
    """
    Evaluate a checkpoint on a session.

    Returns (loss_over_time_fig, distribution_fig, stats_dict)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=state.device, weights_only=False)
    state.world_model.autoencoder.load_state_dict(checkpoint["model_state_dict"])

    # Temporarily set session state
    old_session = state.session_state
    state.session_state = session

    try:
        result = evaluation.evaluate_full_session()
        if result and len(result) >= 5:
            status, fig_loss, fig_dist, stats_text, stats = result
            return fig_loss, fig_dist, stats
    except Exception as e:
        print(f"  Evaluation error: {e}")
    finally:
        state.session_state = old_session

    return None, None, {}


def generate_stage_report(
    result: StageResult,
    train_session: dict,
    val_session: dict,
    original_session: dict,
    cfg: StagedTrainingConfig,
    output_dir: Path,
) -> None:
    """Generate HTML report for a stage run."""
    print(f"\nGenerating stage {result.stage_num} run {result.run_num} report...")

    # Check for valid checkpoint path
    if not result.best_checkpoint_path:
        print(f"  WARNING: No checkpoint available for run {result.run_num} (stop_reason: {result.stop_reason})")
        print(f"  Skipping report generation for this run.")
        return

    # Create assets directory
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Collect images for HTML
    images = {}

    # Generate training progress plots from cumulative metrics
    if result.cumulative_metrics.get("samples_seen"):
        fig_progress = visualization.create_loss_vs_samples_plot(result.cumulative_metrics)
        if fig_progress:
            save_fig_to_file(fig_progress, assets_dir / "training_progress.png")
            images["training_progress"] = fig_to_base64(
                visualization.create_loss_vs_samples_plot(result.cumulative_metrics)
            )

        fig_rolling = visualization.create_loss_vs_recent_checkpoints_plot(
            result.cumulative_metrics, cfg.window_size
        )
        if fig_rolling:
            save_fig_to_file(fig_rolling, assets_dir / "training_progress_rolling.png")
            images["training_rolling"] = fig_to_base64(
                visualization.create_loss_vs_recent_checkpoints_plot(
                    result.cumulative_metrics, cfg.window_size
                )
            )

        # Generate learning rate graph
        fig_lr = visualization.create_lr_vs_samples_plot(result.cumulative_metrics)
        if fig_lr:
            save_fig_to_file(fig_lr, assets_dir / "learning_rate.png")
            images["learning_rate"] = fig_to_base64(
                visualization.create_lr_vs_samples_plot(result.cumulative_metrics)
            )

    # Load best checkpoint for evaluations
    setup_world_model(train_session, result.best_checkpoint_path, cfg)
    state.validation_session_state = val_session

    # Generate sample weights plot if available
    if hasattr(state, "sample_losses") and state.sample_losses:
        # Compute valid indices based on canvas history size
        min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
        num_observations = len(train_session.get("observations", []))
        all_valid_indices = list(range(min_frames_needed - 1, num_observations))

        if all_valid_indices:
            weights, _ = state.compute_sample_weights(cfg.loss_weight_temperature, all_valid_indices)
            if weights is not None:
                frame_indices = list(range(len(weights)))
                fig_weights = visualization.create_sample_weights_plot(
                    weights, frame_indices, cfg.loss_weight_temperature
                )
                if fig_weights:
                    save_fig_to_file(fig_weights, assets_dir / "sample_weights.png")
                    images["sample_weights"] = fig_to_base64(fig_weights)

    # Generate sample counts plot from cumulative metrics
    sample_seen_counts = result.cumulative_metrics.get("sample_seen_counts", {})
    if sample_seen_counts:
        fig_counts = visualization.create_sample_counts_plot(sample_seen_counts)
        if fig_counts:
            save_fig_to_file(fig_counts, assets_dir / "sample_counts.png")
            images["sample_counts"] = fig_to_base64(fig_counts)

    # Generate inference images for selected frame
    selected_frame = cfg.selected_frame_offset
    print(f"  Generating inference for frame {selected_frame}...")
    inference_selected_dir = assets_dir / "inference_selected"
    inference_selected_dir.mkdir(exist_ok=True)
    selected_images = generate_counterfactual_images(
        train_session, selected_frame, inference_selected_dir, "selected"
    )

    # Generate inference for random observations
    print(f"  Generating inference for {cfg.num_random_obs_to_visualize} random observations...")
    inference_random_dir = assets_dir / "inference_random"
    inference_random_dir.mkdir(exist_ok=True)

    min_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    valid_indices = list(range(min_frames - 1, len(train_session["observations"])))
    random_indices = random.sample(valid_indices, min(cfg.num_random_obs_to_visualize, len(valid_indices)))

    random_images = {}
    for idx in random_indices:
        imgs = generate_counterfactual_images(train_session, idx, inference_random_dir, f"obs_{idx}")
        random_images[idx] = imgs

    # Evaluate on train, validation, and original sessions
    print("  Evaluating on training session...")
    fig_train_loss, fig_train_dist, train_stats = evaluate_session_with_checkpoint(
        result.best_checkpoint_path, train_session
    )
    if fig_train_loss:
        save_fig_to_file(fig_train_loss, assets_dir / "hybrid_loss_train.png")
        images["loss_train"] = fig_to_base64(fig_train_loss)

    print("  Evaluating on validation session...")
    fig_val_loss, fig_val_dist, val_stats = evaluate_session_with_checkpoint(
        result.best_checkpoint_path, val_session
    )
    if fig_val_loss:
        save_fig_to_file(fig_val_loss, assets_dir / "hybrid_loss_validate.png")
        images["loss_validate"] = fig_to_base64(fig_val_loss)

    print("  Evaluating on original session...")
    fig_orig_loss, fig_orig_dist, orig_stats = evaluate_session_with_checkpoint(
        result.best_checkpoint_path, original_session
    )
    if fig_orig_loss:
        save_fig_to_file(fig_orig_loss, assets_dir / "hybrid_loss_original.png")
        images["loss_original"] = fig_to_base64(fig_orig_loss)

    # Save metrics.json
    metrics = {
        "stage_num": result.stage_num,
        "run_num": result.run_num,
        "stop_reason": result.stop_reason,
        "elapsed_time_seconds": result.elapsed_time,
        "best_checkpoint": result.best_checkpoint_path,
        "best_loss": result.best_loss,
        "total_samples_trained": result.total_samples_trained,
        "train_eval_stats": train_stats,
        "val_eval_stats": val_stats,
        "original_eval_stats": orig_stats,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Generate HTML report
    html = generate_stage_html(
        result=result,
        images=images,
        selected_frame=selected_frame,
        selected_images=selected_images,
        random_images=random_images,
        train_stats=train_stats,
        val_stats=val_stats,
        orig_stats=orig_stats,
        available_actions=get_available_actions(train_session),
    )

    with open(output_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Report saved to {output_dir / 'report.html'}")


def generate_sweep_history_html(sweep_history: list) -> str:
    """Generate HTML table for sweep history."""
    if not sweep_history:
        return "<p>No plateau-triggered sweeps occurred during this training run.</p>"

    rows = ""
    for sweep in sweep_history:
        # Format improvement percentage with sign
        improvement_pct = sweep.get('improvement_pct', 0)
        improvement_str = f"{improvement_pct:+.2f}%" if improvement_pct != 0 else "0.00%"
        # Color code: green for improvement, red for regression
        if improvement_pct > 0:
            improvement_cell = f'<td style="color: green;">{improvement_str}</td>'
        elif improvement_pct < 0:
            improvement_cell = f'<td style="color: red;">{improvement_str}</td>'
        else:
            improvement_cell = f'<td>{improvement_str}</td>'

        rows += f"""
        <tr>
            <td>{sweep.get('sweep_num', 'N/A')}</td>
            <td>{sweep.get('consecutive_num', 'N/A')}</td>
            <td>{sweep.get('triggered_at_samples', 'N/A'):,}</td>
            <td>{sweep.get('val_loss_at_trigger', 0):.6f}</td>
            <td>{sweep.get('val_loss_after', 0):.6f}</td>
            {improvement_cell}
            <td>{sweep.get('selected_lr', 0):.2e}</td>
            <td>{format_duration(sweep.get('sweep_duration_sec', 0))}</td>
        </tr>"""

    return f"""
    <table class="stats-table">
        <tr>
            <th>Sweep #</th>
            <th>Consec #</th>
            <th>Triggered At</th>
            <th>Val Loss Before</th>
            <th>Val Loss After</th>
            <th>Improvement</th>
            <th>Selected LR</th>
            <th>Duration</th>
        </tr>
        {rows}
    </table>
    """


def generate_plateau_sweep_summary_html(
    completed_stages: list[tuple[int, StageResult, Optional[StageResult]]],
) -> str:
    """
    Generate HTML summary of plateau sweeps across all stages.

    Args:
        completed_stages: List of (stage_num, staged_result, baseline_result) tuples

    Returns:
        HTML string with summary statistics and per-stage sweep tables.
    """
    sweep_data = aggregate_sweep_data(completed_stages)

    if sweep_data['total_sweeps'] == 0:
        return "<p>No plateau-triggered sweeps occurred during training.</p>"

    # Summary metrics box
    summary_html = f"""
    <div class="metric">
        <strong>Total Sweeps:</strong> {sweep_data['total_sweeps']}<br>
        <strong>Stages with Sweeps:</strong> {sweep_data['stages_with_sweeps']} of {len(sweep_data['per_stage'])}<br>
        <strong>Total Sweep Time:</strong> {format_duration(sweep_data['total_sweep_time_sec'])}<br>
        <strong>Average Sweep Duration:</strong> {format_duration(sweep_data['avg_sweep_duration_sec'])}
    </div>
    """

    # Per-stage details
    per_stage_html = ""
    for stage_info in sweep_data['per_stage']:
        stage_num = stage_info['stage_num']
        sweep_count = stage_info['sweep_count']
        initial_lr = stage_info['initial_lr']
        final_lr = stage_info['final_lr']
        sweep_history = stage_info['sweep_history']

        # Build LR progression string
        if sweep_count > 0 and initial_lr > 0:
            lr_progression = f"{initial_lr:.1e}"
            for sweep in sorted(sweep_history, key=lambda s: s.get('sweep_num', 0)):
                lr_progression += f" → {sweep.get('selected_lr', 0):.1e}"
        elif initial_lr > 0:
            lr_progression = f"{initial_lr:.1e} (unchanged)"
        else:
            lr_progression = "N/A"

        per_stage_html += f"""
        <h3>Stage {stage_num}: {sweep_count} sweep{'s' if sweep_count != 1 else ''}</h3>
        <p><strong>LR Progression:</strong> {lr_progression}</p>
        """

        if sweep_count > 0:
            # Generate table for this stage's sweeps
            rows = ""
            for sweep in sorted(sweep_history, key=lambda s: s.get('sweep_num', 0)):
                rows += f"""
                <tr>
                    <td>{sweep.get('sweep_num', 'N/A')}</td>
                    <td>{sweep.get('triggered_at_samples', 0):,}</td>
                    <td>{format_duration(sweep.get('triggered_at_time', 0))}</td>
                    <td>{sweep.get('selected_lr', 0):.2e}</td>
                    <td>{format_duration(sweep.get('sweep_duration_sec', 0))}</td>
                </tr>"""

            per_stage_html += f"""
            <table>
                <tr>
                    <th>Sweep #</th>
                    <th>Triggered At (samples)</th>
                    <th>Wall Time</th>
                    <th>Selected LR</th>
                    <th>Duration</th>
                </tr>
                {rows}
            </table>
            """

    return summary_html + per_stage_html


def generate_stage_html(
    result: StageResult,
    images: dict,
    selected_frame: int,
    selected_images: list,
    random_images: dict,
    train_stats: dict,
    val_stats: dict,
    orig_stats: dict,
    available_actions: list[int],
) -> str:
    """Generate HTML content for stage report."""
    elapsed_str = format_duration(result.elapsed_time)

    # Build inference sections
    selected_inference_html = ""
    for action, b64 in selected_images:
        if b64:
            selected_inference_html += f'<div class="inference-item"><h4>Action {action}</h4><img src="data:image/png;base64,{b64}" /></div>'

    random_inference_html = ""
    for idx, imgs in random_images.items():
        random_inference_html += f'<div class="obs-group"><h4>Observation {idx}</h4><div class="inference-row">'
        for action, b64 in imgs:
            if b64:
                random_inference_html += f'<div class="inference-item"><span>Action {action}</span><img src="data:image/png;base64,{b64}" /></div>'
        random_inference_html += '</div></div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Stage {result.stage_num} Run {result.run_num} Training Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; }}
        .metric {{ background: #f0f7ff; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        .metric strong {{ color: #1976D2; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .inference-row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .inference-item {{ flex: 1; min-width: 200px; text-align: center; }}
        .inference-item img {{ max-width: 300px; }}
        .obs-group {{ margin: 20px 0; padding: 15px; background: #fafafa; border-radius: 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .stats-table {{ max-width: 400px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Stage {result.stage_num} Run {result.run_num} Training Summary</h1>

    <section id="training-stopped">
        <h2>Training Stopped</h2>
        <div class="metric"><strong>Reason:</strong> {result.stop_reason}</div>
    </section>

    <section id="training-time">
        <h2>Training Time</h2>
        <div class="metric"><strong>Duration:</strong> {elapsed_str}</div>
    </section>

    <section id="best-checkpoint">
        <h2>Best Checkpoint</h2>
        <div class="metric">
            <strong>Name:</strong> {Path(result.best_checkpoint_path).name}<br>
            <strong>Hybrid Loss:</strong> {format_loss_safe(result.best_loss)}
        </div>
    </section>

    <section id="lr-sweeps">
        <h2>Learning Rate Sweeps</h2>
        <div class="metric">
            <strong>Initial LR:</strong> {result.initial_lr:.2e}<br>
            <strong>Final LR:</strong> {result.final_lr:.2e}<br>
            <strong>Total Sweeps Triggered:</strong> {result.total_sweeps_triggered}
        </div>
        {generate_sweep_history_html(result.sweep_history)}
    </section>

    <section id="training-progress">
        <h2>Training Progress</h2>
        {'<img src="data:image/png;base64,' + images.get("training_progress", "") + '" />' if images.get("training_progress") else "<p>No training progress data available</p>"}
        {'<img src="data:image/png;base64,' + images.get("training_rolling", "") + '" />' if images.get("training_rolling") else ""}
        <h3>Learning Rate Schedule</h3>
        {'<img src="data:image/png;base64,' + images.get("learning_rate", "") + '" />' if images.get("learning_rate") else "<p>No learning rate data available</p>"}
    </section>

    <section id="sample-weights">
        <h2>Sample Weight Distribution</h2>
        {'<img src="data:image/png;base64,' + images.get("sample_weights", "") + '" />' if images.get("sample_weights") else "<p>No sample weight data available</p>"}
    </section>

    <section id="sample-counts">
        <h2>Sample Counts</h2>
        {'<img src="data:image/png;base64,' + images.get("sample_counts", "") + '" />' if images.get("sample_counts") else "<p>No sample count data available</p>"}
    </section>

    <section id="inference-selected">
        <h2>Inference: Selected Frame {selected_frame}</h2>
        <div class="inference-row">
            {selected_inference_html if selected_inference_html else "<p>No inference images available</p>"}
        </div>
    </section>

    <section id="inference-random">
        <h2>Inference: Random Observations</h2>
        {random_inference_html if random_inference_html else "<p>No random observation images available</p>"}
    </section>

    <section id="hybrid-loss">
        <h2>Hybrid Loss Over Observations</h2>

        <h3>Training Session</h3>
        {'<img src="data:image/png;base64,' + images.get("loss_train", "") + '" />' if images.get("loss_train") else "<p>No evaluation data</p>"}
        {format_stats_table(train_stats, "Training")}

        <h3>Validation Session</h3>
        {'<img src="data:image/png;base64,' + images.get("loss_validate", "") + '" />' if images.get("loss_validate") else "<p>No evaluation data</p>"}
        {format_stats_table(val_stats, "Validation")}

        <h3>Original Session</h3>
        {'<img src="data:image/png;base64,' + images.get("loss_original", "") + '" />' if images.get("loss_original") else "<p>No evaluation data</p>"}
        {format_stats_table(orig_stats, "Original")}
    </section>
</div>
</body>
</html>"""

    return html


def format_stats_table(stats: dict, title: str) -> str:
    """Format evaluation stats as HTML table."""
    if not stats:
        return ""

    hybrid = stats.get("hybrid", {})
    if not hybrid:
        return ""

    return f"""
    <table class="stats-table">
        <tr><th colspan="2">{title} Statistics</th></tr>
        <tr><td>Mean</td><td>{hybrid.get('mean', 'N/A'):.6f}</td></tr>
        <tr><td>Median</td><td>{hybrid.get('median', 'N/A'):.6f}</td></tr>
        <tr><td>Std Dev</td><td>{hybrid.get('std', 'N/A'):.6f}</td></tr>
        <tr><td>Min</td><td>{hybrid.get('min', 'N/A'):.6f}</td></tr>
        <tr><td>Max</td><td>{hybrid.get('max', 'N/A'):.6f}</td></tr>
        <tr><td>P25</td><td>{hybrid.get('p25', 'N/A'):.6f}</td></tr>
        <tr><td>P75</td><td>{hybrid.get('p75', 'N/A'):.6f}</td></tr>
    </table>"""


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_loss_safe(value: float, default: str = "N/A", signed: bool = False) -> str:
    """Format loss value, handling inf/nan gracefully.

    Args:
        value: The loss value to format
        default: Default string to return for inf/nan/None values
        signed: If True, prefix positive values with '+'
    """
    import math
    if value is None or math.isinf(value) or math.isnan(value):
        return default
    if signed and value >= 0:
        return f"+{value:.6f}"
    return f"{value:.6f}"


def generate_final_report(
    completed_stages: list[tuple[int, StageResult, Optional[StageResult]]],
    all_sweep_results: list,
    all_timings: list,
    output_dir: Path,
    run_id: str,
    overall_start_time: float,
    original_session: Optional[dict],
    cfg: StagedTrainingConfig,
    is_final: bool = False,
    all_stage_runs: Optional[dict[int, list[StageResult]]] = None,
) -> str:
    """
    Generate unified final report (updated progressively after each stage).

    This function combines timing/LR sweep info (available immediately) with
    evaluation/inference data (available only at the end).

    Args:
        completed_stages: List of (stage_num, staged_result, baseline_result) tuples
        all_sweep_results: List of LRSweepStageResult
        all_timings: List of StageTiming
        output_dir: Output directory
        run_id: Run identifier
        overall_start_time: Start time of entire training run
        original_session: Original session for evaluation (optional until final)
        cfg: Training configuration
        is_final: True if this is the final call (triggers inference generation)

    Returns:
        Path to generated report
    """
    # Generate dated filename with deduplication for same-day reports
    # Short name in output_dir (run_id already in parent directory path, avoids Windows MAX_PATH)
    # Full name with run_id only for docs/ copy where it's needed for identification
    date_str = datetime.now().strftime("%Y_%b_%d").lower()
    local_base = f"final_report_{date_str}"
    docs_base = f"final_report_{run_id}_{date_str}"

    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Dedup based on docs directory
    counter = 1
    docs_filename = f"{docs_base}.html"
    local_filename = f"{local_base}.html"
    while (docs_dir / docs_filename).exists():
        counter += 1
        docs_filename = f"{docs_base}_{counter}.html"
        local_filename = f"{local_base}_{counter}.html"

    report_path = output_dir / local_filename

    # Calculate total elapsed time
    total_elapsed = time.time() - overall_start_time

    # Build timing summary table (using plateau sweep data from stage results)
    timing_rows = ""
    total_sweep_count = 0
    total_sweep_time = 0.0
    total_main_time = 0.0
    total_stage_time = 0.0

    for stage_num, staged, _ in completed_stages:
        # Calculate plateau sweep time from sweep_history
        sweep_history = staged.sweep_history if hasattr(staged, 'sweep_history') else []
        sweep_time = sum(s.get('sweep_duration_sec', 0) for s in sweep_history)
        sweep_count = len(sweep_history)

        # Main training time = stage time minus sweep time
        main_time = staged.elapsed_time - sweep_time

        total_sweep_count += sweep_count
        total_sweep_time += sweep_time
        total_main_time += main_time
        total_stage_time += staged.elapsed_time

        timing_rows += f"""
        <tr>
            <td>Stage {stage_num}</td>
            <td>{sweep_count}</td>
            <td>{format_duration(sweep_time)}</td>
            <td>{format_duration(main_time)}</td>
            <td>{format_duration(staged.elapsed_time)}</td>
        </tr>
        """

    timing_rows += f"""
    <tr style="background: #e0e0e0; font-weight: bold;">
        <td>TOTAL</td>
        <td>{total_sweep_count}</td>
        <td>{format_duration(total_sweep_time)}</td>
        <td>{format_duration(total_main_time)}</td>
        <td>{format_duration(total_stage_time)}</td>
    </tr>
    """

    # Add initial sweep info (from all_sweep_results, which tracks upfront LR sweeps)
    initial_sweep_note = ""
    if all_sweep_results:
        sweep_notes = []
        for sr in all_sweep_results:
            sweep_notes.append(
                f"Stage {sr.stage_num}: selected LR {sr.selected_lr:.2e} in {format_duration(sr.total_wall_time_sec)}"
            )
        initial_sweep_note = f'<p><strong>Initial LR Sweep:</strong> {"; ".join(sweep_notes)}</p>'

    # Build plateau sweep details (default mode - sweeps triggered on plateau during training)
    lr_details_html = generate_plateau_sweep_summary_html(completed_stages)

    # Build basic stage results table (always available)
    stage_results_rows = ""
    total_plateau_sweeps = 0
    stop_reason_counts = {}
    for stage_num, staged, baseline in completed_stages:
        # Track plateau sweeps
        sweeps = staged.total_sweeps_triggered if hasattr(staged, 'total_sweeps_triggered') else 0
        total_plateau_sweeps += sweeps
        lr_info = f"{staged.initial_lr:.1e}→{staged.final_lr:.1e}" if hasattr(staged, 'initial_lr') and staged.initial_lr != staged.final_lr else f"{staged.initial_lr:.1e}" if hasattr(staged, 'initial_lr') else "N/A"

        # Track stop reasons
        reason_key = staged.stop_reason.split("(")[0].strip()  # Get base reason
        stop_reason_counts[reason_key] = stop_reason_counts.get(reason_key, 0) + 1

        stage_results_rows += f"""
        <tr>
            <td>Stage {stage_num}</td>
            <td>{format_loss_safe(staged.best_loss)}</td>
            <td>{staged.stop_reason}</td>
            <td>{staged.total_samples_trained:,}</td>
            <td>{format_duration(staged.elapsed_time)}</td>
            <td>{sweeps}</td>
            <td>{lr_info}</td>
        </tr>
        """

    # Build stop reason breakdown
    stop_reason_html = "<ul>" + "".join(f"<li><strong>{reason}:</strong> {count} stages</li>" for reason, count in sorted(stop_reason_counts.items())) + "</ul>"

    # === Sections that require original_session evaluation (only on final call) ===
    best_checkpoint_html = ""
    stage_progression_html = ""
    comparison_html = ""
    sample_counts_html = ""
    hybrid_loss_html = ""
    inference_html = ""
    stage_links_html = ""

    if is_final and original_session:
        print("\n" + "=" * 60)
        print("Generating Final Summary Report")
        print("=" * 60)

        # Convert completed_stages to list of tuples for evaluation
        staged_results = [(stage_num, staged) for stage_num, staged, _ in completed_stages]
        baseline_results = [(stage_num, baseline) for stage_num, _, baseline in completed_stages if baseline]

        # Evaluate each stage's STAGED checkpoint on the original session
        print("Evaluating staged checkpoints on original session...")
        stage_original_evals = []  # List of (stage_num, result, orig_loss, loss_fig_b64)

        for stage_num, result in staged_results:
            print(f"  Evaluating stage {stage_num}...")
            setup_world_model(original_session, result.best_checkpoint_path, cfg)
            fig_loss, fig_dist, stats = evaluate_session_with_checkpoint(
                result.best_checkpoint_path, original_session
            )
            orig_loss = stats.get("hybrid", {}).get("mean", float("inf"))
            loss_fig_b64 = fig_to_base64(fig_loss) if fig_loss else ""
            if fig_loss:
                plt.close(fig_loss)
            if fig_dist:
                plt.close(fig_dist)
            stage_original_evals.append((stage_num, result, orig_loss, loss_fig_b64))
            print(f"    Original session hybrid loss: {orig_loss:.6f}")

        # Find best staged checkpoint based on original session loss
        best_eval = min(stage_original_evals, key=lambda x: x[2])
        best_stage, best_run_result, best_orig_loss, _ = best_eval
        print(f"Best staged checkpoint: Stage {best_stage} with original session loss {best_orig_loss:.6f}")

        # Evaluate BASELINE checkpoints on original session (if baseline was run)
        baseline_original_evals = []
        comparisons = []
        comparison_plot_b64 = ""

        if baseline_results:
            print("\nEvaluating baseline checkpoints on original session...")
            for stage_num, result in baseline_results:
                print(f"  Evaluating baseline stage {stage_num}...")
                setup_world_model(original_session, result.best_checkpoint_path, cfg)
                fig_loss, fig_dist, stats = evaluate_session_with_checkpoint(
                    result.best_checkpoint_path, original_session
                )
                orig_loss = stats.get("hybrid", {}).get("mean", float("inf"))
                loss_fig_b64 = fig_to_base64(fig_loss) if fig_loss else ""
                if fig_loss:
                    plt.close(fig_loss)
                if fig_dist:
                    plt.close(fig_dist)
                baseline_original_evals.append((stage_num, result, orig_loss, loss_fig_b64))
                print(f"    Original session hybrid loss: {orig_loss:.6f}")

            # Find best baseline checkpoint
            best_baseline_eval = min(baseline_original_evals, key=lambda x: x[2])
            best_baseline_stage, best_baseline_result, best_baseline_orig_loss, _ = best_baseline_eval
            print(f"Best baseline checkpoint: Stage {best_baseline_stage} with original session loss {best_baseline_orig_loss:.6f}")

            # Compute comparisons for each stage
            for (staged_stage, staged_result, staged_loss, _), (baseline_stage, baseline_result, baseline_loss, _) in zip(
                stage_original_evals, baseline_original_evals
            ):
                comparison = compute_stage_comparison(
                    staged_result, baseline_result, staged_loss, baseline_loss
                )
                comparisons.append(comparison)

            # Create comparison progression plot
            fig_comparison = create_staged_vs_baseline_plot(stage_original_evals, baseline_original_evals)
            comparison_plot_b64 = fig_to_base64(fig_comparison)
            plt.close(fig_comparison)

        # Best checkpoint section
        best_checkpoint_html = f"""
    <section id="best-checkpoint">
        <h2>Best Checkpoint</h2>
        <div class="metric">
            <strong>Name:</strong> {Path(best_run_result.best_checkpoint_path).name}<br>
            <strong>Stage:</strong> {best_stage}<br>
            <strong>Hybrid Loss (full session):</strong> {format_loss_safe(best_orig_loss)}
        </div>
    </section>
"""

        # Learning rate timeline with plateau sweeps across all stages
        lr_b64 = ""
        fig_lr = create_plateau_sweep_timeline_plot(completed_stages)
        if fig_lr:
            lr_b64 = fig_to_base64(fig_lr)
            plt.close(fig_lr)

        lr_schedule_html = f"""
    <section id="lr-timeline">
        <h2>Learning Rate Timeline with Plateau Sweeps</h2>
        {'<img src="data:image/png;base64,' + lr_b64 + '" />' if lr_b64 else "<p>No plateau sweep data available</p>"}
    </section>
""" if lr_b64 else ""

        # Stage progression with original session losses
        fig_progression, ax = plt.subplots(figsize=(10, 6))
        stages = [s for s, _, _, _ in stage_original_evals]
        orig_losses = [loss for _, _, loss, _ in stage_original_evals]
        ax.plot(stages, orig_losses, "o-", linewidth=2, markersize=8, label="Staged (weight carryover)")
        if baseline_original_evals:
            baseline_losses = [loss for _, _, loss, _ in baseline_original_evals]
            ax.plot(stages, baseline_losses, "s--", linewidth=2, markersize=8, label="Baseline (fresh each stage)")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Hybrid Loss (Original Session)")
        ax.set_title("Stage Progression: Loss on Original Session")
        ax.grid(True, alpha=0.3)
        ax.legend()
        progression_b64 = fig_to_base64(fig_progression)

        # Build stage table with original session loss
        stage_progression_rows = ""
        for stage_num, result, orig_loss, _ in stage_original_evals:
            is_best = stage_num == best_stage
            row_style = ' style="background: #e8f5e9; font-weight: bold;"' if is_best else ''
            best_marker = " ⭐" if is_best else ""
            stage_progression_rows += f"""
        <tr{row_style}>
            <td>{stage_num}{best_marker}</td>
            <td>{format_loss_safe(orig_loss)}</td>
            <td>{format_loss_safe(result.best_loss)}</td>
            <td>{format_duration(result.elapsed_time)}</td>
            <td>{result.total_samples_trained}</td>
            <td>{result.stop_reason}</td>
        </tr>"""

        # Build stage loss graphs section
        stage_loss_graphs_html = ""
        for stage_num, result, orig_loss, loss_fig_b64 in stage_original_evals:
            if loss_fig_b64:
                is_best = stage_num == best_stage
                best_label = " (Best)" if is_best else ""
                stage_loss_graphs_html += f"""
            <div class="stage-loss-graph">
                <h4>Stage {stage_num}{best_label} - Hybrid Loss: {format_loss_safe(orig_loss)}</h4>
                <img src="data:image/png;base64,{loss_fig_b64}" />
            </div>"""

        stage_progression_html = f"""
    <section id="stage-progression">
        <h2>Stage Progression</h2>
        <img src="data:image/png;base64,{progression_b64}" />
        <table>
            <tr>
                <th>Stage</th>
                <th>Orig Loss</th>
                <th>Train Loss</th>
                <th>Time</th>
                <th>Samples</th>
                <th>Stop Reason</th>
            </tr>
            {stage_progression_rows}
        </table>
        <h3>Hybrid Loss Over Original Session (per Stage)</h3>
        <div class="stage-loss-graphs">
            {stage_loss_graphs_html if stage_loss_graphs_html else "<p>No loss graphs available</p>"}
        </div>
    </section>
"""

        # Comparison section (if baseline)
        if comparisons:
            stages_won_staged = sum(1 for c in comparisons if c.improvement_absolute > 0)
            stages_won_baseline = sum(1 for c in comparisons if c.improvement_absolute < 0)
            avg_improvement_ratio = sum(c.improvement_ratio for c in comparisons) / len(comparisons)
            avg_improvement_absolute = sum(c.improvement_absolute for c in comparisons) / len(comparisons)

            if stages_won_staged > stages_won_baseline:
                winner = "Staged Training"
                winner_color = "#4CAF50"
            elif stages_won_baseline > stages_won_staged:
                winner = "Baseline Training"
                winner_color = "#2196F3"
            else:
                winner = "Tie"
                winner_color = "#9E9E9E"

            comparison_rows = ""
            for c in comparisons:
                if c.improvement_absolute > 0:
                    stage_winner = "Staged"
                    winner_style = 'style="color: #4CAF50; font-weight: bold;"'
                elif c.improvement_absolute < 0:
                    stage_winner = "Baseline"
                    winner_style = 'style="color: #2196F3; font-weight: bold;"'
                else:
                    stage_winner = "Tie"
                    winner_style = 'style="color: #9E9E9E;"'

                comparison_rows += f"""
            <tr>
                <td>{c.stage_num}</td>
                <td>{format_loss_safe(c.staged_orig_loss)}</td>
                <td>{format_loss_safe(c.baseline_orig_loss)}</td>
                <td>{format_loss_safe(c.improvement_absolute, "N/A", signed=True)}</td>
                <td>{c.improvement_ratio:.3f}</td>
                <td>{c.staged_samples_trained:,}</td>
                <td>{c.baseline_samples_trained:,}</td>
                <td {winner_style}>{stage_winner}</td>
            </tr>"""

            avg_improvement_abs_str = format_loss_safe(avg_improvement_absolute, "N/A", signed=True)

            comparison_html = f"""
    <section id="staged-vs-baseline">
        <h2>Staged vs Baseline Comparison</h2>
        <div class="metric" style="background: #f0f7ff;">
            <strong>Overall Winner:</strong> <span style="color: {winner_color}; font-weight: bold;">{winner}</span><br>
            <strong>Stages Won (Staged):</strong> {stages_won_staged}<br>
            <strong>Stages Won (Baseline):</strong> {stages_won_baseline}<br>
            <strong>Average Improvement Ratio:</strong> {avg_improvement_ratio:.3f} (>1 = staged better)<br>
            <strong>Average Improvement:</strong> {avg_improvement_abs_str} (positive = staged better)
        </div>
        <h3>Progression Comparison</h3>
        <img src="data:image/png;base64,{comparison_plot_b64}" />
        <h3>Per-Stage Comparison</h3>
        <table>
            <tr>
                <th>Stage</th>
                <th>Staged Loss</th>
                <th>Baseline Loss</th>
                <th>Improvement</th>
                <th>Ratio</th>
                <th>Staged Samples</th>
                <th>Baseline Samples</th>
                <th>Winner</th>
            </tr>
            {comparison_rows}
        </table>
    </section>
"""

        # Sample counts section
        cumulative_sample_counts: dict[int, int] = {}
        for stage_num, result, orig_loss, _ in stage_original_evals:
            stage_counts = result.cumulative_metrics.get("sample_seen_counts", {})
            for frame_idx, count in stage_counts.items():
                cumulative_sample_counts[frame_idx] = cumulative_sample_counts.get(frame_idx, 0) + count

        cumulative_counts_b64 = ""
        if cumulative_sample_counts:
            fig_cumulative = visualization.create_sample_counts_plot(
                cumulative_sample_counts, title="Cumulative Sample Counts (All Stages)"
            )
            if fig_cumulative:
                cumulative_counts_b64 = fig_to_base64(fig_cumulative)
                plt.close(fig_cumulative)

        stage_counts_graphs_html = ""
        for stage_num, result, orig_loss, _ in stage_original_evals:
            stage_counts = result.cumulative_metrics.get("sample_seen_counts", {})
            if stage_counts:
                is_best = stage_num == best_stage
                best_label = " (Best)" if is_best else ""
                fig_stage_counts = visualization.create_sample_counts_plot(
                    stage_counts, title=f"Stage {stage_num}{best_label} Sample Counts"
                )
                if fig_stage_counts:
                    counts_b64 = fig_to_base64(fig_stage_counts)
                    plt.close(fig_stage_counts)
                    total_samples = sum(stage_counts.values())
                    stage_counts_graphs_html += f"""
            <div class="stage-loss-graph">
                <h4>Stage {stage_num}{best_label} - Total Samples: {total_samples:,}</h4>
                <img src="data:image/png;base64,{counts_b64}" />
            </div>"""

        sample_counts_html = f"""
    <section id="sample-counts">
        <h2>Sample Counts</h2>
        <h3>Cumulative Across All Stages</h3>
        {'<img src="data:image/png;base64,' + cumulative_counts_b64 + '" />' if cumulative_counts_b64 else "<p>No cumulative sample count data available</p>"}
        <h3>Per Stage</h3>
        <div class="stage-loss-graphs">
            {stage_counts_graphs_html if stage_counts_graphs_html else "<p>No per-stage sample count graphs available</p>"}
        </div>
    </section>
"""

        # Inference section
        print("Generating inference for best checkpoint...")
        inference_dir = output_dir / "best_inference"
        inference_dir.mkdir(exist_ok=True)

        selected_frame = cfg.selected_frame_offset
        selected_images = generate_counterfactual_images(
            original_session, selected_frame, inference_dir, "best_selected"
        )

        min_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
        valid_indices = list(range(min_frames - 1, len(original_session["observations"])))
        random_indices = random.sample(valid_indices, min(cfg.num_random_obs_to_visualize, len(valid_indices)))

        random_images = {}
        for idx in random_indices:
            imgs = generate_counterfactual_images(original_session, idx, inference_dir, f"best_obs_{idx}")
            random_images[idx] = imgs

        selected_html = ""
        for action, b64 in selected_images:
            if b64:
                selected_html += f'<div class="inference-item"><h4>Action {action}</h4><img src="data:image/png;base64,{b64}" /></div>'

        random_html = ""
        for idx, imgs in random_images.items():
            random_html += f'<div class="obs-group"><h4>Observation {idx}</h4><div class="inference-row">'
            for action, b64 in imgs:
                if b64:
                    random_html += f'<div class="inference-item"><span>Action {action}</span><img src="data:image/png;base64,{b64}" /></div>'
            random_html += '</div></div>'

        inference_html = f"""
    <section id="best-inference">
        <h2>Best Checkpoint Inference</h2>
        <h3>Selected Frame {selected_frame}</h3>
        <div class="inference-row">
            {selected_html if selected_html else "<p>No inference images available</p>"}
        </div>
        <h3>Random Observations</h3>
        {random_html if random_html else "<p>No random observation images available</p>"}
    </section>
"""

        # Stage report links
        stage_links = ""
        for stage_num, result, orig_loss, _ in stage_original_evals:
            stage_dir = f"stage{stage_num}_run{result.run_num}"
            is_best = stage_num == best_stage
            best_marker = " ⭐" if is_best else ""
            stage_links += f'<li><a href="{stage_dir}/report.html">Stage {stage_num} Run {result.run_num} (Staged)</a> (orig loss: {format_loss_safe(orig_loss)}){best_marker}</li>'

        if baseline_original_evals:
            stage_links += '<li style="margin-top: 10px;"><strong>Baseline Reports:</strong></li>'
            for stage_num, result, orig_loss, _ in baseline_original_evals:
                stage_dir = f"stage{stage_num}_baseline_run{result.run_num}"
                stage_links += f'<li style="margin-left: 20px;"><a href="{stage_dir}/report.html">Stage {stage_num} Run {result.run_num} (Baseline)</a> (orig loss: {format_loss_safe(orig_loss)})</li>'

        stage_links_html = f"""
    <section id="stage-links">
        <h2>Individual Stage Reports</h2>
        <ul>
            {stage_links}
        </ul>
    </section>
"""

        # Save summary JSON
        summary = {
            "total_time_seconds": total_elapsed,
            "staged": {
                "best_checkpoint": best_run_result.best_checkpoint_path,
                "best_stage": best_stage,
                "best_loss_original": best_orig_loss,
                "stages": [
                    {
                        "stage": stage_num,
                        "original_loss": orig_loss,
                        "train_loss": result.best_loss,
                        "time": result.elapsed_time,
                        "samples": result.total_samples_trained,
                        "stop_reason": result.stop_reason,
                        "checkpoint": result.best_checkpoint_path,
                    }
                    for stage_num, result, orig_loss, _ in stage_original_evals
                ],
            },
        }

        if baseline_original_evals:
            best_baseline_eval = min(baseline_original_evals, key=lambda x: x[2])
            best_baseline_stage, best_baseline_result, best_baseline_orig_loss, _ = best_baseline_eval

            summary["baseline"] = {
                "best_checkpoint": best_baseline_result.best_checkpoint_path,
                "best_stage": best_baseline_stage,
                "best_loss_original": best_baseline_orig_loss,
                "stages": [
                    {
                        "stage": stage_num,
                        "original_loss": orig_loss,
                        "train_loss": result.best_loss,
                        "time": result.elapsed_time,
                        "samples": result.total_samples_trained,
                        "stop_reason": result.stop_reason,
                        "checkpoint": result.best_checkpoint_path,
                    }
                    for stage_num, result, orig_loss, _ in baseline_original_evals
                ],
            }

            stages_won_staged = sum(1 for c in comparisons if c.improvement_absolute > 0)
            stages_won_baseline = sum(1 for c in comparisons if c.improvement_absolute < 0)
            avg_improvement_ratio = sum(c.improvement_ratio for c in comparisons) / len(comparisons)

            if stages_won_staged > stages_won_baseline:
                winner = "staged"
            elif stages_won_baseline > stages_won_staged:
                winner = "baseline"
            else:
                winner = "tie"

            summary["comparison"] = {
                "winner": winner,
                "stages_won_staged": stages_won_staged,
                "stages_won_baseline": stages_won_baseline,
                "improvement_ratio_mean": avg_improvement_ratio,
                "per_stage": [
                    {
                        "stage": c.stage_num,
                        "staged_loss": c.staged_orig_loss,
                        "baseline_loss": c.baseline_orig_loss,
                        "improvement_absolute": c.improvement_absolute,
                        "improvement_ratio": c.improvement_ratio,
                        "staged_samples": c.staged_samples_trained,
                        "baseline_samples": c.baseline_samples_trained,
                        "winner": "staged" if c.improvement_absolute > 0 else ("baseline" if c.improvement_absolute < 0 else "tie"),
                    }
                    for c in comparisons
                ],
            }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Assemble final-only sections
        best_checkpoint_html = best_checkpoint_html
        stage_progression_html = lr_schedule_html + stage_progression_html

    # Generate config section (always available)
    config_html = generate_config_html(cfg)

    # Generate full training loss plots (always available, grows with each stage)
    full_loss_html = ""
    fig_full, fig_zoomed = create_full_training_loss_plot(completed_stages)
    if fig_full:
        full_b64 = fig_to_base64(fig_full)
        zoomed_b64 = fig_to_base64(fig_zoomed) if fig_zoomed else ""
        full_loss_html = f"""
    <section id="full-training-loss">
        <h2>Loss Across Full Training Run</h2>
        <img src="data:image/png;base64,{full_b64}" />
        {'<h3>Loss Detail (Post Initial Drop)</h3><img src="data:image/png;base64,' + zoomed_b64 + '" />' if zoomed_b64 else ""}
    </section>
"""

    # Generate multi-run statistics (always available)
    multi_run_html = generate_multi_run_stats_html(
        all_stage_runs or {}, completed_stages
    )

    # Build HTML
    status_indicator = "✓ Complete" if is_final else "⏳ In Progress"
    status_color = "#4CAF50" if is_final else "#FF9800"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Staged Training Report - {run_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3, h4 {{ color: #666; }}
        .metric {{ background: #e8f5e9; padding: 20px; border-radius: 4px; margin: 10px 0; font-size: 1.1em; }}
        .metric strong {{ color: #2E7D32; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .inference-row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .inference-item {{ flex: 1; min-width: 200px; text-align: center; }}
        .inference-item img {{ max-width: 300px; }}
        .obs-group {{ margin: 20px 0; padding: 15px; background: #fafafa; border-radius: 4px; }}
        .stage-loss-graph {{ margin: 20px 0; padding: 15px; background: #fafafa; border-radius: 4px; border-left: 4px solid #4CAF50; }}
        .stage-loss-graph h4 {{ margin-top: 0; color: #333; }}
        .stage-loss-graphs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        ul {{ line-height: 2; }}
        a {{ color: #1976D2; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; color: white; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Staged Training Report <span class="status-badge" style="background: {status_color};">{status_indicator}</span></h1>

    <section id="summary">
        <div class="metric">
            <strong>Run ID:</strong> {run_id}<br>
            <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            <strong>Stages Completed:</strong> {len(completed_stages)}<br>
            <strong>Total Elapsed Time:</strong> {format_duration(total_elapsed)}
        </div>
    </section>

    {config_html}

    <section id="timing">
        <h2>Timing Summary</h2>
        <table>
            <tr>
                <th>Stage</th>
                <th>Plateau Sweeps</th>
                <th>Sweep Time</th>
                <th>Training Time</th>
                <th>Stage Total</th>
            </tr>
            {timing_rows}
        </table>
        {initial_sweep_note}
    </section>

    {f'<section id="plateau-sweeps"><h2>Plateau Sweep Details</h2>{lr_details_html}</section>' if total_plateau_sweeps > 0 else ""}

    <section id="stage-results">
        <h2>Stage Results</h2>
        <table>
            <tr>
                <th>Stage</th>
                <th>Best Loss</th>
                <th>Stop Reason</th>
                <th>Samples Trained</th>
                <th>Time</th>
                <th>Sweeps</th>
                <th>LR (Initial→Final)</th>
            </tr>
            {stage_results_rows}
        </table>
        <p><strong>Total Plateau Sweeps:</strong> {total_plateau_sweeps}</p>
    </section>

    <section id="stop-reasons">
        <h2>Stop Reason Breakdown</h2>
        {stop_reason_html}
    </section>

    {full_loss_html}
    {multi_run_html}

    {best_checkpoint_html}
    {stage_progression_html}
    {comparison_html}
    {sample_counts_html}
    {inference_html}
    {stage_links_html}
</div>
</body>
</html>"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Copy to docs folder for easy access (only for final reports)
    if is_final:
        docs_copy_path = docs_dir / docs_filename
        shutil.copy(report_path, docs_copy_path)
        print(f"[REPORT] Report updated: {report_path}")
        print(f"[REPORT] Copied to: {docs_copy_path}")
    else:
        print(f"[REPORT] Report updated: {report_path}")

    return str(report_path)


def cleanup_stage_checkpoints(
    stage_results: list[StageResult],
    best_result: StageResult,
) -> None:
    """Delete checkpoints from non-best runs of a stage."""
    for result in stage_results:
        if result.best_checkpoint_path != best_result.best_checkpoint_path:
            if os.path.exists(result.best_checkpoint_path):
                os.remove(result.best_checkpoint_path)
                print(f"  Deleted: {Path(result.best_checkpoint_path).name}")


def run_staged_training(
    root_session_path: str,
    cfg: StagedTrainingConfig,
    output_dir: str,
    run_id: str,
    single_stage: Optional[int] = None,
    direct_train_session: Optional[str] = None,
    direct_val_session: Optional[str] = None,
    original_session_path: Optional[str] = None,
    starting_checkpoint: Optional[str] = None,
) -> None:
    """
    Main function to run staged training.

    Args:
        root_session_path: Path to root session with staged splits
        cfg: Training configuration
        output_dir: Output directory for reports
        run_id: Unique identifier for this run (used in checkpoint names to avoid collisions)
        single_stage: If specified, run only this stage number
        direct_train_session: Path to training session (direct mode)
        direct_val_session: Path to validation session (direct mode)
        original_session_path: Path to original session for evaluation (optional)
        starting_checkpoint: Path to starting checkpoint (optional)
    """
    # Set instance_id for checkpoint naming (prevents collisions with concurrent runs)
    state.instance_id = run_id
    print(f"Run ID: {run_id}")
    if cfg.seed is not None:
        print(f"Reproducibility seed: {cfg.seed}")

    # Track overall training time
    overall_start_time = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save run_id to config for reference/reproducibility
    cfg.run_id = run_id

    # Save config
    cfg.to_yaml(str(output_path / "config.yaml"))

    # Discover stages based on mode
    if direct_train_session and direct_val_session:
        # Direct mode: create synthetic stage entry
        print("Direct mode: using provided train/val sessions")
        stages = [(1, direct_train_session, direct_val_session)]

        # Determine original session for evaluation
        if original_session_path:
            effective_original_path = original_session_path
            print(f"Original session for evaluation: {Path(original_session_path).name}")
        else:
            # Default to train session (report will note this)
            effective_original_path = direct_train_session
            print("Note: Using training session as original for evaluation. "
                  "Use --original-session to specify a different session.")
    else:
        # Discover stages from root session
        print("Discovering staged splits...")
        effective_original_path = root_session_path
        stages = discover_staged_splits(root_session_path)

        if not stages:
            raise ValueError(f"No staged splits found for {root_session_path}")

        if single_stage is not None:
            # Filter to only the requested stage
            stages = [(n, t, v) for n, t, v in stages if n == single_stage]
            if not stages:
                raise ValueError(f"Stage {single_stage} not found in staged splits")
            print(f"Single stage mode: running stage {single_stage} only")

    print(f"Found {len(stages)} stage(s):")
    for stage_num, train_path, val_path in stages:
        print(f"  Stage {stage_num}: train={Path(train_path).name}, val={Path(val_path).name}")

    # Load original session for final evaluation
    print("\nLoading original session for evaluation...")
    original_session = load_session_for_training(effective_original_path)

    # Clean old checkpoints if enabled (only cleans checkpoints from THIS run_id to avoid
    # interfering with concurrent runs using different run_ids)
    if cfg.clean_old_checkpoints:
        root_session_name = Path(root_session_path).name
        # Get checkpoint directory based on session type
        checkpoint_dir = state.get_checkpoint_dir_for_session(root_session_path)
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.exists():
            # Only clean checkpoints matching THIS run's pattern (includes run_id)
            # Pattern: best_model_auto_{session_name}_{run_id}_*
            old_checkpoints = list(checkpoint_dir.glob(f"best_model_auto_{root_session_name}_{run_id}_*"))
            # Also clean up stage-specific checkpoints for this run
            for stage_num, train_path, _ in stages:
                train_session_name = Path(train_path).name
                old_checkpoints.extend(checkpoint_dir.glob(f"best_model_auto_{train_session_name}_{run_id}_*"))

            if old_checkpoints:
                print(f"\nCleaning {len(old_checkpoints)} old checkpoints from this run_id...")
                for cp in old_checkpoints:
                    try:
                        cp.unlink()
                        print(f"  Deleted: {cp.name}")
                    except Exception as e:
                        print(f"  Failed to delete {cp.name}: {e}")
            else:
                print("\nNo old checkpoints to clean up for this run_id.")

    # Run STAGED training (weights carry over between stages)
    staged_results = []
    current_checkpoint = starting_checkpoint  # Use provided checkpoint or None for fresh weights
    all_sweep_results = []  # Track LR sweep results for reporting
    all_timings = []  # Track timing for each stage
    completed_stages = []  # For incremental final reports
    all_stage_runs: dict[int, list[StageResult]] = {}  # Track ALL runs per stage for multi-run stats

    # Store state for interrupt/crash recovery (module-level so main() can access it)
    global _partial_report_state
    _partial_report_state = {
        'completed_stages': completed_stages,
        'all_sweep_results': all_sweep_results,
        'all_timings': all_timings,
        'output_dir': output_path,
        'run_id': run_id,
        'overall_start_time': overall_start_time,
        'original_session': original_session,
        'cfg': cfg,
        'all_stage_runs': all_stage_runs,
        'current_stage_info': None,  # Updated per-stage for interrupted stage recovery
    }

    for stage_num, train_path, val_path in stages:
        stage_start_time = time.time()

        # Track in-progress stage for interrupt recovery
        _partial_report_state['current_stage_info'] = {
            'stage_num': stage_num,
            'start_time': stage_start_time,
            'train_path': train_path,
        }

        print(f"\n{'#'*60}")
        print(f"# STAGE {stage_num}")
        print(f"{'#'*60}")

        # Load sessions
        print(f"Loading training session: {Path(train_path).name}")
        train_session = load_session_for_training(train_path)

        print(f"Loading validation session: {Path(val_path).name}")
        val_session = load_session_for_training(val_path)

        # Upfront LR Sweep (when plateau sweep is disabled, or initial_sweep_enabled is True)
        selected_lr = cfg.custom_lr
        sweep_result = None
        sweep_elapsed = 0.0

        if not cfg.plateau_sweep.enabled or cfg.initial_sweep_enabled:
            sweep_start = time.time()
            print(f"\n--- LR Sweep for Stage {stage_num} ---")

            # Allocate time budget for sweep if stage has total budget
            sweep_budget = 0  # Unlimited by default
            if cfg.stage_time_budget_min > 0:
                # Allocate up to 70% of stage time for sweep
                sweep_budget = cfg.stage_time_budget_min * 0.7

            sweep_result = run_lr_sweep_for_stage(
                stage_num=stage_num,
                train_session_path=train_path,
                val_session_path=val_path,
                starting_checkpoint=current_checkpoint,
                cfg_dict=cfg.to_dict(),
                output_dir=output_path / f"stage{stage_num}_lr_sweep",
                run_id=run_id,
                lr_min=cfg.lr_sweep.lr_min,
                lr_max=cfg.lr_sweep.lr_max,
                phase_a_num_candidates=cfg.lr_sweep.phase_a_num_candidates,
                phase_a_seeds=cfg.lr_sweep.phase_a_seeds,
                phase_a_time_budget_min=cfg.lr_sweep.phase_a_time_budget_min,
                phase_a_survivor_count=cfg.lr_sweep.phase_a_survivor_count,
                phase_b_seeds=cfg.lr_sweep.phase_b_seeds,
                phase_b_time_budget_min=cfg.lr_sweep.phase_b_time_budget_min,
                ranking_metric=cfg.lr_sweep.ranking_metric,
                save_state=cfg.lr_sweep.save_sweep_state,
                time_budget_min=sweep_budget,
            )

            selected_lr = sweep_result.selected_lr
            sweep_elapsed = time.time() - sweep_start
            all_sweep_results.append(sweep_result)
            print(f"LR Sweep selected: {selected_lr:.2e} (took {format_duration(sweep_elapsed)})")

            # Force GPU memory cleanup after sweep
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Override LR for main training
        stage_cfg = StagedTrainingConfig.from_dict(cfg.to_dict())
        stage_cfg.custom_lr = selected_lr
        stage_cfg.disable_lr_scaling = True

        # Calculate remaining time budget for main training
        main_training_budget = 0
        if cfg.stage_time_budget_min > 0:
            remaining = cfg.stage_time_budget_min - (sweep_elapsed / 60)
            main_training_budget = max(1.0, remaining)  # At least 1 minute
            print(f"Main training time budget: {main_training_budget:.1f} min")

        stage_results = []
        main_training_start = time.time()

        per_run_budget = main_training_budget / cfg.runs_per_stage if main_training_budget > 0 else 0

        if cfg.runs_per_stage > 1 and not cfg.serial_runs:
            # Parallel execution for multiple runs
            stage_results = run_main_training_parallel(
                stage_num=stage_num,
                num_runs=cfg.runs_per_stage,
                train_session_path=train_path,
                val_session_path=val_path,
                starting_checkpoint=current_checkpoint,
                selected_lr=selected_lr,
                cfg_dict=stage_cfg.to_dict(),
                output_dir=output_path,
                run_id=run_id,
                is_baseline=False,
                time_budget_min=per_run_budget,
            )
        elif cfg.runs_per_stage > 1 and cfg.serial_runs:
            # Serial execution for multiple runs
            print(f"\n[TRAINING] Running {cfg.runs_per_stage} runs serially for stage {stage_num}")
            stage_results = []
            for run_num in range(1, cfg.runs_per_stage + 1):
                print(f"\n[TRAINING] Starting serial run {run_num}/{cfg.runs_per_stage}")
                if cfg.seed is not None:
                    seed = derive_seed(cfg.seed, stage_num, run_num, False)
                else:
                    seed = hash((run_id, stage_num, run_num, False)) % (2**32)
                set_all_seeds(seed, deterministic_cudnn=(cfg.seed is not None))

                # Reload sessions for each run to ensure clean state
                run_train_session = load_session_for_training(train_path)
                run_val_session = load_session_for_training(val_path)

                run_output_dir = output_path / f"stage{stage_num}_run{run_num}"
                result = run_stage_training(
                    stage_num=stage_num,
                    run_num=run_num,
                    train_session=run_train_session,
                    val_session=run_val_session,
                    checkpoint_path=current_checkpoint,
                    cfg=stage_cfg,
                    output_dir=run_output_dir,
                    is_baseline=False,
                    run_id=run_id,
                    time_budget_min=per_run_budget,
                )
                stage_results.append(result)
                print(f"[TRAINING] Serial run {run_num} complete: loss={result.best_loss:.6f}")
        else:
            # Single run - no parallelization overhead
            if cfg.seed is not None:
                seed = derive_seed(cfg.seed, stage_num, 1, False)
                set_all_seeds(seed, deterministic_cudnn=True)
            run_output_dir = output_path / f"stage{stage_num}_run1"
            result = run_stage_training(
                stage_num=stage_num,
                run_num=1,
                train_session=train_session,
                val_session=val_session,
                checkpoint_path=current_checkpoint,
                cfg=stage_cfg,
                output_dir=run_output_dir,
                is_baseline=False,
                run_id=run_id,
                time_budget_min=main_training_budget if main_training_budget > 0 else 0,
            )
            stage_results = [result]

        # Generate reports AFTER all runs complete
        for result in stage_results:
            run_output_dir = output_path / f"stage{stage_num}_run{result.run_num}"
            generate_stage_report(
                result=result,
                train_session=train_session,
                val_session=val_session,
                original_session=original_session,
                cfg=stage_cfg,
                output_dir=run_output_dir,
            )

        main_training_elapsed = time.time() - main_training_start

        # Select best checkpoint from all runs (filter out failed runs with no checkpoint)
        valid_results = [r for r in stage_results if r.best_checkpoint_path and "nan" not in r.stop_reason]
        if not valid_results:
            print(f"\nERROR: All runs failed for stage {stage_num}. Cannot continue.")
            print("Failed runs:")
            for r in stage_results:
                print(f"  Run {r.run_num}: {r.stop_reason}")
            raise RuntimeError(f"All runs failed for stage {stage_num}")

        best_run = min(valid_results, key=lambda r: r.best_loss)
        print(f"\nBest run for stage {stage_num}: Run {best_run.run_num} (loss: {best_run.best_loss:.6f})")

        # Update current checkpoint for next stage
        current_checkpoint = best_run.best_checkpoint_path

        # Cleanup non-best checkpoints
        if cfg.runs_per_stage > 1:
            print("Cleaning up non-best checkpoints...")
            cleanup_stage_checkpoints(stage_results, best_run)

        staged_results.append((stage_num, best_run))
        all_stage_runs[stage_num] = stage_results

        # Track timing
        stage_total_time = time.time() - stage_start_time
        timing = StageTiming(
            stage_num=stage_num,
            lr_sweep_total_sec=sweep_elapsed,
            lr_sweep_phase_a_sec=sweep_result.phase_a.total_wall_time_sec if sweep_result else 0,
            lr_sweep_phase_b_sec=sweep_result.phase_b.total_wall_time_sec if sweep_result else 0,
            lr_sweep_trial_count=(
                len(sweep_result.phase_a.lr_results) * cfg.lr_sweep.phase_a_seeds +
                len(sweep_result.phase_b.lr_results) * cfg.lr_sweep.phase_b_seeds
            ) if sweep_result else 0,
            main_training_sec=main_training_elapsed,
            main_training_samples=best_run.total_samples_trained,
            total_stage_sec=stage_total_time,
        )
        all_timings.append(timing)
        completed_stages.append((stage_num, best_run, None))  # baseline added later
        _partial_report_state['current_stage_info'] = None  # Stage completed

        # Generate progressive final report (updated after each stage)
        generate_final_report(
            completed_stages=completed_stages,
            all_sweep_results=all_sweep_results,
            all_timings=all_timings,
            output_dir=output_path,
            run_id=run_id,
            overall_start_time=overall_start_time,
            original_session=None,  # Don't evaluate until final
            cfg=cfg,
            is_final=False,
            all_stage_runs=all_stage_runs,
        )

    # Run BASELINE training if enabled (fresh weights each stage)
    baseline_results = []
    baseline_sweep_results = []  # Track LR sweep results for baseline
    baseline_timings = []

    if cfg.enable_baseline:
        print("\n" + "#" * 60)
        print("# BASELINE TRAINING (Fresh weights each stage)")
        print("#" * 60)

        for stage_num, train_path, val_path in stages:
            baseline_stage_start = time.time()

            print(f"\n{'#'*60}")
            print(f"# BASELINE STAGE {stage_num}")
            print(f"{'#'*60}")

            # Load sessions (reload to ensure clean state)
            print(f"Loading training session: {Path(train_path).name}")
            train_session = load_session_for_training(train_path)

            print(f"Loading validation session: {Path(val_path).name}")
            val_session = load_session_for_training(val_path)

            # Baseline Upfront LR Sweep (when plateau sweep is disabled) - separate sweep with fresh weights
            baseline_selected_lr = cfg.custom_lr
            baseline_sweep_result = None
            baseline_sweep_elapsed = 0.0

            if not cfg.plateau_sweep.enabled:
                baseline_sweep_start = time.time()
                print(f"\n--- Baseline LR Sweep for Stage {stage_num} ---")

                # Allocate time budget for sweep if stage has total budget
                sweep_budget = 0  # Unlimited by default
                if cfg.stage_time_budget_min > 0:
                    sweep_budget = cfg.stage_time_budget_min * 0.7

                baseline_sweep_result = run_lr_sweep_for_stage(
                    stage_num=stage_num,
                    train_session_path=train_path,
                    val_session_path=val_path,
                    starting_checkpoint=None,  # Fresh weights for baseline
                    cfg_dict=cfg.to_dict(),
                    output_dir=output_path / f"stage{stage_num}_baseline_lr_sweep",
                    run_id=run_id,
                    lr_min=cfg.lr_sweep.lr_min,
                    lr_max=cfg.lr_sweep.lr_max,
                    phase_a_num_candidates=cfg.lr_sweep.phase_a_num_candidates,
                    phase_a_seeds=cfg.lr_sweep.phase_a_seeds,
                    phase_a_time_budget_min=cfg.lr_sweep.phase_a_time_budget_min,
                    phase_a_survivor_count=cfg.lr_sweep.phase_a_survivor_count,
                    phase_b_seeds=cfg.lr_sweep.phase_b_seeds,
                    phase_b_time_budget_min=cfg.lr_sweep.phase_b_time_budget_min,
                    ranking_metric=cfg.lr_sweep.ranking_metric,
                    save_state=cfg.lr_sweep.save_sweep_state,
                    time_budget_min=sweep_budget,
                )

                baseline_selected_lr = baseline_sweep_result.selected_lr
                baseline_sweep_elapsed = time.time() - baseline_sweep_start
                baseline_sweep_results.append(baseline_sweep_result)
                print(f"Baseline LR Sweep selected: {baseline_selected_lr:.2e} (took {format_duration(baseline_sweep_elapsed)})")

                # Force GPU memory cleanup after sweep
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Override LR for baseline training
            baseline_cfg = StagedTrainingConfig.from_dict(cfg.to_dict())
            baseline_cfg.custom_lr = baseline_selected_lr
            baseline_cfg.disable_lr_scaling = True

            # Calculate remaining time budget for main training
            baseline_main_budget = 0
            if cfg.stage_time_budget_min > 0:
                remaining = cfg.stage_time_budget_min - (baseline_sweep_elapsed / 60)
                baseline_main_budget = max(1.0, remaining)

            baseline_stage_results = []
            baseline_main_start = time.time()

            baseline_per_run_budget = baseline_main_budget / cfg.baseline_runs_per_stage if baseline_main_budget > 0 else 0

            if cfg.baseline_runs_per_stage > 1 and not cfg.serial_runs:
                # Parallel execution for multiple baseline runs
                baseline_stage_results = run_main_training_parallel(
                    stage_num=stage_num,
                    num_runs=cfg.baseline_runs_per_stage,
                    train_session_path=train_path,
                    val_session_path=val_path,
                    starting_checkpoint=None,  # ALWAYS fresh weights for baseline
                    selected_lr=baseline_selected_lr,
                    cfg_dict=baseline_cfg.to_dict(),
                    output_dir=output_path,
                    run_id=run_id,
                    is_baseline=True,
                    time_budget_min=baseline_per_run_budget,
                )
            elif cfg.baseline_runs_per_stage > 1 and cfg.serial_runs:
                # Serial execution for multiple baseline runs
                print(f"\n[BASELINE] Running {cfg.baseline_runs_per_stage} baseline runs serially for stage {stage_num}")
                baseline_stage_results = []
                for run_num in range(1, cfg.baseline_runs_per_stage + 1):
                    print(f"\n[BASELINE] Starting serial baseline run {run_num}/{cfg.baseline_runs_per_stage}")
                    if cfg.seed is not None:
                        seed = derive_seed(cfg.seed, stage_num, run_num, True)
                    else:
                        seed = hash((run_id, stage_num, run_num, True)) % (2**32)
                    set_all_seeds(seed, deterministic_cudnn=(cfg.seed is not None))

                    run_train_session = load_session_for_training(train_path)
                    run_val_session = load_session_for_training(val_path)

                    run_output_dir = output_path / f"stage{stage_num}_baseline_run{run_num}"
                    result = run_stage_training(
                        stage_num=stage_num,
                        run_num=run_num,
                        train_session=run_train_session,
                        val_session=run_val_session,
                        checkpoint_path=None,  # ALWAYS fresh weights for baseline
                        cfg=baseline_cfg,
                        output_dir=run_output_dir,
                        is_baseline=True,
                        run_id=run_id,
                        time_budget_min=baseline_per_run_budget,
                    )
                    baseline_stage_results.append(result)
                    print(f"[BASELINE] Serial baseline run {run_num} complete: loss={result.best_loss:.6f}")
            else:
                # Single run - no parallelization overhead
                if cfg.seed is not None:
                    seed = derive_seed(cfg.seed, stage_num, 1, True)
                    set_all_seeds(seed, deterministic_cudnn=True)
                run_output_dir = output_path / f"stage{stage_num}_baseline_run1"
                result = run_stage_training(
                    stage_num=stage_num,
                    run_num=1,
                    train_session=train_session,
                    val_session=val_session,
                    checkpoint_path=None,  # ALWAYS fresh weights for baseline
                    cfg=baseline_cfg,
                    output_dir=run_output_dir,
                    is_baseline=True,
                    run_id=run_id,
                    time_budget_min=baseline_main_budget if baseline_main_budget > 0 else 0,
                )
                baseline_stage_results = [result]

            # Generate reports AFTER all baseline runs complete
            for result in baseline_stage_results:
                run_output_dir = output_path / f"stage{stage_num}_baseline_run{result.run_num}"
                generate_stage_report(
                    result=result,
                    train_session=train_session,
                    val_session=val_session,
                    original_session=original_session,
                    cfg=baseline_cfg,
                    output_dir=run_output_dir,
                )

            baseline_main_elapsed = time.time() - baseline_main_start

            # Select best baseline run for this stage (filter out failed runs with no checkpoint)
            valid_baseline_results = [r for r in baseline_stage_results if r.best_checkpoint_path]
            if not valid_baseline_results:
                print(f"\nWARNING: All baseline runs failed for stage {stage_num}.")
                print("Failed baseline runs:")
                for r in baseline_stage_results:
                    print(f"  Run {r.run_num}: {r.stop_reason}")
                # Use a dummy result with inf loss for comparison
                best_baseline_run = baseline_stage_results[0] if baseline_stage_results else None
            else:
                best_baseline_run = min(valid_baseline_results, key=lambda r: r.best_loss)
                print(f"\nBest baseline run for stage {stage_num}: Run {best_baseline_run.run_num} (loss: {best_baseline_run.best_loss:.6f})")

            # Cleanup non-best baseline checkpoints
            if cfg.baseline_runs_per_stage > 1:
                print("Cleaning up non-best baseline checkpoints...")
                cleanup_stage_checkpoints(baseline_stage_results, best_baseline_run)

            baseline_results.append((stage_num, best_baseline_run))

            # Track baseline timing
            baseline_stage_total = time.time() - baseline_stage_start
            baseline_timing = StageTiming(
                stage_num=stage_num,
                lr_sweep_total_sec=baseline_sweep_elapsed,
                lr_sweep_phase_a_sec=baseline_sweep_result.phase_a.total_wall_time_sec if baseline_sweep_result else 0,
                lr_sweep_phase_b_sec=baseline_sweep_result.phase_b.total_wall_time_sec if baseline_sweep_result else 0,
                lr_sweep_trial_count=(
                    len(baseline_sweep_result.phase_a.lr_results) * cfg.lr_sweep.phase_a_seeds +
                    len(baseline_sweep_result.phase_b.lr_results) * cfg.lr_sweep.phase_b_seeds
                ) if baseline_sweep_result else 0,
                main_training_sec=baseline_main_elapsed,
                main_training_samples=best_baseline_run.total_samples_trained,
                total_stage_sec=baseline_stage_total,
            )
            baseline_timings.append(baseline_timing)

            # Update completed_stages with baseline result
            for i, (s_num, staged, _) in enumerate(completed_stages):
                if s_num == stage_num:
                    completed_stages[i] = (s_num, staged, best_baseline_run)
                    break

    # Generate final summary with evaluation and inference
    final_report_path = generate_final_report(
        completed_stages=completed_stages,
        all_sweep_results=all_sweep_results,
        all_timings=all_timings,
        output_dir=output_path,
        run_id=run_id,
        overall_start_time=overall_start_time,
        original_session=original_session,
        cfg=cfg,
        is_final=True,
        all_stage_runs=all_stage_runs,
    )

    print("\n" + "=" * 60)
    print("STAGED TRAINING COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"Final report: {final_report_path}")

    # Clear partial report state on normal completion
    _partial_report_state = None


def regenerate_report_from_artifacts(output_dir: str, original_session_path: str):
    """Regenerate final report from saved training artifacts.

    Reconstructs StageResult/StageTiming from saved metrics.json and summary.json,
    loads the original session and best checkpoint, and generates the final HTML report.
    """
    output_path = Path(output_dir)

    # Load config
    config_path = output_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {output_dir}")
    cfg = StagedTrainingConfig.from_yaml(str(config_path))
    run_id = cfg.run_id

    # Load summary
    summary_path = output_path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json found in {output_dir}")
    with open(summary_path) as f:
        summary = json.load(f)

    print(f"Regenerating report for run_id: {run_id}")
    print(f"Loading original session: {original_session_path}")
    original_session = load_session_for_training(original_session_path)

    # Reconstruct completed_stages from saved metrics
    completed_stages = []
    all_timings = []
    all_stage_runs = {}

    for stage_info in summary["staged"]["stages"]:
        stage_num = stage_info["stage"]
        run_results = []

        for run_dir in sorted(output_path.glob(f"stage{stage_num}_run*")):
            if "baseline" in run_dir.name:
                continue
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            with open(metrics_file) as f:
                metrics = json.load(f)

            result = StageResult(
                stage_num=stage_num,
                run_num=metrics["run_num"],
                is_baseline=False,
                best_checkpoint_path=metrics["best_checkpoint"],
                best_loss=metrics["best_loss"],
                stop_reason=metrics["stop_reason"],
                elapsed_time=metrics["elapsed_time_seconds"],
                total_samples_trained=metrics["total_samples_trained"],
                cumulative_metrics={},
                final_train_loss=metrics.get("train_eval_stats", {}).get("hybrid", {}).get("mean", 0),
                final_val_loss=metrics.get("val_eval_stats", {}).get("hybrid", {}).get("mean"),
            )
            run_results.append(result)

        if run_results:
            best_run = min(run_results, key=lambda r: r.best_loss)
            completed_stages.append((stage_num, best_run, None))
            all_stage_runs[stage_num] = run_results
            all_timings.append(StageTiming(
                stage_num=stage_num,
                lr_sweep_total_sec=0,
                lr_sweep_phase_a_sec=0,
                lr_sweep_phase_b_sec=0,
                lr_sweep_trial_count=0,
                main_training_sec=stage_info.get("time", 0),
                main_training_samples=stage_info.get("samples", 0),
                total_stage_sec=stage_info.get("time", 0),
            ))

    if not completed_stages:
        raise ValueError("No completed stages found in saved artifacts")

    print(f"Reconstructed {len(completed_stages)} stage(s)")
    total_time = summary.get("total_time_seconds", 0)

    report_path = generate_final_report(
        completed_stages=completed_stages,
        all_sweep_results=[],
        all_timings=all_timings,
        output_dir=output_path,
        run_id=run_id,
        overall_start_time=time.time() - total_time,
        original_session=original_session,
        cfg=cfg,
        is_final=True,
        all_stage_runs=all_stage_runs,
    )
    print(f"\nReport regenerated: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run staged training on progressively larger session splits"
    )
    parser.add_argument(
        "--root-session",
        help="Path to the root session directory (staged splits should exist)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for reports (default: saved/staged_training_reports/{session_name})",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config file (overrides defaults)",
    )
    parser.add_argument(
        "--runs-per-stage",
        type=int,
        default=None,
        help="Number of training runs per stage (default: from config)",
    )
    parser.add_argument(
        "--serial-runs",
        action="store_true",
        help="Run runs_per_stage serially instead of in parallel (lower memory usage)",
    )
    parser.add_argument(
        "--run-id",
        help="Unique run identifier for concurrent execution (default: auto-generated timestamp)",
    )
    # LR Sweep arguments
    parser.add_argument(
        "--disable-plateau-sweep",
        action="store_true",
        help="Disable plateau-triggered sweeps (default mode). Enables upfront LR sweep before each stage instead.",
    )
    parser.add_argument(
        "--disable-initial-sweep",
        action="store_true",
        help="Disable the initial LR sweep at the start of each stage (enabled by default).",
    )
    parser.add_argument(
        "--lr-sweep-lr-min",
        type=float,
        default=1e-6,
        help="Minimum LR for sweep search space (default: 1e-6)",
    )
    parser.add_argument(
        "--lr-sweep-lr-max",
        type=float,
        default=1e-2,
        help="Maximum LR for sweep search space (default: 1e-2)",
    )
    parser.add_argument(
        "--lr-sweep-phase-a-candidates",
        type=int,
        default=40,
        help="Number of LR candidates in Phase A (default: 40)",
    )
    parser.add_argument(
        "--lr-sweep-phase-a-budget-min",
        type=float,
        default=3.0,
        help="Time budget per Phase A trial in minutes (default: 3)",
    )
    parser.add_argument(
        "--lr-sweep-phase-a-survivors",
        type=int,
        default=5,
        help="Number of LRs to advance from Phase A to Phase B (default: 5)",
    )
    parser.add_argument(
        "--lr-sweep-phase-b-seeds",
        type=int,
        default=3,
        help="Seeds per LR in Phase B (default: 3)",
    )
    parser.add_argument(
        "--lr-sweep-phase-b-budget-min",
        type=float,
        default=10.0,
        help="Time budget per Phase B trial in minutes (default: 10)",
    )
    parser.add_argument(
        "--stage-time-budget-min",
        type=float,
        default=0,
        help="Total time budget per stage in minutes, including LR sweep and main training (0 = unlimited)",
    )
    parser.add_argument(
        "--disable-baseline",
        action="store_true",
        help="Disable baseline comparison runs",
    )
    # Single stage / direct session mode
    parser.add_argument(
        "--stage",
        type=int,
        help="Run only this specific stage number (requires staged splits to exist)",
    )
    parser.add_argument(
        "--train-session",
        help="Path to training session (use with --val-session for direct mode without staged splits)",
    )
    parser.add_argument(
        "--val-session",
        help="Path to validation session (use with --train-session for direct mode)",
    )
    parser.add_argument(
        "--original-session",
        help="Path to original (full) session for evaluation (optional in direct mode, defaults to train session)",
    )
    parser.add_argument(
        "--checkpoint",
        help="Starting checkpoint path (optional, for resuming training)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility (default: None = non-deterministic)",
    )
    parser.add_argument(
        "--regenerate-report",
        help="Regenerate final report from a completed run's output directory",
    )

    args = parser.parse_args()

    # Handle report regeneration mode (early return)
    if args.regenerate_report:
        original_path = args.original_session or args.root_session
        if not original_path:
            parser.error("--regenerate-report requires --root-session or --original-session")
        regenerate_report_from_artifacts(args.regenerate_report, original_path)
        return

    # Determine training mode and validate arguments
    if args.train_session and args.val_session:
        # Direct mode: use provided sessions as a single "stage"
        training_mode = "direct"
        if args.stage:
            print("Warning: --stage is ignored when using --train-session/--val-session")
        if not args.root_session:
            # In direct mode, root_session is not required but we need a value for output_dir
            args.root_session = args.train_session  # Use train session name
    elif args.stage is not None:
        # Single stage mode: run only the specified stage from staged splits
        training_mode = "single_stage"
        if not args.root_session:
            parser.error("--root-session is required when using --stage")
    else:
        # Full staged training mode (original behavior)
        training_mode = "full"
        if not args.root_session:
            parser.error("--root-session is required (or use --train-session + --val-session)")

    # Validate direct mode has both sessions
    if (args.train_session and not args.val_session) or (args.val_session and not args.train_session):
        parser.error("--train-session and --val-session must be used together")

    # Generate run_id if not provided (enables concurrent execution without conflicts)
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    # Load config
    if args.config:
        cfg = StagedTrainingConfig.from_yaml(args.config)
    else:
        cfg = StagedTrainingConfig()

    # Override runs_per_stage from CLI only if explicitly provided
    if args.runs_per_stage is not None:
        cfg.runs_per_stage = args.runs_per_stage
    if args.serial_runs:
        cfg.serial_runs = True

    # Override sweep mode from CLI
    if args.disable_plateau_sweep:
        cfg.plateau_sweep.enabled = False
    if args.disable_initial_sweep:
        cfg.initial_sweep_enabled = False
    if args.lr_sweep_lr_min != 1e-6:
        cfg.lr_sweep.lr_min = args.lr_sweep_lr_min
    if args.lr_sweep_lr_max != 1e-2:
        cfg.lr_sweep.lr_max = args.lr_sweep_lr_max
    if args.lr_sweep_phase_a_candidates != 40:
        cfg.lr_sweep.phase_a_num_candidates = args.lr_sweep_phase_a_candidates
    if args.lr_sweep_phase_a_budget_min != 3.0:
        cfg.lr_sweep.phase_a_time_budget_min = args.lr_sweep_phase_a_budget_min
    if args.lr_sweep_phase_a_survivors != 5:
        cfg.lr_sweep.phase_a_survivor_count = args.lr_sweep_phase_a_survivors
    if args.lr_sweep_phase_b_seeds != 3:
        cfg.lr_sweep.phase_b_seeds = args.lr_sweep_phase_b_seeds
    if args.lr_sweep_phase_b_budget_min != 10.0:
        cfg.lr_sweep.phase_b_time_budget_min = args.lr_sweep_phase_b_budget_min
    if args.stage_time_budget_min != 0:
        cfg.stage_time_budget_min = args.stage_time_budget_min
    if args.disable_baseline:
        cfg.enable_baseline = False
    if args.seed is not None:
        cfg.seed = args.seed

    # Determine output directory (include run_id to prevent conflicts between concurrent runs)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        session_name = Path(args.root_session).name
        output_dir = f"saved/staged_training_reports/{session_name}/{run_id}"

    # Run training (with interrupt/crash recovery for final report)
    try:
        run_staged_training(
            root_session_path=args.root_session,
            cfg=cfg,
            output_dir=output_dir,
            run_id=run_id,
            single_stage=args.stage,
            direct_train_session=args.train_session,
            direct_val_session=args.val_session,
            original_session_path=args.original_session,
            starting_checkpoint=args.checkpoint,
        )
    except (KeyboardInterrupt, Exception) as e:
        is_interrupt = isinstance(e, KeyboardInterrupt)
        label = "Interrupted (Ctrl+C)" if is_interrupt else f"Error: {e}"
        print(f"\n{'='*60}")
        print(f"STAGED TRAINING {label}")
        print(f"{'='*60}")

        # Try to recover data from the interrupted stage
        if _partial_report_state and _partial_report_state.get('current_stage_info'):
            stage_info = _partial_report_state['current_stage_info']
            try:
                from concat_world_model_explorer.state import get_checkpoint_dir_for_session
                checkpoint_dir = Path(get_checkpoint_dir_for_session(str(stage_info['train_path'])))
                session_name = Path(stage_info['train_path']).name
                partial_run_id = _partial_report_state['run_id']

                # Find auto-saved checkpoints on disk for this session/run_id
                pattern = f"best_model_auto_{session_name}_{partial_run_id}_*"
                checkpoints = list(checkpoint_dir.glob(pattern))

                if checkpoints:
                    # Parse checkpoint filenames to find best (lowest) loss
                    # Pattern: best_model_auto_{session}_{run_id}_{samples:08d}_{origin}_{type}_{loss:.6f}.pth
                    best_ckpt = None
                    best_loss = float('inf')
                    best_is_val = False
                    for ckpt in checkpoints:
                        # Loss is always the last part before .pth
                        parts = ckpt.stem.split('_')
                        try:
                            loss = float(parts[-1])
                            is_val = (len(parts) >= 2 and parts[-2] == "val")
                            # Prefer val loss checkpoints; among same type, prefer lower loss
                            if best_ckpt is None or (is_val and not best_is_val) or \
                               (is_val == best_is_val and loss < best_loss):
                                best_loss = loss
                                best_ckpt = ckpt
                                best_is_val = is_val
                        except (ValueError, IndexError):
                            continue

                    if best_ckpt:
                        # Try to read samples_seen from checkpoint metadata
                        samples_trained = 0
                        try:
                            ckpt_data = torch.load(str(best_ckpt), map_location='cpu', weights_only=True)
                            samples_trained = ckpt_data.get('samples_seen', 0)
                        except Exception:
                            pass

                        elapsed = time.time() - stage_info['start_time']
                        partial_result = StageResult(
                            stage_num=stage_info['stage_num'],
                            run_num=1,
                            is_baseline=False,
                            best_checkpoint_path=str(best_ckpt),
                            best_loss=best_loss,
                            stop_reason="interrupted",
                            elapsed_time=elapsed,
                            total_samples_trained=samples_trained,
                            cumulative_metrics={},
                            final_train_loss=best_loss,
                            final_val_loss=best_loss if best_is_val else None,
                        )
                        _partial_report_state['completed_stages'].append(
                            (stage_info['stage_num'], partial_result, None)
                        )
                        print(f"Recovered interrupted stage {stage_info['stage_num']} "
                              f"(checkpoint: {best_ckpt.name}, loss: {best_loss:.6f})")
                else:
                    print(f"No checkpoints found for interrupted stage {stage_info['stage_num']}.")
            except Exception as stage_err:
                print(f"Could not recover interrupted stage data: {stage_err}")

        if _partial_report_state and _partial_report_state['completed_stages']:
            print(f"Generating partial report with {len(_partial_report_state['completed_stages'])} completed stage(s)...")
            try:
                report_path = generate_final_report(
                    completed_stages=_partial_report_state['completed_stages'],
                    all_sweep_results=_partial_report_state['all_sweep_results'],
                    all_timings=_partial_report_state['all_timings'],
                    output_dir=_partial_report_state['output_dir'],
                    run_id=_partial_report_state['run_id'],
                    overall_start_time=_partial_report_state['overall_start_time'],
                    original_session=_partial_report_state['original_session'],
                    cfg=_partial_report_state['cfg'],
                    is_final=True,
                    all_stage_runs=_partial_report_state['all_stage_runs'],
                )
                print(f"Partial report saved: {report_path}")
            except Exception as report_err:
                print(f"Failed to generate partial report: {report_err}")
        else:
            print("No completed stages to report.")

        if not is_interrupt:
            raise


if __name__ == "__main__":
    main()
