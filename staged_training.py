"""
Staged Training Script

Trains progressively on staged splits of a session, using batch training
with loss-weighted sampling and divergence-based early stopping.

Usage:
    python staged_training.py --root-session saved/sessions/so101/my_session
    python staged_training.py --root-session saved/sessions/so101/my_session --runs-per-stage 3
    python staged_training.py --root-session saved/sessions/so101/my_session --config my_config.yaml

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
import io
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
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


@dataclass
class StageResult:
    """Result from training a single stage/run."""
    stage_num: int
    run_num: int
    best_checkpoint_path: str
    best_loss: float
    stop_reason: str
    elapsed_time: float
    total_samples_trained: int
    cumulative_metrics: dict
    final_train_loss: float
    final_val_loss: Optional[float]


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


def run_stage_training(
    stage_num: int,
    run_num: int,
    train_session: dict,
    val_session: dict,
    checkpoint_path: Optional[str],
    cfg: StagedTrainingConfig,
    output_dir: Path,
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

    Returns:
        StageResult with training outcomes
    """
    print(f"\n{'='*60}")
    print(f"Stage {stage_num} Run {run_num}")
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

    # W&B run name
    wandb_run_name = f"{train_session['session_name']}_run{run_num}"

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

    # Run batch training generator
    try:
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
            resume_mode=resume_mode,
            samples_mode=cfg.samples_mode,
            starting_samples=starting_samples,
            preserve_optimizer=cfg.preserve_optimizer,
            preserve_scheduler=cfg.preserve_scheduler,
            # Learning rate
            custom_lr=cfg.custom_lr,
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
            # Validation plateau early stopping
            val_plateau_patience=cfg.val_plateau_patience,
            val_plateau_min_delta=cfg.val_plateau_min_delta,
            # ReduceLROnPlateau
            plateau_factor=cfg.plateau_factor,
            plateau_patience=cfg.plateau_patience,
        )

        # Consume generator and collect results
        for result in generator:
            if result is None:
                continue

            # Unpack result tuple (matches generate_batch_training_update output)
            (status_msg, loss_fig, loss_recent_fig, lr_fig, weights_fig,
             eval_loss_fig, eval_dist_fig, obs_status, obs_fig) = result

            # Parse status for metrics
            if "samples" in status_msg.lower():
                # Extract samples count from status
                match = re.search(r"(\d+)\s*/\s*(\d+|\?+)", status_msg)
                if match:
                    samples_seen = int(match.group(1))

            # Check for stop reason in status (use specific phrases to avoid false positives)
            status_lower = status_msg.lower()
            if "nan" in status_lower or "inf loss" in status_lower:
                stop_reason = "nan_loss"
            elif "divergence detected" in status_lower:
                stop_reason = "divergence"
            elif "validation plateau" in status_lower:
                stop_reason = "val_plateau"
            elif "training complete" in status_lower or "training finished" in status_lower:
                stop_reason = "completed"

            # Close figures to free memory
            for fig in [loss_fig, loss_recent_fig, lr_fig, weights_fig,
                        eval_loss_fig, eval_dist_fig, obs_fig]:
                if fig is not None:
                    plt.close(fig)

    except Exception as e:
        print(f"Training error: {e}")
        stop_reason = f"error: {str(e)}"

    elapsed_time = time.time() - start_time

    # Make stop_reason more informative for sample budget completion
    if stop_reason in ("completed", "max_samples"):
        if cfg.stage_samples_multiplier > 0:
            stop_reason = f"sample_budget ({num_valid_frames} frames x {cfg.stage_samples_multiplier} = {effective_total_samples:,})"
        else:
            stop_reason = f"sample_budget ({effective_total_samples:,} fixed)"

    # Extract total_samples_trained from cumulative_metrics
    if state.cumulative_metrics and state.cumulative_metrics.get("samples_seen"):
        total_samples_trained = state.cumulative_metrics["samples_seen"][-1] - starting_samples
    else:
        total_samples_trained = 0

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
        best_checkpoint_path=best_checkpoint_path,
        best_loss=best_loss,
        stop_reason=stop_reason,
        elapsed_time=elapsed_time,
        total_samples_trained=total_samples_trained,
        cumulative_metrics=state.cumulative_metrics,  # Get from state (populated by run_world_model_batch)
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
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
            <strong>Hybrid Loss:</strong> {result.best_loss:.6f}
        </div>
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


def generate_final_report(
    all_results: list[tuple[int, StageResult]],
    original_session: dict,
    output_dir: Path,
    cfg: StagedTrainingConfig,
) -> None:
    """Generate final summary HTML report."""
    print("\n" + "=" * 60)
    print("Generating Final Summary Report")
    print("=" * 60)

    # Calculate totals
    total_time = sum(r.elapsed_time for _, r in all_results)

    # Evaluate each stage's checkpoint on the original session
    # This determines the true best checkpoint and provides loss graphs
    print("Evaluating all stage checkpoints on original session...")
    stage_original_evals = []  # List of (stage_num, result, orig_loss, loss_fig_b64)

    for stage_num, result in all_results:
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

    # Find best checkpoint based on original session loss (not training loss)
    best_eval = min(stage_original_evals, key=lambda x: x[2])
    best_stage, best_run_result, best_orig_loss, _ = best_eval
    print(f"Best checkpoint: Stage {best_stage} with original session loss {best_orig_loss:.6f}")

    # Create stage progression plot using original session losses
    fig_progression, ax = plt.subplots(figsize=(10, 6))
    stages = [s for s, _, _, _ in stage_original_evals]
    orig_losses = [loss for _, _, loss, _ in stage_original_evals]
    ax.plot(stages, orig_losses, "o-", linewidth=2, markersize=8, label="Original Session Loss")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Hybrid Loss (Original Session)")
    ax.set_title("Stage Progression: Loss on Original Session")
    ax.grid(True, alpha=0.3)
    ax.legend()
    progression_b64 = fig_to_base64(fig_progression)

    # Generate learning rate graph for best stage
    lr_b64 = ""
    if best_run_result.cumulative_metrics.get("lr_at_sample"):
        fig_lr = visualization.create_lr_vs_samples_plot(best_run_result.cumulative_metrics)
        if fig_lr:
            lr_b64 = fig_to_base64(fig_lr)
            plt.close(fig_lr)

    # Generate inference for best checkpoint
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

    # Build stage table with original session loss
    stage_rows = ""
    for stage_num, result, orig_loss, _ in stage_original_evals:
        is_best = stage_num == best_stage
        row_style = ' style="background: #e8f5e9; font-weight: bold;"' if is_best else ''
        best_marker = " ⭐" if is_best else ""
        stage_rows += f"""
        <tr{row_style}>
            <td>{stage_num}{best_marker}</td>
            <td>{orig_loss:.6f}</td>
            <td>{result.best_loss:.6f}</td>
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
                <h4>Stage {stage_num}{best_label} - Hybrid Loss: {orig_loss:.6f}</h4>
                <img src="data:image/png;base64,{loss_fig_b64}" />
            </div>"""

    # Build cumulative sample counts across all stages
    cumulative_sample_counts: dict[int, int] = {}
    for stage_num, result, orig_loss, _ in stage_original_evals:
        stage_counts = result.cumulative_metrics.get("sample_seen_counts", {})
        for frame_idx, count in stage_counts.items():
            cumulative_sample_counts[frame_idx] = cumulative_sample_counts.get(frame_idx, 0) + count

    # Generate cumulative sample counts plot
    cumulative_counts_b64 = ""
    if cumulative_sample_counts:
        fig_cumulative = visualization.create_sample_counts_plot(
            cumulative_sample_counts, title="Cumulative Sample Counts (All Stages)"
        )
        if fig_cumulative:
            cumulative_counts_b64 = fig_to_base64(fig_cumulative)
            plt.close(fig_cumulative)

    # Build per-stage sample counts graphs section
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

    # Build inference HTML
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

    # Stage report links
    stage_links = ""
    for stage_num, result, orig_loss, _ in stage_original_evals:
        stage_dir = f"stage{stage_num}_run{result.run_num}"
        is_best = stage_num == best_stage
        best_marker = " ⭐" if is_best else ""
        stage_links += f'<li><a href="{stage_dir}/report.html">Stage {stage_num} Run {result.run_num}</a> (orig loss: {orig_loss:.6f}){best_marker}</li>'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Staged Training Final Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
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
    </style>
</head>
<body>
<div class="container">
    <h1>Staged Training Final Summary</h1>

    <section id="total-time">
        <h2>Total Training Time</h2>
        <div class="metric"><strong>Duration:</strong> {format_duration(total_time)}</div>
    </section>

    <section id="best-checkpoint">
        <h2>Best Checkpoint</h2>
        <div class="metric">
            <strong>Name:</strong> {Path(best_run_result.best_checkpoint_path).name}<br>
            <strong>Stage:</strong> {best_stage}<br>
            <strong>Hybrid Loss (full session):</strong> {best_orig_loss:.6f}
        </div>
    </section>

    <section id="learning-rate">
        <h2>Learning Rate Schedule (Best Stage)</h2>
        {'<img src="data:image/png;base64,' + lr_b64 + '" />' if lr_b64 else "<p>No learning rate data available</p>"}
    </section>

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
            {stage_rows}
        </table>
        <h3>Hybrid Loss Over Original Session (per Stage)</h3>
        <div class="stage-loss-graphs">
            {stage_loss_graphs_html if stage_loss_graphs_html else "<p>No loss graphs available</p>"}
        </div>
    </section>

    <section id="sample-counts">
        <h2>Sample Counts</h2>
        <h3>Cumulative Across All Stages</h3>
        {'<img src="data:image/png;base64,' + cumulative_counts_b64 + '" />' if cumulative_counts_b64 else "<p>No cumulative sample count data available</p>"}
        <h3>Per Stage</h3>
        <div class="stage-loss-graphs">
            {stage_counts_graphs_html if stage_counts_graphs_html else "<p>No per-stage sample count graphs available</p>"}
        </div>
    </section>

    <section id="best-inference">
        <h2>Best Checkpoint Inference</h2>
        <h3>Selected Frame {selected_frame}</h3>
        <div class="inference-row">
            {selected_html if selected_html else "<p>No inference images available</p>"}
        </div>
        <h3>Random Observations</h3>
        {random_html if random_html else "<p>No random observation images available</p>"}
    </section>

    <section id="stage-links">
        <h2>Individual Stage Reports</h2>
        <ul>
            {stage_links}
        </ul>
    </section>
</div>
</body>
</html>"""

    with open(output_dir / "final_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    # Save summary JSON
    summary = {
        "total_time_seconds": total_time,
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
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nFinal report saved to {output_dir / 'final_report.html'}")


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
) -> None:
    """
    Main function to run staged training.

    Args:
        root_session_path: Path to root session with staged splits
        cfg: Training configuration
        output_dir: Output directory for reports
        run_id: Unique identifier for this run (used in checkpoint names to avoid collisions)
    """
    # Set instance_id for checkpoint naming (prevents collisions with concurrent runs)
    state.instance_id = run_id
    print(f"Run ID: {run_id}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save run_id to config for reference/reproducibility
    cfg.run_id = run_id

    # Save config
    cfg.to_yaml(str(output_path / "config.yaml"))

    # Discover stages
    print("Discovering staged splits...")
    stages = discover_staged_splits(root_session_path)
    if not stages:
        raise ValueError(f"No staged splits found for {root_session_path}")

    print(f"Found {len(stages)} stages:")
    for stage_num, train_path, val_path in stages:
        print(f"  Stage {stage_num}: train={Path(train_path).name}, val={Path(val_path).name}")

    # Load original session for final evaluation
    print("\nLoading original session...")
    original_session = load_session_for_training(root_session_path)

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

    # Run training for each stage
    all_results = []
    current_checkpoint = None  # Fresh weights for stage 1

    for stage_num, train_path, val_path in stages:
        print(f"\n{'#'*60}")
        print(f"# STAGE {stage_num}")
        print(f"{'#'*60}")

        # Load sessions
        print(f"Loading training session: {Path(train_path).name}")
        train_session = load_session_for_training(train_path)

        print(f"Loading validation session: {Path(val_path).name}")
        val_session = load_session_for_training(val_path)

        stage_results = []

        for run_num in range(1, cfg.runs_per_stage + 1):
            run_output_dir = output_path / f"stage{stage_num}_run{run_num}"

            # Run training
            result = run_stage_training(
                stage_num=stage_num,
                run_num=run_num,
                train_session=train_session,
                val_session=val_session,
                checkpoint_path=current_checkpoint,
                cfg=cfg,
                output_dir=run_output_dir,
            )

            stage_results.append(result)

            # Generate stage report
            generate_stage_report(
                result=result,
                train_session=train_session,
                val_session=val_session,
                original_session=original_session,
                cfg=cfg,
                output_dir=run_output_dir,
            )

        # Select best checkpoint from all runs
        best_run = min(stage_results, key=lambda r: r.best_loss)
        print(f"\nBest run for stage {stage_num}: Run {best_run.run_num} (loss: {best_run.best_loss:.6f})")

        # Update current checkpoint for next stage
        current_checkpoint = best_run.best_checkpoint_path

        # Cleanup non-best checkpoints
        if cfg.runs_per_stage > 1:
            print("Cleaning up non-best checkpoints...")
            cleanup_stage_checkpoints(stage_results, best_run)

        all_results.append((stage_num, best_run))

    # Generate final summary
    generate_final_report(all_results, original_session, output_path, cfg)

    print("\n" + "=" * 60)
    print("STAGED TRAINING COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"Final report: {output_path / 'final_report.html'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run staged training on progressively larger session splits"
    )
    parser.add_argument(
        "--root-session",
        required=True,
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
        "--run-id",
        help="Unique run identifier for concurrent execution (default: auto-generated timestamp)",
    )

    args = parser.parse_args()

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

    # Determine output directory (include run_id to prevent conflicts between concurrent runs)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        session_name = Path(args.root_session).name
        output_dir = f"saved/staged_training_reports/{session_name}/{run_id}"

    # Run training
    run_staged_training(args.root_session, cfg, output_dir, run_id)


if __name__ == "__main__":
    main()
