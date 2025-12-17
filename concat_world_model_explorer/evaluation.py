"""
Full session evaluation for computing model performance metrics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from . import state
from .utils import format_loss
from .canvas_ops import build_canvas_from_frame
from models.autoencoder_concat_predictor import (
    canvas_to_tensor,
    compute_randomized_patch_mask_for_last_slot,
    compute_hybrid_loss_on_masked_patches,
)


def evaluate_full_session():
    """Evaluate model loss on all observations in the session"""
    if state.world_model is None:
        return "Please load a session first", None, None, "", {}

    if not state.session_state.get("observations") or not state.session_state.get("actions"):
        return "No session data available", None, None, "", {}

    observations = state.session_state["observations"]
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE

    # Collect results
    results = {
        'observation_indices': [],
        'loss_hybrid': [],
        'loss_standard': [],
        'loss_plain': [],
        'loss_focal': [],
        'timestamps': [],
    }

    # Iterate through all observations with enough history
    state.world_model.autoencoder.eval()
    for frame_idx in range(min_frames_needed - 1, len(observations)):
        # Build canvas
        training_canvas, error, start_idx, interleaved = build_canvas_from_frame(frame_idx)
        if training_canvas is None:
            continue

        # Compute patch mask
        canvas_tensor = canvas_to_tensor(training_canvas).to(state.device)
        canvas_height, canvas_width = canvas_tensor.shape[-2:]
        num_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE

        patch_mask = compute_randomized_patch_mask_for_last_slot(
            img_size=(canvas_height, canvas_width),
            patch_size=config.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE,
            num_frame_slots=num_frames,
            sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
            mask_ratio_min=config.MASK_RATIO_MIN,
            mask_ratio_max=config.MASK_RATIO_MAX,
        ).to(state.device)

        # Run inference
        with torch.no_grad():
            pred_patches, _ = state.world_model.autoencoder.forward_with_patch_mask(canvas_tensor, patch_mask)
            target_patches = state.world_model.autoencoder.patchify(canvas_tensor)

            # Select masked patches
            masked_pred = pred_patches[patch_mask]
            masked_target = target_patches[patch_mask]

            # Compute loss
            loss_dict = compute_hybrid_loss_on_masked_patches(
                masked_pred,
                masked_target,
                focal_alpha=config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_LOSS_ALPHA,
                focal_beta=config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_BETA
            )

            # Store results
            results['observation_indices'].append(frame_idx)
            results['loss_hybrid'].append(loss_dict['loss_hybrid'].item() if torch.is_tensor(loss_dict['loss_hybrid']) else loss_dict['loss_hybrid'])
            results['loss_standard'].append(loss_dict['loss_standard'].item() if torch.is_tensor(loss_dict['loss_standard']) else loss_dict['loss_standard'])
            results['loss_plain'].append(loss_dict['loss_plain'].item() if torch.is_tensor(loss_dict['loss_plain']) else loss_dict['loss_plain'])
            results['loss_focal'].append(loss_dict['loss_focal'].item() if torch.is_tensor(loss_dict['loss_focal']) else loss_dict['loss_focal'])
            results['timestamps'].append(observations[frame_idx]['timestamp'])

    if len(results['observation_indices']) == 0:
        return "No observations could be evaluated", None, None, "", {}

    # Compute statistics
    loss_hybrid_array = np.array(results['loss_hybrid'])
    loss_standard_array = np.array(results['loss_standard'])

    # Check for NaN values
    if np.any(np.isnan(loss_hybrid_array)) or np.any(np.isnan(loss_standard_array)):
        nan_count = np.sum(np.isnan(loss_hybrid_array))
        error_msg = f"❌ Evaluation failed: {nan_count}/{len(loss_hybrid_array)} loss values are NaN\n\n"
        error_msg += "This indicates the model weights are corrupted (likely from gradient explosion during training).\n\n"
        error_msg += "**Solutions:**\n"
        error_msg += "1. Reload the page to reset the model\n"
        error_msg += "2. Load a saved checkpoint\n"
        error_msg += "3. Reduce learning rate in config.py and retrain\n"
        error_msg += "4. Use gradient clipping (add to train_on_canvas)"
        return error_msg, None, None, "", {}

    stats = {
        'num_observations': len(results['observation_indices']),
        'hybrid': {
            'mean': np.mean(loss_hybrid_array),
            'median': np.median(loss_hybrid_array),
            'std': np.std(loss_hybrid_array),
            'min': np.min(loss_hybrid_array),
            'max': np.max(loss_hybrid_array),
            'p25': np.percentile(loss_hybrid_array, 25),
            'p75': np.percentile(loss_hybrid_array, 75),
            'p90': np.percentile(loss_hybrid_array, 90),
            'p95': np.percentile(loss_hybrid_array, 95),
            'p99': np.percentile(loss_hybrid_array, 99),
        },
        'standard': {
            'mean': np.mean(loss_standard_array),
            'median': np.median(loss_standard_array),
            'std': np.std(loss_standard_array),
            'min': np.min(loss_standard_array),
            'max': np.max(loss_standard_array),
        }
    }

    # Create statistics display
    stats_text = f"## Evaluation Statistics\n\n"
    stats_text += f"**Evaluated {stats['num_observations']} observations** (from {len(observations)} total)\n\n"
    stats_text += f"### Hybrid Loss (Training Objective)\n"
    stats_text += f"| Statistic | Value |\n"
    stats_text += f"|-----------|-------|\n"
    stats_text += f"| Mean | {format_loss(stats['hybrid']['mean'])} |\n"
    stats_text += f"| Median | {format_loss(stats['hybrid']['median'])} |\n"
    stats_text += f"| Std Dev | {format_loss(stats['hybrid']['std'])} |\n"
    stats_text += f"| Min | {format_loss(stats['hybrid']['min'])} |\n"
    stats_text += f"| Max | {format_loss(stats['hybrid']['max'])} |\n"
    stats_text += f"| 25th %ile | {format_loss(stats['hybrid']['p25'])} |\n"
    stats_text += f"| 75th %ile | {format_loss(stats['hybrid']['p75'])} |\n"
    stats_text += f"| 90th %ile | {format_loss(stats['hybrid']['p90'])} |\n"
    stats_text += f"| 95th %ile | {format_loss(stats['hybrid']['p95'])} |\n"
    stats_text += f"| 99th %ile | {format_loss(stats['hybrid']['p99'])} |\n"
    stats_text += f"\n### Standard Loss (Unweighted MSE)\n"
    stats_text += f"| Statistic | Value |\n"
    stats_text += f"|-----------|-------|\n"
    stats_text += f"| Mean | {format_loss(stats['standard']['mean'])} |\n"
    stats_text += f"| Median | {format_loss(stats['standard']['median'])} |\n"
    stats_text += f"| Std Dev | {format_loss(stats['standard']['std'])} |\n"
    stats_text += f"| Min | {format_loss(stats['standard']['min'])} |\n"
    stats_text += f"| Max | {format_loss(stats['standard']['max'])} |\n"

    # Create line plot: Loss over observations
    fig_loss_over_time, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot hybrid loss
    axes[0].plot(results['observation_indices'], results['loss_hybrid'], 'b-', linewidth=1, alpha=0.7, label='Hybrid Loss')
    axes[0].axhline(y=stats['hybrid']['mean'], color='r', linestyle='--', linewidth=1.5, label=f"Mean: {format_loss(stats['hybrid']['mean'])}")
    axes[0].axhline(y=stats['hybrid']['median'], color='g', linestyle='--', linewidth=1.5, label=f"Median: {format_loss(stats['hybrid']['median'])}")
    axes[0].set_xlabel('Observation Index')
    axes[0].set_ylabel('Hybrid Loss')
    axes[0].set_title('Hybrid Loss Over Session')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot standard loss
    axes[1].plot(results['observation_indices'], results['loss_standard'], 'purple', linewidth=1, alpha=0.7, label='Standard Loss')
    axes[1].axhline(y=stats['standard']['mean'], color='r', linestyle='--', linewidth=1.5, label=f"Mean: {format_loss(stats['standard']['mean'])}")
    axes[1].axhline(y=stats['standard']['median'], color='g', linestyle='--', linewidth=1.5, label=f"Median: {format_loss(stats['standard']['median'])}")
    axes[1].set_xlabel('Observation Index')
    axes[1].set_ylabel('Standard Loss (MSE)')
    axes[1].set_title('Standard Loss Over Session')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Create distribution histogram
    fig_distribution, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Hybrid loss distribution
    axes[0].hist(results['loss_hybrid'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=stats['hybrid']['mean'], color='r', linestyle='--', linewidth=2, label=f"Mean: {format_loss(stats['hybrid']['mean'])}")
    axes[0].axvline(x=stats['hybrid']['median'], color='g', linestyle='--', linewidth=2, label=f"Median: {format_loss(stats['hybrid']['median'])}")
    axes[0].set_xlabel('Hybrid Loss')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Hybrid Loss Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Standard loss distribution
    axes[1].hist(results['loss_standard'], bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=stats['standard']['mean'], color='r', linestyle='--', linewidth=2, label=f"Mean: {format_loss(stats['standard']['mean'])}")
    axes[1].axvline(x=stats['standard']['median'], color='g', linestyle='--', linewidth=2, label=f"Median: {format_loss(stats['standard']['median'])}")
    axes[1].set_xlabel('Standard Loss (MSE)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Standard Loss Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    status_msg = f"✅ **Evaluation complete!** Processed {stats['num_observations']} observations."

    return status_msg, fig_loss_over_time, fig_distribution, stats_text, stats
