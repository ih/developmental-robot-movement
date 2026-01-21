"""
Visualization functions for training progress, loss plots, and observation grids.
"""

import random
import math
from datetime import timedelta
import matplotlib.pyplot as plt
import torch

import config
from . import state
from .canvas_ops import build_canvas_from_frame
from .utils import compute_canvas_figsize
from models.autoencoder_concat_predictor import (
    canvas_to_tensor,
    compute_randomized_patch_mask_for_last_slot,
)


def create_loss_vs_samples_plot(cumulative_metrics):
    """Create plot of loss vs samples seen, with optional validation loss"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Training loss (blue solid line)
    ax.plot(cumulative_metrics['samples_seen'],
            cumulative_metrics['loss_at_sample'],
            'b-o', linewidth=2, markersize=6, label='Train Loss')

    # Validation loss (orange dashed line) - if available
    val_losses = cumulative_metrics.get('val_loss_at_sample', [])
    if val_losses and any(v is not None for v in val_losses):
        # Filter out None values for plotting
        valid_val_data = [(s, v) for s, v in zip(cumulative_metrics['samples_seen'], val_losses) if v is not None]
        if valid_val_data:
            val_samples, val_loss_values = zip(*valid_val_data)
            ax.plot(val_samples, val_loss_values,
                    'orange', linestyle='--', marker='s', linewidth=2, markersize=6, label='Val Loss')
            ax.legend(loc='upper right')

    ax.set_xlabel('Samples Seen', fontsize=12)
    ax.set_ylabel('Mean Hybrid Loss', fontsize=12)
    ax.set_title('Training Progress: Loss vs Samples', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_loss_vs_recent_checkpoints_plot(cumulative_metrics, window_size=10):
    """Create plot of loss vs recent checkpoints (rolling window), with optional validation loss"""
    if not cumulative_metrics['samples_seen']:
        return None

    # Get last N checkpoints
    samples = cumulative_metrics['samples_seen'][-window_size:]
    losses = cumulative_metrics['loss_at_sample'][-window_size:]

    if not samples:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Training loss (green solid line)
    ax.plot(samples, losses, 'g-o', linewidth=2, markersize=6, label='Train Loss')

    # Validation loss (orange dashed line) - if available
    val_losses = cumulative_metrics.get('val_loss_at_sample', [])
    if val_losses:
        val_losses_window = val_losses[-window_size:]
        # Filter out None values
        valid_val_data = [(s, v) for s, v in zip(samples, val_losses_window) if v is not None]
        if valid_val_data:
            val_samples, val_loss_values = zip(*valid_val_data)
            ax.plot(val_samples, val_loss_values,
                    'orange', linestyle='--', marker='s', linewidth=2, markersize=6, label='Val Loss')
            ax.legend(loc='upper right')

    ax.set_xlabel('Samples Seen', fontsize=12)
    ax.set_ylabel('Mean Hybrid Loss', fontsize=12)
    ax.set_title(f'Recent Progress: Last {len(samples)} Checkpoints', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_lr_vs_samples_plot(cumulative_metrics):
    """Create plot of learning rate vs samples seen"""
    if not cumulative_metrics.get('samples_seen') or not cumulative_metrics.get('lr_at_sample'):
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_metrics['samples_seen'],
            cumulative_metrics['lr_at_sample'],
            'purple', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Samples Seen', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')  # Log scale for better visualization of LR decay
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_sample_weights_plot(weights, frame_indices, temperature=None, sample_counts=None):
    """
    Create stem plot of sample weights vs frame numbers for loss-weighted sampling.

    Helps users identify high-weight frames for inspection. When loss-weighted
    sampling is active, higher weight = higher loss = more likely to be sampled
    for training.

    Args:
        weights: torch.Tensor or list of sample weights (normalized probabilities)
        frame_indices: List of frame indices corresponding to weights
        temperature: Optional temperature value for title context
        sample_counts: Optional dict mapping frame_idx -> times sampled

    Returns:
        matplotlib.figure.Figure or None if no valid data
    """
    import numpy as np

    # Validate inputs
    if weights is None or frame_indices is None:
        return None

    # Handle empty containers
    try:
        if len(weights) == 0 or len(frame_indices) == 0 or len(weights) != len(frame_indices):
            return None
    except (TypeError, AttributeError):
        return None

    # Data preparation
    if hasattr(weights, 'cpu'):  # torch.Tensor
        weights = weights.cpu().numpy()
    else:
        weights = np.array(weights)

    frame_indices = np.array(frame_indices)

    # Sort by frame index for cleaner visualization
    sort_idx = np.argsort(frame_indices)
    frame_indices = frame_indices[sort_idx]
    weights = weights[sort_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Adaptive visualization based on data size
    num_frames = len(frame_indices)
    if num_frames > 1000:
        # Scatter plot for many frames (lighter rendering)
        ax.scatter(frame_indices, weights, s=20, alpha=0.7, color='purple', label='Weight')
    else:
        # Stem plot (lollipop chart) for <= 1000 frames
        markerline, stemlines, baseline = ax.stem(
            frame_indices, weights,
            linefmt='purple', markerfmt='o',
            basefmt='gray'
        )
        markerline.set_markersize(4)
        stemlines.set_linewidth(1)

    # Styling
    title = "Sample Weights Distribution (Loss-Weighted Sampling"
    if temperature is not None:
        title += f", temp={temperature:.2f}"
    title += ")"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Sample Weight (Probability)', fontsize=12, color='purple')
    ax.tick_params(axis='y', labelcolor='purple')
    ax.grid(True, alpha=0.3)

    # Set y-axis range to show full scale
    ax.set_ylim([0, np.max(weights) * 1.1])

    # Add sample counts on secondary y-axis if provided
    if sample_counts is not None and len(sample_counts) > 0:
        ax2 = ax.twinx()
        counts = np.array([sample_counts.get(int(idx), 0) for idx in frame_indices])
        ax2.bar(frame_indices, counts, alpha=0.3, color='green', width=0.8, label='Times Sampled')
        ax2.set_ylabel('Times Sampled', fontsize=12, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        # Set y-axis range to show full scale
        if np.max(counts) > 0:
            ax2.set_ylim([0, np.max(counts) * 1.1])

    # Add statistics text box
    max_idx = np.argmax(weights)
    stats_text = f"Tracked: {len(weights)} frames\n"
    stats_text += f"Max weight: {weights[max_idx]:.6f} @ frame {frame_indices[max_idx]}\n"
    stats_text += f"Mean weight: {np.mean(weights):.6f}\n"
    stats_text += f"Min weight: {np.min(weights):.6f}"

    if sample_counts is not None and len(sample_counts) > 0:
        total_samples = sum(sample_counts.values())
        max_count = max(sample_counts.values()) if sample_counts else 0
        stats_text += f"\nTotal samples: {total_samples:,}"
        stats_text += f"\nMax count: {max_count}"

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

    plt.tight_layout()
    return fig


def generate_multiple_observation_canvases(observation_indices):
    """Generate grid of canvas visualizations for multiple observations"""
    if state.world_model is None:
        return "Please load session and train model first", None

    if not observation_indices:
        return "No observations to visualize", None

    observations = state.session_state.get("observations", [])
    valid_indices = [idx for idx in observation_indices if idx < len(observations)]

    if not valid_indices:
        return "No valid observation indices", None

    # Build first canvas to get dimensions for figsize calculation
    first_canvas, first_error, _, _ = build_canvas_from_frame(valid_indices[0])
    if first_canvas is not None:
        canvas_h, canvas_w = first_canvas.shape[:2]
        # Compute per-row height based on canvas aspect ratio
        single_figsize = compute_canvas_figsize(canvas_h, canvas_w, fig_width=16.0)
        row_height = single_figsize[1]
    else:
        row_height = 4  # Fallback

    # Create grid: N rows × 2 columns (original + composite)
    n_obs = len(valid_indices)
    fig, axes = plt.subplots(n_obs, 2, figsize=(16, row_height * n_obs))

    # Handle single observation case (axes won't be 2D)
    if n_obs == 1:
        axes = axes.reshape(1, -1)

    for i, obs_idx in enumerate(valid_indices):
        # Build canvas (reuse first canvas if this is the first observation)
        if i == 0 and first_canvas is not None:
            training_canvas, error = first_canvas, first_error
        else:
            training_canvas, error, _, _ = build_canvas_from_frame(obs_idx)
        if training_canvas is None:
            # Show error in both subplots
            axes[i, 0].text(0.5, 0.5, f"Error: {error}", ha='center', va='center')
            axes[i, 0].axis('off')
            axes[i, 1].text(0.5, 0.5, f"Error: {error}", ha='center', va='center')
            axes[i, 1].axis('off')
            continue

        # Convert to tensor and generate mask
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
        state.world_model.autoencoder.eval()
        with torch.no_grad():
            pred_patches, _ = state.world_model.autoencoder.forward_with_patch_mask(
                canvas_tensor, patch_mask
            )

        # Get composite
        composite = state.world_model.get_canvas_inpainting_composite(training_canvas, patch_mask)

        # Plot original canvas
        axes[i, 0].imshow(training_canvas)
        axes[i, 0].set_title(f"Obs {obs_idx}: Original Canvas", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Plot composite
        axes[i, 1].imshow(composite)
        axes[i, 1].set_title(f"Obs {obs_idx}: Composite", fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

    plt.tight_layout()

    status = f"**Showing {len(valid_indices)} observations**: {valid_indices}"
    return status, fig


def generate_batch_training_update(samples_seen, total_samples, cumulative_metrics,
                                   eval_status, eval_loss_fig, eval_dist_fig,
                                   current_observation_idx, window_size=10, num_random_obs=5,
                                   completed=False, elapsed_time=None,
                                   sampling_mode=None, sample_weights_data=None):
    """Generate all UI outputs for batch training update"""
    # Status message
    if completed:
        # Use eval_status if it contains completion info (starts with **Training)
        if eval_status and eval_status.startswith("**Training"):
            status = eval_status
        else:
            status = f"✅ **Training Complete: {samples_seen} samples**"
            if elapsed_time is not None:
                td = timedelta(seconds=int(elapsed_time))
                status += f"\n\n⏱️ **Time elapsed:** {td}"
    else:
        progress_pct = (samples_seen / total_samples) * 100
        status = f"🔄 **Training... {samples_seen}/{total_samples} samples ({progress_pct:.1f}%)**"
        if elapsed_time is not None:
            td = timedelta(seconds=int(elapsed_time))
            status += f"\n\n⏱️ **Elapsed:** {td}"

    # Loss vs samples plot (full history)
    fig_loss_vs_samples = create_loss_vs_samples_plot(cumulative_metrics)

    # Create rolling window plot
    fig_loss_vs_recent = create_loss_vs_recent_checkpoints_plot(cumulative_metrics, window_size)

    # Learning rate vs samples plot
    fig_lr_vs_samples = create_lr_vs_samples_plot(cumulative_metrics)

    # Sample weights plot (loss-weighted sampling only)
    fig_sample_weights = None
    if sampling_mode == "Loss-weighted" and sample_weights_data is not None:
        # Unpack data - supports both 3-tuple (legacy) and 4-tuple (with sample_counts)
        if len(sample_weights_data) == 4:
            weights, valid_indices, temperature, sample_counts = sample_weights_data
        else:
            weights, valid_indices, temperature = sample_weights_data
            sample_counts = None
        fig_sample_weights = create_sample_weights_plot(weights, valid_indices, temperature, sample_counts)

    # Sample observations: N random + current frame
    observations = state.session_state.get("observations", [])
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    all_valid_indices = list(range(min_frames_needed - 1, len(observations)))

    if all_valid_indices:
        # Sample N random observations
        sample_size = min(num_random_obs, len(all_valid_indices))
        sampled_indices = random.sample(all_valid_indices, sample_size)

        # Add current observation if valid and not already sampled
        if current_observation_idx in all_valid_indices and current_observation_idx not in sampled_indices:
            sampled_indices.append(current_observation_idx)

        obs_status, obs_combined_fig = generate_multiple_observation_canvases(sampled_indices)
    else:
        obs_status = "No valid observations"
        obs_combined_fig = None

    return (status, fig_loss_vs_samples, fig_loss_vs_recent, fig_lr_vs_samples,
            fig_sample_weights,  # Sample weights distribution (loss-weighted mode only)
            eval_loss_fig, eval_dist_fig, obs_status, obs_combined_fig)


def calculate_training_info(total_samples, batch_size, sampling_mode="Epoch-based (shuffle each epoch)"):
    """Calculate and display training parameters with linear scaling rule"""
    # Validate inputs
    try:
        total_samples = int(total_samples)
        batch_size = int(batch_size)
    except (ValueError, TypeError):
        return "Invalid inputs"

    if total_samples <= 0 or batch_size <= 0:
        return "Total samples and batch size must be greater than 0"

    # Calculate number of gradient updates
    num_gradient_updates = math.ceil(total_samples / batch_size)

    # Linear scaling rule: scale LR proportionally with batch size
    # base_batch_size = 1 (base LR is defined for batch size 1)
    base_batch_size = 1
    base_lr = config.AutoencoderConcatPredictorWorldModelConfig.AUTOENCODER_LR  # 1e-3

    # Scaled LR = base_LR * (current_batch_size / base_batch_size)
    # For BS=1: LR=1e-3, BS=32: LR=0.032, BS=64: LR=0.064
    scaled_lr = base_lr * (batch_size / base_batch_size)

    # Scheduler parameters
    lr_min_ratio = config.AutoencoderConcatPredictorWorldModelConfig.LR_MIN_RATIO  # 0.01
    min_lr = scaled_lr * lr_min_ratio

    # Warmup steps (scaled inversely with batch size to see same data)
    base_warmup_steps = config.AutoencoderConcatPredictorWorldModelConfig.WARMUP_STEPS
    scaled_warmup_steps = max(1, int(base_warmup_steps / batch_size))
    warmup_samples_seen = scaled_warmup_steps * batch_size
    warmup_percentage = (scaled_warmup_steps / num_gradient_updates) * 100 if num_gradient_updates > 0 else 0

    # Format output
    info = f"""**Training Configuration**

📊 **Gradient Updates**: {num_gradient_updates:,} steps
- Total samples: {total_samples:,}
- Batch size: {batch_size}
- Sampling mode: {sampling_mode}
- Updates per epoch: {num_gradient_updates:,}

📈 **Learning Rate (Linear Scaling)**
- Base LR (BS={base_batch_size}): {base_lr:.6f}
- Scaled LR (BS={batch_size}): {scaled_lr:.6f}
- Scaling factor: {batch_size / base_batch_size:.2f}x
- Min LR (cosine decay): {min_lr:.6f}

🔄 **Scheduler (Warmup + Cosine Annealing)**
- Total steps: {num_gradient_updates:,}
- Warmup (base): {base_warmup_steps} steps @ BS={base_batch_size} = {base_warmup_steps} samples
- Warmup (adjusted): {scaled_warmup_steps} steps @ BS={batch_size} = {warmup_samples_seen} samples
- Warmup duration: {warmup_percentage:.1f}% of training
- LR during warmup: 0.0 → {scaled_lr:.6f} (linear ramp)
- LR after warmup: {scaled_lr:.6f} → {min_lr:.6f} (cosine decay)
"""

    return info
