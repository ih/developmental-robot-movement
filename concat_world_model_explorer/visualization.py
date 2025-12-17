"""
Visualization functions for training progress, loss plots, and observation grids.
"""

import random
import math
import matplotlib.pyplot as plt
import torch

import config
from . import state
from .canvas_ops import build_canvas_from_frame
from models.autoencoder_concat_predictor import (
    canvas_to_tensor,
    compute_randomized_patch_mask_for_last_slot,
)


def create_loss_vs_samples_plot(cumulative_metrics):
    """Create plot of loss vs samples seen"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_metrics['samples_seen'],
            cumulative_metrics['loss_at_sample'],
            'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Samples Seen', fontsize=12)
    ax.set_ylabel('Mean Hybrid Loss (Full Session Eval)', fontsize=12)
    ax.set_title('Training Progress: Loss vs Samples', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_loss_vs_recent_checkpoints_plot(cumulative_metrics, window_size=10):
    """Create plot of loss vs recent checkpoints (rolling window)"""
    if not cumulative_metrics['samples_seen']:
        return None

    # Get last N checkpoints
    samples = cumulative_metrics['samples_seen'][-window_size:]
    losses = cumulative_metrics['loss_at_sample'][-window_size:]

    if not samples:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(samples, losses, 'g-o', linewidth=2, markersize=6)
    ax.set_xlabel('Samples Seen', fontsize=12)
    ax.set_ylabel('Mean Hybrid Loss', fontsize=12)
    ax.set_title(f'Recent Progress: Last {len(samples)} Checkpoints', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
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

    # Create grid: N rows × 2 columns (original + composite)
    n_obs = len(valid_indices)
    fig, axes = plt.subplots(n_obs, 2, figsize=(16, 4 * n_obs))

    # Handle single observation case (axes won't be 2D)
    if n_obs == 1:
        axes = axes.reshape(1, -1)

    for i, obs_idx in enumerate(valid_indices):
        # Build canvas
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
                                   completed=False, elapsed_time=None):
    """Generate all UI outputs for batch training update"""
    # Status message
    if completed:
        status = f"✅ **Training Complete: {samples_seen} samples**"
        if elapsed_time is not None:
            mins = int(elapsed_time // 60)
            secs = elapsed_time % 60
            if mins > 0:
                status += f"\n\n⏱️ **Time elapsed:** {mins}m {secs:.1f}s"
            else:
                status += f"\n\n⏱️ **Time elapsed:** {secs:.1f}s"
    else:
        progress_pct = (samples_seen / total_samples) * 100
        status = f"🔄 **Training... {samples_seen}/{total_samples} samples ({progress_pct:.1f}%)**"
        if elapsed_time is not None:
            mins = int(elapsed_time // 60)
            secs = elapsed_time % 60
            if mins > 0:
                status += f"\n\n⏱️ **Elapsed:** {mins}m {secs:.1f}s"
            else:
                status += f"\n\n⏱️ **Elapsed:** {secs:.1f}s"

    # Loss vs samples plot (full history)
    fig_loss_vs_samples = create_loss_vs_samples_plot(cumulative_metrics)

    # Create rolling window plot
    fig_loss_vs_recent = create_loss_vs_recent_checkpoints_plot(cumulative_metrics, window_size)

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

    return (status, fig_loss_vs_samples, fig_loss_vs_recent, eval_loss_fig, eval_dist_fig,
            obs_status, obs_combined_fig)


def calculate_training_info(total_samples, batch_size):
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
