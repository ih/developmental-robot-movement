"""
Training functions for the concat world model explorer.

This module contains all training-related functionality including:
- Single canvas training
- Batch training with DataLoader optimization
- Best model checkpoint management
- Batch size comparison experiments
"""

import os
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

import config
import world_model_utils
from models.autoencoder_concat_predictor import (
    canvas_to_tensor,
    compute_randomized_patch_mask_for_last_slot,
    compute_randomized_patch_mask_for_last_slot_gpu,
    compute_hybrid_loss_on_masked_patches,
)
from models.canvas_dataset import create_canvas_dataloader
from session_explorer_lib import load_frame_image

from . import state
from .utils import format_loss, format_grad_diagnostics, compute_canvas_figsize
from .canvas_ops import build_canvas_from_frame
from .evaluation import evaluate_full_session, evaluate_validation_session
from .visualization import (
    generate_batch_training_update,
    create_loss_vs_samples_plot,
    create_loss_vs_recent_checkpoints_plot,
)


def calculate_validation_loss(batch_size):
    """
    Sample batch from validation set and compute loss without training.

    Args:
        batch_size: Number of samples to evaluate

    Returns:
        Validation loss (float) or None if no validation session loaded or incompatible
    """
    val_state = state.validation_session_state
    if not val_state or "canvas_cache" not in val_state:
        return None

    # Check frame size compatibility
    val_frame_size = val_state.get("detected_frame_size")
    train_frame_size = state.session_state.get("detected_frame_size")
    if val_frame_size and train_frame_size and val_frame_size != train_frame_size:
        # Frame sizes don't match - skip validation to avoid errors
        print(f"[WARNING] Validation frame size {val_frame_size} != training frame size {train_frame_size}, skipping validation")
        return None

    val_canvas_cache = val_state["canvas_cache"]
    val_observations = val_state["observations"]

    # Get all valid indices (frames with enough history for canvas)
    min_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    all_valid_indices = list(range(min_frames - 1, len(val_observations)))

    if not all_valid_indices:
        return None

    # Sample batch indices
    batch_indices = random.choices(all_valid_indices, k=min(batch_size, len(all_valid_indices)))

    # Build batch tensor from cache
    canvases = []
    for idx in batch_indices:
        if idx in val_canvas_cache:
            canvases.append(val_canvas_cache[idx]['canvas'])

    if not canvases:
        return None

    # Stack canvases into tensor
    # canvas_to_tensor returns [1, 3, H, W], so we use cat instead of stack
    canvas_tensors = [canvas_to_tensor(c) for c in canvases]
    canvas_batch = torch.cat(canvas_tensors, dim=0).to(state.device)

    # Get canvas dimensions for mask generation
    canvas_height, canvas_width = canvas_batch.shape[-2:]
    num_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE

    # Generate mask for the batch
    patch_mask = compute_randomized_patch_mask_for_last_slot_gpu(
        batch_size=len(canvases),
        img_size=(canvas_height, canvas_width),
        patch_size=config.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE,
        num_frame_slots=num_frames,
        sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
        mask_ratio_min=config.MASK_RATIO_MIN,
        mask_ratio_max=config.MASK_RATIO_MAX,
        device=state.device,
    )

    # Forward pass (eval mode, no gradients)
    state.world_model.autoencoder.eval()
    with torch.no_grad():
        # Get predictions - forward_with_patch_mask returns (pred_patches, latent)
        pred_patches, _ = state.world_model.autoencoder.forward_with_patch_mask(
            canvas_batch, patch_mask
        )

        # Get ground truth patches from the canvas
        target_patches = state.world_model.autoencoder.patchify(canvas_batch)

        # Select only masked patches for loss computation
        masked_pred = pred_patches[patch_mask]
        masked_target = target_patches[patch_mask]

        # Compute loss
        loss_dict = compute_hybrid_loss_on_masked_patches(
            masked_pred, masked_target,
            focal_beta=config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_BETA,
            focal_alpha=config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_LOSS_ALPHA,
        )
        val_loss = loss_dict['loss_hybrid']

    # Switch back to train mode
    state.world_model.autoencoder.train()

    return val_loss.item()


def log_training_debug_state(world_model, batch_count, samples_seen, loss, enable_wandb=False):
    """
    Log debug information to help diagnose periodic loss spikes.

    Logs:
    - Optimizer momentum/velocity norms (AdamW exp_avg, exp_avg_sq)
    - Model weight norms (first parameter)
    - Scheduler internal state
    - Current batch/sample info

    Args:
        world_model: The world model instance with autoencoder, optimizer, scheduler
        batch_count: Current batch number
        samples_seen: Total samples seen so far
        loss: Current batch loss
        enable_wandb: Whether to log to wandb
    """
    debug_info = {
        'batch_count': batch_count,
        'samples_seen': samples_seen,
        'loss': loss,
    }

    # 1. Optimizer state (momentum and velocity norms)
    try:
        # Get a sample parameter to check optimizer state
        params = list(world_model.autoencoder.parameters())
        if params and params[0] in world_model.ae_optimizer.state:
            opt_state = world_model.ae_optimizer.state[params[0]]
            if 'exp_avg' in opt_state:
                debug_info['momentum_norm'] = opt_state['exp_avg'].norm().item()
            if 'exp_avg_sq' in opt_state:
                debug_info['velocity_norm'] = opt_state['exp_avg_sq'].norm().item()
            if 'step' in opt_state:
                # step can be a tensor in newer PyTorch versions
                step_val = opt_state['step']
                debug_info['optimizer_step'] = step_val.item() if torch.is_tensor(step_val) else step_val
    except Exception as e:
        debug_info['optimizer_error'] = str(e)

    # 2. Model weight norms (first and last parameter)
    try:
        params = list(world_model.autoencoder.parameters())
        if params:
            debug_info['first_param_norm'] = params[0].data.norm().item()
            debug_info['last_param_norm'] = params[-1].data.norm().item()
            # Also track total model norm
            total_norm = sum(p.data.norm().item() ** 2 for p in params) ** 0.5
            debug_info['total_model_norm'] = total_norm
    except Exception as e:
        debug_info['weight_error'] = str(e)

    # 3. Scheduler state
    try:
        debug_info['scheduler_last_epoch'] = world_model.ae_scheduler.last_epoch
        debug_info['scheduler_lr'] = world_model.ae_scheduler.get_last_lr()[0]
    except Exception as e:
        debug_info['scheduler_error'] = str(e)

    # Print to console
    print(f"[DEBUG STATE] batch={batch_count}, samples={samples_seen}, loss={loss:.6f}")
    if 'momentum_norm' in debug_info:
        print(f"  Optimizer: momentum_norm={debug_info['momentum_norm']:.6f}, "
              f"velocity_norm={debug_info.get('velocity_norm', 'N/A')}, "
              f"step={debug_info.get('optimizer_step', 'N/A')}")
    if 'first_param_norm' in debug_info:
        print(f"  Weights: first_param={debug_info['first_param_norm']:.6f}, "
              f"last_param={debug_info['last_param_norm']:.6f}, "
              f"total={debug_info['total_model_norm']:.6f}")
    if 'scheduler_last_epoch' in debug_info:
        print(f"  Scheduler: last_epoch={debug_info['scheduler_last_epoch']}, "
              f"lr={debug_info['scheduler_lr']:.6e}")

    # Log to wandb if enabled
    if enable_wandb:
        try:
            wandb_debug = {
                'debug/batch_count': batch_count,
                'debug/loss': loss,
            }
            if 'momentum_norm' in debug_info:
                wandb_debug['debug/optimizer_momentum_norm'] = debug_info['momentum_norm']
            if 'velocity_norm' in debug_info:
                wandb_debug['debug/optimizer_velocity_norm'] = debug_info['velocity_norm']
            if 'optimizer_step' in debug_info:
                wandb_debug['debug/optimizer_step'] = debug_info['optimizer_step']
            if 'first_param_norm' in debug_info:
                wandb_debug['debug/first_param_norm'] = debug_info['first_param_norm']
            if 'last_param_norm' in debug_info:
                wandb_debug['debug/last_param_norm'] = debug_info['last_param_norm']
            if 'total_model_norm' in debug_info:
                wandb_debug['debug/total_model_norm'] = debug_info['total_model_norm']
            if 'scheduler_last_epoch' in debug_info:
                wandb_debug['debug/scheduler_last_epoch'] = debug_info['scheduler_last_epoch']
            if 'scheduler_lr' in debug_info:
                wandb_debug['debug/scheduler_lr'] = debug_info['scheduler_lr']

            wandb.log(wandb_debug, step=samples_seen)
        except Exception as e:
            print(f"[DEBUG] Failed to log to wandb: {e}")

    return debug_info

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not available. Training metrics will not be logged to Weights & Biases.")


def train_on_single_canvas(frame_idx, num_training_steps):
    """Train autoencoder on a single canvas built from selected frame and its history"""

    if state.world_model is None:
        return "Please load a session first", "", "", None, None, None, None, None

    if not state.session_state.get("observations") or not state.session_state.get("actions"):
        return "No session data available", "", "", None, None, None, None, None

    # Build canvas
    training_canvas, error, start_idx, interleaved = build_canvas_from_frame(frame_idx)
    if training_canvas is None:
        return error, "", "", None, None, None, None, None

    frame_idx = int(frame_idx)

    # Train N times and collect losses
    num_training_steps = int(num_training_steps)
    if num_training_steps <= 0:
        return "Number of training steps must be greater than 0", "", "", None, None, None, None, None

    loss_history = []
    for step in range(num_training_steps):
        loss = state.world_model.train_autoencoder(training_canvas)
        loss_history.append(loss)

    # Store training canvas and mask for visualization
    state.world_model.last_training_canvas = training_canvas
    state.world_model.last_training_loss = loss_history[-1] if loss_history else None

    # Generate visualizations
    status_msg = f"**Trained on canvas from frames {start_idx+1}-{frame_idx+1}**\n\n"
    status_msg += f"Training steps: {num_training_steps}\n\n"
    status_msg += f"Final loss: {format_loss(loss_history[-1] if loss_history else None)}"

    # Training info with gradient diagnostics
    final_loss = loss_history[-1] if loss_history else None
    training_info = f"**Training Loss (Hybrid):** {format_loss(final_loss)}"

    # Add standard loss if available from diagnostics
    if state.world_model.last_grad_diagnostics and 'loss_standard' in state.world_model.last_grad_diagnostics:
        std_loss = state.world_model.last_grad_diagnostics['loss_standard']
        training_info += f"\n\n**Standard Loss (unweighted):** {format_loss(std_loss)}"

    # Gradient diagnostics
    grad_diag_info = format_grad_diagnostics(state.world_model.last_grad_diagnostics)

    # Loss history plot
    fig_loss_history = None
    if len(loss_history) > 1:
        fig_loss_history, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(range(1, len(loss_history) + 1), loss_history, 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss History')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    # Training canvas visualizations
    fig_training_canvas = None
    fig_training_canvas_masked = None
    fig_training_inpainting_full = None
    fig_training_inpainting_composite = None

    if state.world_model.last_training_canvas is not None:
        # Compute dynamic figsize based on canvas dimensions
        canvas = state.world_model.last_training_canvas
        canvas_h, canvas_w = canvas.shape[:2]
        figsize = compute_canvas_figsize(canvas_h, canvas_w)

        # 1. Original training canvas
        fig_training_canvas, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(canvas)
        ax.set_title(f"Training Canvas (Frames {start_idx+1}-{frame_idx+1})")
        ax.axis("off")
        plt.tight_layout()

        # Generate additional visualizations if mask is available
        if state.world_model.last_training_mask is not None:
            # 2. Canvas with mask overlay
            canvas_with_mask = state.world_model.get_canvas_with_mask_overlay(
                canvas,
                state.world_model.last_training_mask
            )
            fig_training_canvas_masked, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(canvas_with_mask)
            ax.set_title("Training Canvas with Mask (Red = Masked Patches)")
            ax.axis("off")
            plt.tight_layout()

            # 3. Full model output
            inpainting_full = state.world_model.get_canvas_inpainting_full_output(
                canvas,
                state.world_model.last_training_mask
            )
            fig_training_inpainting_full, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(inpainting_full)
            ax.set_title("Training Inpainting - Full Model Output")
            ax.axis("off")
            plt.tight_layout()

            # 4. Composite
            inpainting_composite = state.world_model.get_canvas_inpainting_composite(
                canvas,
                state.world_model.last_training_mask
            )
            fig_training_inpainting_composite, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(inpainting_composite)
            ax.set_title("Training Inpainting - Composite")
            ax.axis("off")
            plt.tight_layout()

    return status_msg, training_info, grad_diag_info, fig_loss_history, fig_training_canvas, fig_training_canvas_masked, fig_training_inpainting_full, fig_training_inpainting_composite


def save_best_model_checkpoint(current_loss, samples_seen, world_model, auto_saved_checkpoints, num_best_models_to_keep, is_val_loss=False):
    """
    Save a best model checkpoint and manage the list to keep only N best models.

    Args:
        current_loss: Current loss value
        samples_seen: Number of samples seen at this checkpoint
        world_model: The world model instance
        auto_saved_checkpoints: List of (loss, filepath) tuples for auto-saved checkpoints
        num_best_models_to_keep: Maximum number of best models to keep
        is_val_loss: Whether the loss is validation loss (True) or training loss (False)

    Returns:
        Tuple of (success: bool, message: str, updated_checkpoints_list)
    """
    checkpoint_dir = state.get_checkpoint_dir_for_session(state.session_state["session_dir"])
    loss_type = "val" if is_val_loss else "train"
    session_name = state.session_state.get('session_name', 'unknown')
    checkpoint_name = f"best_model_auto_{session_name}_{samples_seen:08d}_{loss_type}_{current_loss:.6f}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Save the new checkpoint
        checkpoint = {
            'model_state_dict': world_model.autoencoder.state_dict(),
            'optimizer_state_dict': world_model.ae_optimizer.state_dict(),
            'scheduler_state_dict': world_model.ae_scheduler.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'samples_seen': samples_seen,
            'loss': current_loss,
            # Preserve original peak LR for global schedule calculation when resuming
            'original_peak_lr': state.loaded_checkpoint_metadata.get('original_peak_lr')
                                or world_model.ae_optimizer.param_groups[0]['lr'],
            'config': {
                'frame_size': config.AutoencoderConcatPredictorWorldModelConfig.FRAME_SIZE,
                'separator_width': config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
                'canvas_history_size': config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE,
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"[AUTO-SAVE] New best loss {current_loss:.6f} at {samples_seen} samples → saved to {checkpoint_name}")

        # Add to tracking list
        auto_saved_checkpoints.append((current_loss, checkpoint_path))

        # If we exceed the limit, delete the worst checkpoint
        message = f"🏆 **New Best Model Saved!**\n- Loss: {current_loss:.6f}\n- Checkpoint: {checkpoint_name}"

        if len(auto_saved_checkpoints) > num_best_models_to_keep:
            # Sort by loss (ascending) and remove the worst one (highest loss)
            auto_saved_checkpoints.sort(key=lambda x: x[0])
            worst_loss, worst_path = auto_saved_checkpoints.pop()  # Remove last (highest loss)

            # Delete the file
            if os.path.exists(worst_path):
                os.remove(worst_path)
                worst_filename = os.path.basename(worst_path)
                print(f"[AUTO-SAVE] Deleted worse checkpoint: {worst_filename} (loss: {worst_loss:.6f})")
                message += f"\n- Deleted worse checkpoint: {worst_filename} (loss: {worst_loss:.6f})"

        message += f"\n- Keeping {len(auto_saved_checkpoints)}/{num_best_models_to_keep} best models"

        return True, message, auto_saved_checkpoints

    except Exception as e:
        error_msg = f"Failed to save best model: {str(e)}"
        print(f"[AUTO-SAVE ERROR] {error_msg}")
        return False, error_msg, auto_saved_checkpoints


def generate_preflight_summary(total_samples, batch_size, resume_mode, samples_mode, starting_samples,
                                preserve_optimizer, preserve_scheduler, custom_lr, disable_lr_scaling,
                                custom_warmup, lr_min_ratio, resume_warmup_ratio=0.01,
                                sampling_mode="Random (with replacement)",
                                stop_on_divergence=False, divergence_gap=0.001, divergence_ratio=2.5,
                                divergence_patience=3, divergence_min_updates=5):
    """
    Generate a pre-flight summary of the training configuration.

    Returns a markdown string showing what will happen when training starts.
    """
    import math

    # Compute derived values
    total_samples = int(total_samples)
    batch_size = int(batch_size)
    starting_samples = int(starting_samples)
    custom_warmup = int(custom_warmup)
    divergence_patience = int(divergence_patience)
    divergence_min_updates = int(divergence_min_updates)

    # Build summary
    summary = "📋 **Training Configuration Summary**\n\n"

    # Check for divergence mode first - it changes everything
    if stop_on_divergence:
        # Divergence mode - check if validation session is loaded
        has_validation = bool(state.validation_session_state.get("canvas_cache"))

        summary += f"**Mode:** 🎯 Train Until Divergence\n"
        if not has_validation:
            summary += f"- ⚠️ **WARNING: No validation session loaded!**\n"
            summary += f"- This mode requires a validation session to detect divergence.\n"
            summary += f"- Please load a validation session before starting.\n\n"
        else:
            val_name = state.validation_session_state.get("session_name", "unknown")
            summary += f"- Validation session: {val_name}\n"
            summary += f"- Training will run until validation loss diverges from training loss\n"
            summary += f"- `Total Samples` setting is ignored\n\n"

        summary += f"**Divergence Detection:**\n"
        summary += f"- Gap threshold: {divergence_gap} (stop if val - train >= {divergence_gap})\n"
        summary += f"- Ratio threshold: {divergence_ratio} (stop if val / train >= {divergence_ratio})\n"
        summary += f"- Patience: {divergence_patience} consecutive checks\n"
        summary += f"- Min updates before checking: {divergence_min_updates}\n\n"

        # Calculate epochs for divergence mode
        if state.session_state.get("observations"):
            observations = state.session_state["observations"]
            min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
            samples_per_epoch = len(observations) - (min_frames_needed - 1)
            epochs_per_chunk = 100
            samples_per_chunk = samples_per_epoch * epochs_per_chunk
            summary += f"**Training Parameters:**\n"
            summary += f"- Batch size: {batch_size}\n"
            summary += f"- Sampling mode: Epoch-based (forced for divergence mode)\n"
            summary += f"- Samples per epoch: {samples_per_epoch:,}\n"
            summary += f"- Epochs per chunk: {epochs_per_chunk}\n"
            summary += f"- Samples per chunk: {samples_per_chunk:,}\n"
            summary += f"- Training continues indefinitely until divergence detected\n\n"
        else:
            summary += f"**Training Parameters:**\n"
            summary += f"- Batch size: {batch_size}\n"
            summary += f"- Sampling mode: Epoch-based (forced for divergence mode)\n\n"

        # Compute peak LR for display
        if custom_lr > 0:
            peak_lr = custom_lr
            lr_source = "custom override"
            if not disable_lr_scaling:
                peak_lr = peak_lr * batch_size
        elif resume_mode and state.loaded_checkpoint_metadata.get('original_peak_lr'):
            peak_lr = state.loaded_checkpoint_metadata['original_peak_lr']
            lr_source = "from checkpoint (original peak)"
        else:
            base_lr_config = config.AutoencoderConcatPredictorWorldModelConfig.AUTOENCODER_LR
            lr_source = "config default"
            if disable_lr_scaling:
                peak_lr = base_lr_config
            else:
                peak_lr = base_lr_config * batch_size

        global_min_lr = peak_lr * lr_min_ratio

        summary += f"**Learning Rate (ReduceLROnPlateau):**\n"
        summary += f"- Starting LR: {peak_lr:.2e} ({lr_source})\n"
        summary += f"- Min LR: {global_min_lr:.2e}\n"
        summary += f"- Scheduler: ReduceLROnPlateau (stepped at update intervals with val_loss)\n"
        summary += f"- Plateau patience: {divergence_patience * 2} updates\n"
        summary += f"- Reduction factor: 0.5\n"

        return summary

    # Standard mode (not divergence) - original logic
    # Compute actual samples to train
    if resume_mode:
        if samples_mode == "Train additional samples":
            samples_to_train = total_samples
            final_samples_target = starting_samples + total_samples
        else:
            samples_to_train = max(0, total_samples - starting_samples)
            final_samples_target = total_samples
    else:
        samples_to_train = total_samples
        starting_samples = 0
        final_samples_target = total_samples

    num_gradient_updates = math.ceil(samples_to_train / batch_size)

    # Compute imaginary total for global schedule
    imaginary_total = starting_samples + samples_to_train

    # Compute peak LR (same logic as training)
    if custom_lr > 0:
        peak_lr = custom_lr
        lr_source = "custom override"
        if not disable_lr_scaling:
            peak_lr = peak_lr * batch_size
    elif resume_mode and state.loaded_checkpoint_metadata.get('original_peak_lr'):
        peak_lr = state.loaded_checkpoint_metadata['original_peak_lr']
        lr_source = "from checkpoint (original peak)"
    else:
        base_lr_config = config.AutoencoderConcatPredictorWorldModelConfig.AUTOENCODER_LR
        lr_source = "config default"
        if disable_lr_scaling:
            peak_lr = base_lr_config
        else:
            peak_lr = base_lr_config * batch_size

    # Compute global min LR
    global_min_lr = peak_lr * lr_min_ratio

    # Compute starting LR based on global schedule
    if resume_mode and starting_samples > 0:
        global_progress = starting_samples / imaginary_total
        target_lr = global_min_lr + 0.5 * (peak_lr - global_min_lr) * (1 + math.cos(math.pi * global_progress))
        checkpoint_lr = state.loaded_checkpoint_metadata.get('learning_rate') or target_lr
    else:
        global_progress = 0.0
        target_lr = peak_lr
        checkpoint_lr = None

    # Compute warmup
    base_warmup = config.AutoencoderConcatPredictorWorldModelConfig.WARMUP_STEPS
    if custom_warmup == 0:
        warmup_steps = 0
        warmup_note = "disabled"
    elif custom_warmup > 0:
        warmup_steps = custom_warmup
        warmup_note = "custom"
    else:
        warmup_steps = max(1, int(base_warmup / batch_size))
        warmup_note = f"scaled (base {base_warmup} / batch_size)"

    # Resume warmup steps (proportional to session steps, capped at 25% of total)
    resume_warmup_steps = int(num_gradient_updates * resume_warmup_ratio) if resume_mode and starting_samples > 0 else 0
    resume_warmup_steps = min(resume_warmup_steps, num_gradient_updates // 4)  # Cap at 25% to leave room for decay

    # Mode
    if resume_mode:
        summary += f"**Mode:** 🔄 Resuming from checkpoint\n"
        summary += f"- Starting samples: {starting_samples:,}\n"
        if samples_mode == "Train additional samples":
            summary += f"- Additional samples: {samples_to_train:,}\n"
        else:
            summary += f"- Target total: {total_samples:,} (training {samples_to_train:,} more)\n"
        summary += f"- Final samples: {final_samples_target:,}\n\n"
    else:
        summary += f"**Mode:** 🆕 Fresh training\n"
        summary += f"- Total samples: {total_samples:,}\n\n"

    # Training parameters
    summary += f"**Training Parameters:**\n"
    summary += f"- Batch size: {batch_size}\n"
    summary += f"- Gradient updates: {num_gradient_updates:,}\n"
    summary += f"- Sampling mode: {sampling_mode}\n"

    # For epoch-based mode, show epoch breakdown if we have session info
    if sampling_mode == "Epoch-based (shuffle each epoch)" and state.session_state.get("observations"):
        observations = state.session_state["observations"]
        min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
        num_valid = len(observations) - (min_frames_needed - 1)
        if num_valid > 0:
            num_epochs = samples_to_train // num_valid
            remainder = samples_to_train % num_valid
            summary += f"  - {num_epochs} complete epoch(s) + {remainder} remainder samples\n"
            summary += f"  - {num_valid} unique frames per epoch\n"
    summary += "\n"

    # Learning rate (global schedule)
    summary += f"**Learning Rate (Global Schedule):**\n"
    summary += f"- Peak LR: {peak_lr:.2e} ({lr_source})\n"
    summary += f"- Min LR: {global_min_lr:.2e} (ratio: {lr_min_ratio})\n"

    if resume_mode and starting_samples > 0:
        summary += f"- Global progress: {global_progress:.1%} ({starting_samples:,} / {imaginary_total:,})\n"
        summary += f"- Target LR: {target_lr:.2e} (from cosine at {global_progress:.1%})\n"
        if checkpoint_lr is not None:
            summary += f"- Checkpoint LR: {checkpoint_lr:.2e}\n"
        if resume_warmup_steps > 0:
            summary += f"- Resume warmup: {resume_warmup_steps} steps ({resume_warmup_ratio:.0%} of {num_gradient_updates} steps)\n"
            summary += f"  ({checkpoint_lr:.2e} → {target_lr:.2e})\n"
    else:
        summary += f"- Starting LR: {target_lr:.2e}\n"
    summary += "\n"

    # Warmup (for fresh training)
    if not resume_mode or starting_samples == 0:
        summary += f"**Warmup:**\n"
        summary += f"- Warmup steps: {warmup_steps} ({warmup_note})\n"
        if warmup_steps > 0:
            summary += f"- Warmup samples: {warmup_steps * batch_size:,}\n"
        summary += "\n"

    # State preservation (only relevant for resume mode)
    if resume_mode:
        summary += f"**State Preservation:**\n"
        summary += f"- Optimizer state: {'✅ Preserved' if preserve_optimizer else '🔄 Reset'}\n"
        summary += f"- Scheduler state: {'✅ Preserved' if preserve_scheduler else '🔄 Reset'}\n"

        # Get current LR from optimizer if available
        if state.world_model is not None:
            current_optimizer_lr = state.world_model.ae_optimizer.param_groups[0]['lr']
            summary += f"- Current optimizer LR: {current_optimizer_lr:.2e}\n"

        if state.loaded_checkpoint_metadata.get('loss') is not None:
            summary += f"- Checkpoint loss: {state.loaded_checkpoint_metadata['loss']:.6f}\n"

    return summary


def run_world_model_batch(total_samples, batch_size, current_observation_idx, update_interval=100,
                          window_size=10, num_random_obs=5, num_best_models_to_keep=3,
                          enable_wandb=False, wandb_run_name="",
                          # Resume mode parameters
                          resume_mode=False, samples_mode="Train additional samples",
                          starting_samples=0,
                          preserve_optimizer=True, preserve_scheduler=True,
                          # Learning rate parameters
                          custom_lr=0, disable_lr_scaling=False,
                          custom_warmup=-1, lr_min_ratio=0.01, resume_warmup_ratio=0.01,
                          # Sampling mode
                          sampling_mode="Random (with replacement)",
                          # Divergence-based early stopping parameters
                          stop_on_divergence=False, divergence_gap=0.001, divergence_ratio=2.5,
                          divergence_patience=3, divergence_min_updates=5):
    """
    Run batch training with periodic full-session evaluation.

    Args:
        total_samples: Total number of training samples (or additional samples if resume_mode).
                      Ignored if stop_on_divergence=True.
        batch_size: Batch size for training
        current_observation_idx: Currently selected observation to refresh during updates
        update_interval: Evaluate every N samples (default: 100)
        window_size: Number of recent checkpoints for rolling window plot (default: 10)
        num_random_obs: Number of random observations to visualize (default: 5)
        num_best_models_to_keep: Number of best model checkpoints to keep (default: 3)
        enable_wandb: Whether to log metrics to Weights & Biases (default: False)
        wandb_run_name: Optional custom name for wandb run (default: "")
        resume_mode: Whether to resume from loaded checkpoint (default: False)
        samples_mode: "Train additional samples" or "Train to total samples" (default: "Train additional samples")
        starting_samples: Starting samples seen count for resume (default: 0)
        preserve_optimizer: Keep optimizer state from checkpoint (default: True)
        preserve_scheduler: Keep scheduler state from checkpoint (default: True)
        custom_lr: Override base learning rate, 0 = use config default (default: 0)
        disable_lr_scaling: Use exact LR instead of scaling by batch size (default: False)
        custom_warmup: Override warmup steps, -1 = scaled default, 0 = none (default: -1)
        lr_min_ratio: Minimum LR as ratio of base LR (default: 0.01)
        resume_warmup_ratio: Warmup steps as ratio of session gradient updates when resuming (default: 0.01)
        sampling_mode: "Random (with replacement)" or "Epoch-based (shuffle each epoch)" (default: "Random (with replacement)")
        stop_on_divergence: Train until validation loss diverges from training loss (default: False).
                           When True, total_samples is ignored and requires a validation session.
        divergence_gap: Stop if (val_loss - train_loss) >= this value (default: 0.001)
        divergence_ratio: Stop if (val_loss / train_loss) >= this ratio (default: 2.5)
        divergence_patience: Consecutive divergence checks before stopping (default: 3)
        divergence_min_updates: Minimum update intervals before checking divergence (default: 5)

    Yields:
        Tuple of (status, loss_vs_samples_plot, loss_vs_recent_plot, eval_loss_plot, eval_dist_plot,
                  obs_status, obs_combined_fig)
    """

    print(f"[DEBUG] run_world_model_batch called with: total_samples={total_samples}, batch_size={batch_size}, "
          f"update_interval={update_interval}, num_best_models_to_keep={num_best_models_to_keep}, "
          f"resume_mode={resume_mode}, preserve_optimizer={preserve_optimizer}, custom_lr={custom_lr}, "
          f"sampling_mode={sampling_mode}, stop_on_divergence={stop_on_divergence}")

    # Validation
    if state.world_model is None:
        yield "Please load a session first", None, None, None, None, "", None
        return

    if not state.session_state.get("observations") or not state.session_state.get("actions"):
        yield "No session data available", None, None, None, None, "", None
        return

    total_samples = int(total_samples)
    batch_size = int(batch_size)
    current_observation_idx = int(current_observation_idx)
    update_interval = int(update_interval)
    starting_samples = int(starting_samples)
    custom_warmup = int(custom_warmup)
    divergence_patience = int(divergence_patience)
    divergence_min_updates = int(divergence_min_updates)

    # Divergence mode validation
    if stop_on_divergence:
        if not state.validation_session_state.get("canvas_cache"):
            yield "Train-until-divergence mode requires a validation session. Please load one first.", None, None, None, None, "", None
            return
        # In divergence mode, we train until divergence is detected
        # We can't pre-generate infinite samples, so we use epoch-based mode with many epochs
        # Force epoch-based mode for divergence training (more memory efficient for long runs)
        sampling_mode = "Epoch-based (shuffle each epoch)"
        print(f"[DEBUG] Divergence mode enabled: forcing epoch-based sampling")

    if total_samples <= 0:
        yield "Total samples must be greater than 0", None, None, None, None, "", None
        return

    if batch_size <= 0:
        yield "Batch size must be greater than 0", None, None, None, None, "", None
        return

    if update_interval <= 0:
        yield "Update interval must be greater than 0", None, None, None, None, "", None
        return

    observations = state.session_state["observations"]
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    max_samples_per_epoch = len(observations) - (min_frames_needed - 1)

    if max_samples_per_epoch <= 0:
        yield f"Session too small: need at least {min_frames_needed} observations", None, None, None, None, "", None
        return

    # Check if canvas cache exists
    canvas_cache = state.session_state.get("canvas_cache", {})
    if not canvas_cache:
        yield "Canvas cache not found. Please reload the session.", None, None, None, None, "", None
        return

    # For divergence mode, we'll use chunked training - generate epochs in batches
    # and keep looping until divergence is detected
    divergence_epochs_per_chunk = 100  # Generate 100 epochs at a time
    if stop_on_divergence:
        # Start with one chunk, we'll regenerate as needed
        total_samples = max_samples_per_epoch * divergence_epochs_per_chunk
        print(f"[DEBUG] Divergence mode: starting with {divergence_epochs_per_chunk} epochs ({total_samples:,} samples), will regenerate as needed")

    # Handle resume mode: compute actual samples to train
    if resume_mode:
        if samples_mode == "Train additional samples":
            # total_samples is the number of ADDITIONAL samples to train
            samples_to_train = total_samples
            final_samples_target = starting_samples + total_samples
        else:
            # total_samples is the TOTAL target, train the difference
            samples_to_train = max(0, total_samples - starting_samples)
            final_samples_target = total_samples
            if samples_to_train <= 0:
                yield f"Already at {starting_samples:,} samples, target is {total_samples:,}. Nothing to train.", None, None, None, None, "", None
                return
        print(f"[DEBUG] Resume mode: starting_samples={starting_samples}, samples_to_train={samples_to_train}, final_target={final_samples_target}")
    else:
        samples_to_train = total_samples
        starting_samples = 0  # Fresh start
        final_samples_target = total_samples

    # Sample indices based on sampling mode
    all_valid_indices = list(range(min_frames_needed - 1, len(observations)))
    chunk_number = 0  # For divergence mode - track which chunk we're on

    if sampling_mode == "Random (with replacement)":
        # Random sampling with replacement (allows looping through session)
        sampled_indices = random.choices(all_valid_indices, k=samples_to_train)
        print(f"[DEBUG] Random sampling: {len(sampled_indices)} samples from {len(all_valid_indices)} valid observations")
    else:
        # Epoch-based: shuffle each epoch, allow partial final epoch
        num_epochs = samples_to_train // len(all_valid_indices)
        remainder = samples_to_train % len(all_valid_indices)

        sampled_indices = []
        for _ in range(num_epochs):
            epoch_indices = all_valid_indices.copy()
            random.shuffle(epoch_indices)
            sampled_indices.extend(epoch_indices)

        if remainder > 0:
            final_epoch = all_valid_indices.copy()
            random.shuffle(final_epoch)
            sampled_indices.extend(final_epoch[:remainder])

        print(f"[DEBUG] Epoch-based sampling: {num_epochs} complete epochs + {remainder} remainder = {len(sampled_indices)} samples")

    print(f"[DEBUG] Expected batches: {len(sampled_indices) // batch_size} (with batch_size={batch_size})")

    # Store epoch info for debug prints (only relevant for epoch-based mode)
    epoch_mode = (sampling_mode != "Random (with replacement)")
    samples_per_epoch = len(all_valid_indices) if epoch_mode else 0
    if epoch_mode:
        total_epochs = num_epochs + (1 if remainder > 0 else 0)
    else:
        total_epochs = 0

    # Compute learning rate using global schedule for consistent multi-session training
    import math
    num_gradient_updates = math.ceil(samples_to_train / batch_size)
    base_batch_size = 1  # Base LR is defined for batch size 1

    # Compute imaginary total samples for global schedule
    imaginary_total = starting_samples + samples_to_train

    # Determine peak LR (for fresh training or from checkpoint's original peak)
    if custom_lr > 0:
        # User override - use specified LR as peak
        peak_lr = custom_lr
        lr_source = "custom override"
        # Apply scaling unless disabled
        if not disable_lr_scaling:
            peak_lr = peak_lr * (batch_size / base_batch_size)
        print(f"[DEBUG] Using custom LR override: peak_lr={peak_lr:.6e}")
    elif resume_mode and state.loaded_checkpoint_metadata.get('original_peak_lr'):
        # Resume: use checkpoint's original peak LR for global schedule calculation
        peak_lr = state.loaded_checkpoint_metadata['original_peak_lr']
        lr_source = "from checkpoint (original peak)"
        print(f"[DEBUG] Using checkpoint's original peak LR: {peak_lr:.6e}")
    else:
        # Fresh training: compute peak LR from config
        base_lr = config.AutoencoderConcatPredictorWorldModelConfig.AUTOENCODER_LR
        lr_source = "config default"
        if disable_lr_scaling:
            peak_lr = base_lr
        else:
            peak_lr = base_lr * (batch_size / base_batch_size)
        print(f"[DEBUG] Using config default: peak_lr={peak_lr:.6e}")

    # Compute global min LR (same floor for all sessions)
    global_min_lr = peak_lr * lr_min_ratio

    # Compute starting LR based on global schedule position
    if resume_mode and starting_samples > 0:
        # Resume: compute where we are in the global cosine schedule
        global_progress = starting_samples / imaginary_total
        # Cosine annealing formula: lr = min + 0.5 * (max - min) * (1 + cos(π * progress))
        target_lr = global_min_lr + 0.5 * (peak_lr - global_min_lr) * (1 + math.cos(math.pi * global_progress))
        print(f"[DEBUG] Global schedule: progress={global_progress:.2%} ({starting_samples}/{imaginary_total}), target_lr={target_lr:.6e}")

        # Get checkpoint's ending LR for warmup start
        checkpoint_lr = state.loaded_checkpoint_metadata.get('learning_rate') or target_lr
    else:
        # Fresh training: start at peak LR
        target_lr = peak_lr
        checkpoint_lr = None  # Not resuming, no checkpoint LR

    # Compute warmup steps
    base_warmup_steps = config.AutoencoderConcatPredictorWorldModelConfig.WARMUP_STEPS
    if custom_warmup == 0:
        # Warmup disabled
        scaled_warmup_steps = 0
        print(f"[DEBUG] Warmup disabled (custom_warmup=0)")
    elif custom_warmup > 0:
        # Use exact custom warmup steps
        scaled_warmup_steps = custom_warmup
        print(f"[DEBUG] Custom warmup: {scaled_warmup_steps} steps")
    else:
        # Default: scale warmup inversely with batch size
        scaled_warmup_steps = max(1, int(base_warmup_steps / batch_size))
        print(f"[DEBUG] Warmup scaling: base_warmup={base_warmup_steps} steps -> scaled_warmup={scaled_warmup_steps} steps (BS={batch_size})")

    # Resume warmup: additional warmup steps to ramp from checkpoint LR to target LR (proportional to session steps)
    resume_warmup_steps = int(num_gradient_updates * resume_warmup_ratio) if resume_mode and starting_samples > 0 else 0
    resume_warmup_steps = min(resume_warmup_steps, num_gradient_updates // 4)  # Cap at 25% to leave room for decay

    # Handle optimizer state
    import torch.optim as optim
    if resume_mode and preserve_optimizer:
        # Keep existing optimizer, set LR to starting point
        # For resume with warmup, start at checkpoint_lr; otherwise use target_lr
        starting_lr = checkpoint_lr if resume_warmup_steps > 0 else target_lr
        for param_group in state.world_model.ae_optimizer.param_groups:
            param_group['lr'] = starting_lr
        print(f"[DEBUG] Preserved optimizer state, set LR to {starting_lr:.6e}")
    else:
        # Recreate optimizer - start at checkpoint_lr if resuming with warmup, else target_lr
        # Use create_param_groups for consistency with world model initialization (2 param groups)
        starting_lr = checkpoint_lr if (resume_mode and resume_warmup_steps > 0) else target_lr
        param_groups = world_model_utils.create_param_groups(
            state.world_model.autoencoder,
            config.AutoencoderConcatPredictorWorldModelConfig.WEIGHT_DECAY
        )
        state.world_model.ae_optimizer = optim.AdamW(
            param_groups,
            lr=starting_lr,
        )
        print(f"[DEBUG] Created new optimizer with LR={starting_lr:.6e} (2 param groups: decay/no-decay)")

    # Create scheduler
    # Note: When resuming with global schedule (starting_samples > 0), we ALWAYS create a new scheduler
    # because the old scheduler doesn't know about the new total samples. The preserve_scheduler
    # option only applies when NOT using global schedule (e.g., continuing with exact same schedule).
    use_plateau_scheduler = False  # Track which scheduler type we're using

    if stop_on_divergence:
        # Divergence mode: use ReduceLROnPlateau since total steps are unknown
        # This scheduler adapts based on validation loss
        state.world_model.ae_scheduler = world_model_utils.create_reduce_on_plateau_scheduler(
            state.world_model.ae_optimizer,
            patience=divergence_patience * 2,  # More patient than divergence check
            factor=0.5,
            min_lr=global_min_lr,
        )
        use_plateau_scheduler = True
        print(f"[DEBUG] Created ReduceLROnPlateau scheduler (divergence mode): patience={divergence_patience * 2}, min_lr={global_min_lr:.6e}")
    elif resume_mode and preserve_scheduler and starting_samples == 0:
        # Keep existing scheduler only if not using global schedule
        print(f"[DEBUG] Preserved scheduler state (step {state.world_model.ae_scheduler.last_epoch})")
    elif resume_mode and starting_samples > 0:
        # Resume with warmup: use special resume scheduler
        # Reset initial_lr to current lr before creating new scheduler
        # (prevents LambdaLR from reusing checkpoint's initial_lr which was set by previous scheduler)
        for param_group in state.world_model.ae_optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        print(f"[DEBUG] Reset initial_lr to {state.world_model.ae_optimizer.param_groups[0]['lr']:.6e} for all param groups")
        state.world_model.ae_scheduler = world_model_utils.create_resume_scheduler(
            optimizer=state.world_model.ae_optimizer,
            warmup_from_lr=checkpoint_lr,
            warmup_to_lr=target_lr,
            warmup_steps=resume_warmup_steps,
            decay_to_lr=global_min_lr,
            total_steps=num_gradient_updates,
        )
        print(f"[DEBUG] Created resume scheduler: warmup {checkpoint_lr:.6e} -> {target_lr:.6e} ({resume_warmup_steps} steps), "
              f"then decay to {global_min_lr:.6e}")
        # Debug: check what LR the scheduler set after creation
        print(f"[DEBUG] After scheduler creation, optimizer LR = {state.world_model.ae_optimizer.param_groups[0]['lr']:.6e}")
    else:
        # Fresh training: use regular warmup + cosine scheduler
        state.world_model.ae_scheduler = world_model_utils.create_warmup_cosine_scheduler(
            state.world_model.ae_optimizer,
            warmup_steps=scaled_warmup_steps,
            total_steps=num_gradient_updates,
            lr_min=global_min_lr,  # Use absolute min, not ratio
        )
        print(f"[DEBUG] Created new scheduler: {num_gradient_updates} steps (warmup: {scaled_warmup_steps}), "
              f"LR range {target_lr:.6e} -> {global_min_lr:.6e}")

    # Store original peak LR in metadata for future checkpoints (if not already set)
    if not state.loaded_checkpoint_metadata.get('original_peak_lr'):
        state.loaded_checkpoint_metadata['original_peak_lr'] = peak_lr
        print(f"[DEBUG] Stored original_peak_lr={peak_lr:.6e} in metadata")

    # Determine optimal number of workers
    import platform
    if platform.system() == 'Windows':
        num_workers = 0  # Single-process on Windows
    else:
        num_workers = 4  # Multiple workers on Linux

    # Create DataLoader
    use_stream_pipelining = (state.device == 'cuda' and torch.cuda.is_available())

    try:
        dataloader = create_canvas_dataloader(
            canvas_cache=canvas_cache,
            frame_indices=sampled_indices,
            batch_size=batch_size,
            config=config.AutoencoderConcatPredictorWorldModelConfig,
            device=state.device,
            num_workers=num_workers,
            shuffle=False,  # Already sampled with replacement
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            transfer_to_device=(not use_stream_pipelining),
        )
    except Exception as e:
        yield f"Error creating DataLoader: {str(e)}", None, None, None, None, "", None
        return

    # Initialize metrics with resume offset
    cumulative_metrics = {
        'samples_seen': [],
        'loss_at_sample': [],
        'val_loss_at_sample': [],  # Validation loss (if validation session loaded)
    }

    # Check if validation session is loaded
    has_validation = bool(state.validation_session_state.get("canvas_cache"))
    if has_validation:
        val_name = state.validation_session_state.get("session_name", "unknown")
        print(f"[DEBUG] Validation session loaded: {val_name}")
    samples_seen = starting_samples  # Start from checkpoint offset if resuming
    UPDATE_INTERVAL = int(update_interval)  # Evaluate every N samples
    last_eval_samples = starting_samples

    # Track best loss for automatic checkpoint saving
    # Always start fresh - don't inherit previous best_loss even in resume mode
    best_loss = float('inf')

    # Track auto-saved checkpoints for this training run (list of tuples: (loss, filepath))
    auto_saved_checkpoints = []

    # Divergence tracking for early stopping
    divergence_count = 0
    update_count = 0
    stop_early = False
    stop_reason = ""

    # Track total training time
    training_start_time = time.time()

    # Initialize wandb if enabled
    if enable_wandb:
        if not WANDB_AVAILABLE:
            print("[WANDB WARNING] wandb package not installed. Continuing without wandb logging.")
            enable_wandb = False
        else:
            try:
                wandb.init(
                    project="developmental-robot-movement",
                    name=wandb_run_name if wandb_run_name else None,  # auto-generate if empty
                    config={
                        "total_samples": samples_to_train,
                        "final_samples_target": final_samples_target,
                        "batch_size": batch_size,
                        "peak_lr": peak_lr,
                        "target_lr": target_lr,
                        "global_min_lr": global_min_lr,
                        "warmup_steps": scaled_warmup_steps,
                        "total_gradient_updates": num_gradient_updates,
                        "update_interval": update_interval,
                        "session_name": state.session_state.get("session_name", "unknown"),
                        "frame_size": config.AutoencoderConcatPredictorWorldModelConfig.FRAME_SIZE,
                        "separator_width": config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
                        "canvas_history_size": config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE,
                        "patch_size": config.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE,
                        "weight_decay": config.AutoencoderConcatPredictorWorldModelConfig.WEIGHT_DECAY,
                        "lr_min_ratio": lr_min_ratio,
                        "focal_beta": config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_BETA,
                        "focal_loss_alpha": config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_LOSS_ALPHA,
                        # Resume mode info
                        "resume_mode": resume_mode,
                        "starting_samples": starting_samples,
                        "preserve_optimizer": preserve_optimizer,
                        "preserve_scheduler": preserve_scheduler,
                        "custom_lr": custom_lr,
                        "disable_lr_scaling": disable_lr_scaling,
                    }
                )
                print(f"[WANDB] Initialized run: {wandb.run.name}")
            except Exception as e:
                print(f"[WANDB WARNING] Failed to initialize wandb: {str(e)}")
                print("[WANDB WARNING] Continuing training without wandb logging")
                enable_wandb = False  # Disable wandb for rest of training

    # Training loop - outer loop for divergence mode (regenerates epochs as needed)
    state.world_model.autoencoder.train()

    batch_count = 0
    try:
        while True:  # Outer loop for divergence mode - will break when done or stopped early
          if use_stream_pipelining:
            # CUDA stream pipelining for optimal performance
            transfer_stream = torch.cuda.Stream()
            dataloader_iter = iter(dataloader)

            # Prefetch first batch
            try:
                next_canvas_cpu, next_mask_cpu, _ = next(dataloader_iter)
                with torch.cuda.stream(transfer_stream):
                    next_canvas = next_canvas_cpu.to(state.device, non_blocking=True)
                    next_mask = next_mask_cpu.to(state.device, non_blocking=True)
                print(f"[DEBUG] Prefetched first batch successfully")
            except StopIteration:
                next_canvas = None
                print(f"[DEBUG] DataLoader empty - no batches available!")

            print(f"[DEBUG] Starting training loop: next_canvas is {'not None' if next_canvas is not None else 'None'}, samples_seen={samples_seen}, final_target={final_samples_target}")
            while next_canvas is not None and samples_seen < final_samples_target:
                batch_count += 1
                # Wait for transfer to complete
                torch.cuda.current_stream().wait_stream(transfer_stream)
                canvas_tensor = next_canvas
                patch_mask = next_mask

                # Prefetch next batch in parallel with training
                with torch.cuda.stream(transfer_stream):
                    try:
                        next_canvas_cpu, next_mask_cpu, _ = next(dataloader_iter)
                        next_canvas = next_canvas_cpu.to(state.device, non_blocking=True)
                        next_mask = next_mask_cpu.to(state.device, non_blocking=True)
                    except StopIteration:
                        next_canvas = None

                # Train on current batch
                loss, grad_diagnostics = state.world_model.autoencoder.train_on_canvas(
                    canvas_tensor, patch_mask, state.world_model.ae_optimizer
                )
                # Step scheduler every batch only for non-plateau schedulers
                # Plateau scheduler is stepped at update intervals with val_loss
                if not use_plateau_scheduler:
                    state.world_model.ae_scheduler.step()
                samples_seen += canvas_tensor.shape[0]

                # Log per-batch metrics to wandb
                if enable_wandb:
                    try:
                        wandb.log({
                            "train/loss_hybrid": loss,
                            "train/samples_seen": samples_seen,
                            "train/learning_rate": state.world_model.ae_optimizer.param_groups[0]['lr'],
                        }, step=samples_seen)
                    except Exception as e:
                        print(f"[WANDB WARNING] Failed to log per-batch metrics: {str(e)}")

                if batch_count <= 5 or batch_count % 10 == 0:
                    current_lr = state.world_model.ae_optimizer.param_groups[0]['lr']
                    if epoch_mode and samples_per_epoch > 0:
                        current_epoch = (samples_seen - starting_samples) // samples_per_epoch + 1
                        print(f"[DEBUG] Batch {batch_count}: epoch={current_epoch}/{total_epochs}, samples_seen={samples_seen}/{final_samples_target}, loss={loss:.6f}, lr={current_lr:.6e}")
                    else:
                        print(f"[DEBUG] Batch {batch_count}: samples_seen={samples_seen}/{final_samples_target}, loss={loss:.6f}, lr={current_lr:.6e}")

                # Periodic progress update (no evaluation)
                if samples_seen - last_eval_samples >= UPDATE_INTERVAL:
                    last_eval_samples = samples_seen
                    elapsed_time = time.time() - training_start_time

                    # Track training loss for progress plots
                    cumulative_metrics['samples_seen'].append(samples_seen)
                    cumulative_metrics['loss_at_sample'].append(loss)

                    # Calculate validation loss if validation session is loaded
                    val_loss = None
                    if has_validation:
                        val_loss = calculate_validation_loss(batch_size)
                        cumulative_metrics['val_loss_at_sample'].append(val_loss)
                        if val_loss is not None:
                            print(f"[DEBUG] Validation loss: {val_loss:.6f}")

                    # Step plateau scheduler with validation loss (at update intervals, not every batch)
                    if use_plateau_scheduler and val_loss is not None:
                        state.world_model.ae_scheduler.step(val_loss)

                    # Divergence checking for early stopping
                    update_count += 1
                    if stop_on_divergence and val_loss is not None:
                        if update_count >= divergence_min_updates:
                            gap = val_loss - loss
                            ratio = val_loss / max(loss, 1e-8)
                            diverged = (gap >= divergence_gap) or (ratio >= divergence_ratio)
                            if diverged:
                                divergence_count += 1
                                print(f"[DIVERGENCE] Check {divergence_count}/{divergence_patience}: "
                                      f"gap={gap:.6f} (thresh={divergence_gap}), ratio={ratio:.2f} (thresh={divergence_ratio})")
                            else:
                                divergence_count = 0  # Reset on non-divergent check

                            if divergence_count >= divergence_patience:
                                stop_early = True
                                stop_reason = (f"Divergence detected after {update_count} updates: "
                                              f"val={val_loss:.6f} vs train={loss:.6f} "
                                              f"(gap={gap:.6f}, ratio={ratio:.2f})")
                                print(f"[DIVERGENCE] {stop_reason}")

                    # Debug logging to diagnose periodic loss spikes
                    log_training_debug_state(state.world_model, batch_count, samples_seen, loss, enable_wandb)

                    # Save best model if loss improved
                    # Use validation loss if available, otherwise use training loss
                    checkpoint_loss = val_loss if (val_loss is not None) else loss
                    if checkpoint_loss < best_loss:
                        best_loss = checkpoint_loss
                        success, save_msg, auto_saved_checkpoints = save_best_model_checkpoint(
                            checkpoint_loss, samples_seen, state.world_model, auto_saved_checkpoints, num_best_models_to_keep,
                            is_val_loss=(val_loss is not None)
                        )
                        if success:
                            print(f"[AUTO-SAVE] {save_msg}")

                    # Log gradient diagnostics and additional metrics to wandb
                    if enable_wandb and grad_diagnostics:
                        try:
                            log_dict = {
                                "gradients/head_weight_norm": grad_diagnostics.get('head_weight_norm'),
                                "gradients/head_bias_norm": grad_diagnostics.get('head_bias_norm'),
                                "gradients/mask_token_norm": grad_diagnostics.get('mask_token_norm'),
                                "gradients/qkv_weight_norm": grad_diagnostics.get('qkv_weight_norm'),
                                "loss/plain": grad_diagnostics.get('loss_plain'),
                                "loss/focal": grad_diagnostics.get('loss_focal'),
                                "loss/standard": grad_diagnostics.get('loss_standard'),
                                "loss/nonblack": grad_diagnostics.get('loss_nonblack'),
                                "loss/black_baseline": grad_diagnostics.get('black_baseline'),
                                "loss/frac_nonblack": grad_diagnostics.get('frac_nonblack'),
                                "focal/weight_mean": grad_diagnostics.get('focal_weight_mean'),
                                "focal/weight_max": grad_diagnostics.get('focal_weight_max'),
                                "train/best_loss": best_loss,
                                "train/elapsed_time": elapsed_time,
                            }
                            # Add validation loss if available
                            if val_loss is not None:
                                log_dict["val/loss"] = val_loss
                            wandb.log(log_dict, step=samples_seen)
                        except Exception as e:
                            print(f"[WANDB WARNING] Failed to log gradient diagnostics: {str(e)}")

                    # Status message
                    current_lr = state.world_model.ae_optimizer.param_groups[0]['lr']
                    status_msg = f"📊 **Training Progress**\n\n"
                    status_msg += f"- Samples: {samples_seen:,} / {final_samples_target:,}\n"
                    if resume_mode:
                        status_msg += f"- Training: {samples_seen - starting_samples:,} / {samples_to_train:,} additional\n"
                    if stop_on_divergence:
                        status_msg += f"- Mode: Train until divergence\n"
                        status_msg += f"- Divergence checks: {divergence_count}/{divergence_patience}\n"
                    status_msg += f"- Current batch loss: {loss:.6f}\n"
                    status_msg += f"- Best loss: {best_loss:.6f}\n"
                    status_msg += f"- Learning rate: {current_lr:.6e}\n"

                    # Yield update (no evaluation plots)
                    print(f"[DEBUG] Yielding update at {samples_seen} samples (batch {batch_count})")
                    update_result = generate_batch_training_update(
                        samples_seen, final_samples_target, cumulative_metrics,
                        status_msg, None, None, current_observation_idx,
                        window_size, num_random_obs, completed=False, elapsed_time=elapsed_time
                    )

                    yield update_result
                    print(f"[DEBUG] Resumed after yield, continuing training...")

                    # Check for early stop after yield
                    if stop_early:
                        break

            # Check if we exited due to early stop
            if stop_early:
                print(f"[DEBUG] Training stopped early: {stop_reason}")

            print(f"[DEBUG] Exited training loop: batches_processed={batch_count}, samples_seen={samples_seen}, final_target={final_samples_target}")
            print(f"[DEBUG] Exit reason: next_canvas is {'None' if next_canvas is None else 'not None'}, samples_seen >= final_target: {samples_seen >= final_samples_target}")

          else:
            # Fallback: no stream pipelining
            print(f"[DEBUG] Using fallback mode (no stream pipelining)")
            for canvas_tensor, patch_mask, _ in dataloader:
                batch_count += 1
                if samples_seen >= final_samples_target:
                    print(f"[DEBUG] Breaking: samples_seen ({samples_seen}) >= final_target ({final_samples_target})")
                    break

                # Train on batch
                loss, grad_diagnostics = state.world_model.autoencoder.train_on_canvas(
                    canvas_tensor, patch_mask, state.world_model.ae_optimizer
                )
                # Step scheduler every batch only for non-plateau schedulers
                # Plateau scheduler is stepped at update intervals with val_loss
                if not use_plateau_scheduler:
                    state.world_model.ae_scheduler.step()
                samples_seen += canvas_tensor.shape[0]

                # Log per-batch metrics to wandb
                if enable_wandb:
                    try:
                        wandb.log({
                            "train/loss_hybrid": loss,
                            "train/samples_seen": samples_seen,
                            "train/learning_rate": state.world_model.ae_optimizer.param_groups[0]['lr'],
                        }, step=samples_seen)
                    except Exception as e:
                        print(f"[WANDB WARNING] Failed to log per-batch metrics: {str(e)}")

                # Check for NaN loss
                if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                    error_msg = f"❌ Training failed at batch {batch_count}: Loss became NaN/Inf (loss={loss})\n"
                    error_msg += "This usually indicates gradient explosion. Try:\n"
                    error_msg += "1. Lower the learning rate in config.py\n"
                    error_msg += "2. Use a smaller batch size\n"
                    error_msg += "3. Load a pre-trained checkpoint"
                    raise ValueError(error_msg)

                if batch_count <= 5 or batch_count % 10 == 0:
                    current_lr = state.world_model.ae_optimizer.param_groups[0]['lr']
                    if epoch_mode and samples_per_epoch > 0:
                        current_epoch = (samples_seen - starting_samples) // samples_per_epoch + 1
                        print(f"[DEBUG] Batch {batch_count}: epoch={current_epoch}/{total_epochs}, samples_seen={samples_seen}/{final_samples_target}, loss={loss:.6f}, lr={current_lr:.6e}")
                    else:
                        print(f"[DEBUG] Batch {batch_count}: samples_seen={samples_seen}/{final_samples_target}, loss={loss:.6f}, lr={current_lr:.6e}")

                # Periodic progress update (no evaluation)
                if samples_seen - last_eval_samples >= UPDATE_INTERVAL:
                    last_eval_samples = samples_seen
                    elapsed_time = time.time() - training_start_time

                    # Track training loss for progress plots
                    cumulative_metrics['samples_seen'].append(samples_seen)
                    cumulative_metrics['loss_at_sample'].append(loss)

                    # Calculate validation loss if validation session is loaded
                    val_loss = None
                    if has_validation:
                        val_loss = calculate_validation_loss(batch_size)
                        cumulative_metrics['val_loss_at_sample'].append(val_loss)
                        if val_loss is not None:
                            print(f"[DEBUG] Validation loss: {val_loss:.6f}")

                    # Step plateau scheduler with validation loss (at update intervals, not every batch)
                    if use_plateau_scheduler and val_loss is not None:
                        state.world_model.ae_scheduler.step(val_loss)

                    # Divergence checking for early stopping
                    update_count += 1
                    if stop_on_divergence and val_loss is not None:
                        if update_count >= divergence_min_updates:
                            gap = val_loss - loss
                            ratio = val_loss / max(loss, 1e-8)
                            diverged = (gap >= divergence_gap) or (ratio >= divergence_ratio)
                            if diverged:
                                divergence_count += 1
                                print(f"[DIVERGENCE] Check {divergence_count}/{divergence_patience}: "
                                      f"gap={gap:.6f} (thresh={divergence_gap}), ratio={ratio:.2f} (thresh={divergence_ratio})")
                            else:
                                divergence_count = 0  # Reset on non-divergent check

                            if divergence_count >= divergence_patience:
                                stop_early = True
                                stop_reason = (f"Divergence detected after {update_count} updates: "
                                              f"val={val_loss:.6f} vs train={loss:.6f} "
                                              f"(gap={gap:.6f}, ratio={ratio:.2f})")
                                print(f"[DIVERGENCE] {stop_reason}")

                    # Debug logging to diagnose periodic loss spikes
                    log_training_debug_state(state.world_model, batch_count, samples_seen, loss, enable_wandb)

                    # Save best model if loss improved
                    # Use validation loss if available, otherwise use training loss
                    checkpoint_loss = val_loss if (val_loss is not None) else loss
                    if checkpoint_loss < best_loss:
                        best_loss = checkpoint_loss
                        success, save_msg, auto_saved_checkpoints = save_best_model_checkpoint(
                            checkpoint_loss, samples_seen, state.world_model, auto_saved_checkpoints, num_best_models_to_keep,
                            is_val_loss=(val_loss is not None)
                        )
                        if success:
                            print(f"[AUTO-SAVE] {save_msg}")

                    # Log gradient diagnostics and additional metrics to wandb
                    if enable_wandb and grad_diagnostics:
                        try:
                            log_dict = {
                                "gradients/head_weight_norm": grad_diagnostics.get('head_weight_norm'),
                                "gradients/head_bias_norm": grad_diagnostics.get('head_bias_norm'),
                                "gradients/mask_token_norm": grad_diagnostics.get('mask_token_norm'),
                                "gradients/qkv_weight_norm": grad_diagnostics.get('qkv_weight_norm'),
                                "loss/plain": grad_diagnostics.get('loss_plain'),
                                "loss/focal": grad_diagnostics.get('loss_focal'),
                                "loss/standard": grad_diagnostics.get('loss_standard'),
                                "loss/nonblack": grad_diagnostics.get('loss_nonblack'),
                                "loss/black_baseline": grad_diagnostics.get('black_baseline'),
                                "loss/frac_nonblack": grad_diagnostics.get('frac_nonblack'),
                                "focal/weight_mean": grad_diagnostics.get('focal_weight_mean'),
                                "focal/weight_max": grad_diagnostics.get('focal_weight_max'),
                                "train/best_loss": best_loss,
                                "train/elapsed_time": elapsed_time,
                            }
                            # Add validation loss if available
                            if val_loss is not None:
                                log_dict["val/loss"] = val_loss
                            wandb.log(log_dict, step=samples_seen)
                        except Exception as e:
                            print(f"[WANDB WARNING] Failed to log gradient diagnostics: {str(e)}")

                    # Status message
                    current_lr = state.world_model.ae_optimizer.param_groups[0]['lr']
                    status_msg = f"📊 **Training Progress**\n\n"
                    status_msg += f"- Samples: {samples_seen:,} / {final_samples_target:,}\n"
                    if resume_mode:
                        status_msg += f"- Training: {samples_seen - starting_samples:,} / {samples_to_train:,} additional\n"
                    if stop_on_divergence:
                        status_msg += f"- Mode: Train until divergence\n"
                        status_msg += f"- Divergence checks: {divergence_count}/{divergence_patience}\n"
                    status_msg += f"- Current batch loss: {loss:.6f}\n"
                    status_msg += f"- Best loss: {best_loss:.6f}\n"
                    status_msg += f"- Learning rate: {current_lr:.6e}\n"

                    # Yield update (no evaluation plots)
                    print(f"[DEBUG] Yielding update at {samples_seen} samples (batch {batch_count})")
                    update_result = generate_batch_training_update(
                        samples_seen, final_samples_target, cumulative_metrics,
                        status_msg, None, None, current_observation_idx,
                        window_size, num_random_obs, completed=False, elapsed_time=elapsed_time
                    )

                    yield update_result
                    print(f"[DEBUG] Resumed after yield, continuing training...")

                    # Check for early stop after yield
                    if stop_early:
                        break

            # Check if we exited due to early stop
            if stop_early:
                print(f"[DEBUG] Training stopped early: {stop_reason}")

            print(f"[DEBUG] Exited training loop (fallback): batches_processed={batch_count}, samples_seen={samples_seen}, final_target={final_samples_target}")

          # For divergence mode: if not stopped early, regenerate indices and continue
          if stop_on_divergence and not stop_early:
            chunk_number += 1
            print(f"[DEBUG] Divergence mode: generating chunk {chunk_number + 1} ({divergence_epochs_per_chunk} more epochs)")

            # Regenerate indices for next chunk (always epoch-based in divergence mode)
            new_samples = max_samples_per_epoch * divergence_epochs_per_chunk
            sampled_indices = []
            for _ in range(divergence_epochs_per_chunk):
                epoch_indices = all_valid_indices.copy()
                random.shuffle(epoch_indices)
                sampled_indices.extend(epoch_indices)

            # Update targets
            final_samples_target = samples_seen + new_samples

            # Recreate DataLoader
            try:
                dataloader = create_canvas_dataloader(
                    canvas_cache=canvas_cache,
                    frame_indices=sampled_indices,
                    batch_size=batch_size,
                    config=config.AutoencoderConcatPredictorWorldModelConfig,
                    device=state.device,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=True,
                    persistent_workers=(num_workers > 0),
                    transfer_to_device=(not use_stream_pipelining),
                )
            except Exception as e:
                print(f"[DEBUG] Error recreating DataLoader: {e}")
                stop_early = True
                stop_reason = f"Error regenerating training data: {e}"

            if not stop_early:
                # Reset batch count for new chunk
                batch_count = 0
                # Continue to next iteration of the outer training loop
                continue

          # Exit the training loop (either stopped early or not in divergence mode)
          break

        # Final evaluation after training complete (outside while loop, inside try block)
        print(f"[DEBUG] Running final evaluation...")

        # In divergence mode with validation: load best checkpoint and evaluate on validation session
        if stop_on_divergence and has_validation and auto_saved_checkpoints:
            # Load best checkpoint for validation evaluation
            _, best_checkpoint_path = auto_saved_checkpoints[0]  # First is best (sorted by loss)
            print(f"[DEBUG] Loading best checkpoint for validation eval: {best_checkpoint_path}")

            try:
                checkpoint = torch.load(best_checkpoint_path, map_location=state.device)
                state.world_model.autoencoder.load_state_dict(checkpoint['model_state_dict'])
                best_samples = checkpoint.get('samples_seen', 0)
                best_loss = checkpoint.get('loss', 0)
                print(f"[DEBUG] Loaded best checkpoint: samples={best_samples}, loss={best_loss:.6f}")

                # Run evaluation on validation session
                val_status, fig_loss, fig_dist, val_stats_text, val_stats = evaluate_validation_session()

                if val_stats is not None:
                    status_msg = f"## Best Validation Checkpoint Evaluation\n\n"
                    status_msg += f"**Checkpoint:** `{os.path.basename(best_checkpoint_path)}`\n"
                    status_msg += f"**Saved at:** {best_samples:,} samples (val loss: {best_loss:.6f})\n\n"
                    status_msg += val_stats_text
                    stats = val_stats
                    current_loss = stats['hybrid']['mean']
                else:
                    # Fallback to training session evaluation
                    print(f"[DEBUG] Validation evaluation failed, falling back to training evaluation")
                    status_msg, fig_loss, fig_dist, stats_text, stats = evaluate_full_session()
                    current_loss = stats['hybrid']['mean'] if stats else 0
            except Exception as e:
                print(f"[DEBUG] Error loading best checkpoint: {e}, falling back to training evaluation")
                status_msg, fig_loss, fig_dist, stats_text, stats = evaluate_full_session()
                current_loss = stats['hybrid']['mean'] if stats else 0
        else:
            # Normal mode: run evaluation on training session
            status_msg, fig_loss, fig_dist, stats_text, stats = evaluate_full_session()
            current_loss = stats['hybrid']['mean'] if stats else 0

        cumulative_metrics['samples_seen'].append(samples_seen)
        cumulative_metrics['loss_at_sample'].append(current_loss)

        # Calculate final validation loss if validation session is loaded (for non-divergence mode)
        val_loss = None
        if has_validation and not stop_on_divergence:
            val_loss = calculate_validation_loss(batch_size)
            cumulative_metrics['val_loss_at_sample'].append(val_loss)
            if val_loss is not None:
                print(f"[DEBUG] Final validation loss: {val_loss:.6f}")

        # Save best model automatically if final loss is the best
        # Use validation loss if available, otherwise use training loss
        final_checkpoint_loss = val_loss if (has_validation and val_loss is not None) else current_loss
        is_val = has_validation and val_loss is not None
        if final_checkpoint_loss < best_loss:
            best_loss = final_checkpoint_loss
            success, save_msg, auto_saved_checkpoints = save_best_model_checkpoint(
                final_checkpoint_loss, samples_seen, state.world_model, auto_saved_checkpoints, num_best_models_to_keep,
                is_val_loss=is_val
            )
            if success:
                status_msg += f"\n\n{save_msg}"

        # Add early stop reason if applicable
        if stop_early:
            status_msg = f"**Training Stopped Early**\n{stop_reason}\n\n" + status_msg

        # Add final learning rate to status
        final_lr = state.world_model.ae_optimizer.param_groups[0]['lr']
        status_msg += f"\n\n📊 **Final Learning Rate**: {final_lr:.6e}"

        # Log full-session evaluation metrics to wandb
        if enable_wandb and stats:
            try:
                log_dict = {
                    "eval/loss_hybrid_mean": stats['hybrid']['mean'],
                    "eval/loss_hybrid_median": stats['hybrid']['median'],
                    "eval/loss_hybrid_std": stats['hybrid']['std'],
                    "eval/loss_hybrid_min": stats['hybrid']['min'],
                    "eval/loss_hybrid_max": stats['hybrid']['max'],
                    "eval/loss_hybrid_p25": stats['hybrid']['p25'],
                    "eval/loss_hybrid_p75": stats['hybrid']['p75'],
                    "eval/loss_hybrid_p90": stats['hybrid']['p90'],
                    "eval/loss_hybrid_p95": stats['hybrid']['p95'],
                    "eval/loss_hybrid_p99": stats['hybrid']['p99'],
                    "eval/final_lr": final_lr,
                }
                # Add standard loss stats if available (not in validation-only evaluation)
                if 'standard' in stats:
                    log_dict["eval/loss_standard_mean"] = stats['standard']['mean']
                    log_dict["eval/loss_standard_median"] = stats['standard']['median']
                    log_dict["eval/loss_standard_std"] = stats['standard']['std']
                wandb.log(log_dict, step=samples_seen)
            except Exception as e:
                print(f"[WANDB WARNING] Failed to log evaluation metrics: {str(e)}")

        # Calculate total elapsed time
        total_elapsed_time = time.time() - training_start_time

        print(f"[DEBUG] Final yield: samples_seen={samples_seen}, final_target={final_samples_target}, elapsed_time={total_elapsed_time:.2f}s")
        yield generate_batch_training_update(
            samples_seen, final_samples_target, cumulative_metrics,
            status_msg, fig_loss, fig_dist, current_observation_idx,
            window_size, num_random_obs, completed=True, elapsed_time=total_elapsed_time
        )
        print(f"[DEBUG] Training generator complete")

        # Finish wandb run on successful completion
        if enable_wandb:
            try:
                wandb.finish()
                print("[WANDB] Run finished successfully")
            except Exception as e:
                print(f"[WANDB WARNING] Error finishing wandb run: {str(e)}")

    except Exception as e:
        import traceback
        error_msg = f"Error during training: {str(e)}\n\n{traceback.format_exc()}"

        # Finish wandb run on error
        if enable_wandb:
            try:
                wandb.finish(exit_code=1)
                print("[WANDB] Run finished with error")
            except Exception as wandb_err:
                print(f"[WANDB WARNING] Error finishing wandb run: {str(wandb_err)}")

        yield error_msg, None, None, None, None, "", None


def run_batch_comparison(batch_sizes_str, total_samples):
    """
    Compare training efficiency across different batch sizes.

    Trains model with each batch size over the same total number of samples,
    measuring time, final loss, and convergence.
    """

    # Validation
    if state.world_model is None:
        yield "Please load a session first", "", None, None, None, None
        return

    if not state.session_state.get("observations") or not state.session_state.get("actions"):
        yield "No session data available", "", None, None, None, None
        return

    # Parse batch sizes
    try:
        batch_sizes = [int(b.strip()) for b in batch_sizes_str.split(',') if b.strip()]
        batch_sizes = [b for b in batch_sizes if b > 0]
        if not batch_sizes:
            raise ValueError("No valid batch sizes")
    except Exception as e:
        yield f"Error parsing batch sizes: {e}", "", None, None, None, None
        return

    total_samples = int(total_samples)
    observations = state.session_state["observations"]
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    valid_frame_count = len(observations) - (min_frames_needed - 1)

    if valid_frame_count < max(batch_sizes):
        yield f"Session too small: {valid_frame_count} valid frames, need {max(batch_sizes)}", "", None, None, None, None
        return

    # Save initial model state
    initial_state = {
        'model': state.world_model.autoencoder.state_dict(),
        'optimizer': state.world_model.ae_optimizer.state_dict(),
        'scheduler': state.world_model.ae_scheduler.state_dict(),
    }

    # Results storage
    results = {
        'batch_size': [], 'total_time': [], 'num_batches': [],
        'final_loss': [], 'final_loss_std': [],
        'loss_history': [], 'samples_seen': [],
    }

    # Test each batch size
    for batch_idx, batch_size in enumerate(batch_sizes):
        # Reset model to initial state
        state.world_model.autoencoder.load_state_dict(initial_state['model'])
        state.world_model.ae_optimizer.load_state_dict(initial_state['optimizer'])
        state.world_model.ae_scheduler.load_state_dict(initial_state['scheduler'])

        status = f"Testing batch_size={batch_size} ({batch_idx+1}/{len(batch_sizes)})..."
        yield status, "", None, None, None, None

        # Training loop with DataLoader (Phase 2 optimization)
        loss_history = []
        samples_seen_list = []
        total_start = time.time()
        state.world_model.autoencoder.train()

        # Check if we have pre-built canvas cache
        canvas_cache = state.session_state.get("canvas_cache", {})

        if canvas_cache:
            # Phase 2 optimization: Use DataLoader with parallel workers
            # Prepare valid frame indices
            valid_start = min_frames_needed - 1
            valid_end = len(observations) - 1
            all_valid_indices = list(range(valid_start, valid_end + 1))

            # Sample indices with replacement to reach total_samples
            # This maintains same behavior as original code
            sampled_all_indices = random.choices(all_valid_indices, k=total_samples)

            # Determine optimal number of workers for Windows/Linux
            # Note: Windows multiprocessing can be unstable, start with 0 workers (single-process)
            # and enable multiprocessing only on Linux or if explicitly configured
            import platform
            if platform.system() == 'Windows':
                num_workers = 0  # Conservative: single-process on Windows (avoids pickling issues)
            else:
                num_workers = 4  # Linux: use multiple workers for parallelism

            # Create DataLoader with optimized settings
            # Phase 4: Disable automatic GPU transfer for manual stream management
            use_stream_pipelining = (state.device == 'cuda' and torch.cuda.is_available())

            dataloader = create_canvas_dataloader(
                canvas_cache=canvas_cache,
                frame_indices=sampled_all_indices,
                batch_size=batch_size,
                config=config.AutoencoderConcatPredictorWorldModelConfig,
                device=state.device,
                num_workers=num_workers,
                shuffle=False,  # Already shuffled by random.choices
                pin_memory=True,  # Keep pinned memory for fast async transfers
                persistent_workers=(num_workers > 0),
                transfer_to_device=(not use_stream_pipelining),  # Phase 4: manual transfer if using streams
            )

            # Train on batches from DataLoader with CUDA stream pipelining (Phase 4 optimization)
            samples_seen = 0

            if use_stream_pipelining:
                # Phase 4: Use CUDA streams to overlap data transfer with GPU computation
                # Create a separate stream for async data transfers
                transfer_stream = torch.cuda.Stream()

                # Prefetching iterator with stream pipelining
                dataloader_iter = iter(dataloader)

                # Prefetch first batch and transfer to GPU
                try:
                    next_canvas_cpu, next_mask_cpu, next_indices = next(dataloader_iter)
                    with torch.cuda.stream(transfer_stream):
                        # Async transfer to GPU (non_blocking requires pinned memory)
                        next_canvas = next_canvas_cpu.to(state.device, non_blocking=True)
                        next_mask = next_mask_cpu.to(state.device, non_blocking=True)
                except StopIteration:
                    next_canvas = None

                batch_num = 0
                while next_canvas is not None:
                    # Wait for transfer to complete (synchronize streams)
                    torch.cuda.current_stream().wait_stream(transfer_stream)

                    # Current batch is ready (transferred from next)
                    canvas_tensor = next_canvas
                    patch_mask = next_mask

                    # Prefetch and transfer next batch in parallel with training
                    # This overlaps GPU training (current stream) with CPU→GPU transfer (transfer stream)
                    with torch.cuda.stream(transfer_stream):
                        try:
                            next_canvas_cpu, next_mask_cpu, next_indices = next(dataloader_iter)
                            # Async transfer to GPU while current batch trains
                            next_canvas = next_canvas_cpu.to(state.device, non_blocking=True)
                            next_mask = next_mask_cpu.to(state.device, non_blocking=True)
                        except StopIteration:
                            next_canvas = None

                    # Train on current batch while next batch transfers in background
                    loss, _ = state.world_model.autoencoder.train_on_canvas(
                        canvas_tensor, patch_mask, state.world_model.ae_optimizer
                    )
                    state.world_model.ae_scheduler.step()

                    samples_seen += canvas_tensor.shape[0]
                    loss_history.append(loss)
                    samples_seen_list.append(samples_seen)
                    batch_num += 1
            else:
                # Fallback: CPU or no CUDA - no stream pipelining (automatic transfer in collate_fn)
                for batch_num, (canvas_tensor, patch_mask, _) in enumerate(dataloader):
                    # Train
                    loss, _ = state.world_model.autoencoder.train_on_canvas(
                        canvas_tensor, patch_mask, state.world_model.ae_optimizer
                    )
                    state.world_model.ae_scheduler.step()

                    samples_seen += canvas_tensor.shape[0]
                    loss_history.append(loss)
                    samples_seen_list.append(samples_seen)

        else:
            # Fallback: Manual batching (backward compatibility, no cache available)
            num_batches = (total_samples + batch_size - 1) // batch_size

            for batch_num in range(num_batches):
                samples_this_batch = min(batch_size, total_samples - batch_num * batch_size)

                # Sample random frames and build canvases on-demand
                valid_start = min_frames_needed - 1
                valid_end = len(observations) - 1
                valid_indices = list(range(valid_start, valid_end + 1))
                sampled_indices = random.sample(valid_indices, samples_this_batch)

                batch_canvases = []
                for frame_idx in sampled_indices:
                    canvas, error, _, _ = build_canvas_from_frame(frame_idx)
                    if canvas is not None:
                        batch_canvases.append(canvas)

                if not batch_canvases:
                    continue

                # Stack into batch: [B, H, W, 3]
                canvas_batch = np.stack(batch_canvases, axis=0)
                canvas_tensor = canvas_to_tensor(canvas_batch, batch_size=len(batch_canvases)).to(state.device)

                # Generate masks on GPU (Phase 3 optimization)
                canvas_height, canvas_width = canvas_tensor.shape[-2:]
                num_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE

                patch_mask = compute_randomized_patch_mask_for_last_slot_gpu(
                    img_size=(canvas_height, canvas_width),
                    patch_size=config.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE,
                    num_frame_slots=num_frames,
                    sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
                    mask_ratio_min=config.MASK_RATIO_MIN,
                    mask_ratio_max=config.MASK_RATIO_MAX,
                    batch_size=len(batch_canvases),
                    device=state.device,
                )

                # Train
                loss, _ = state.world_model.autoencoder.train_on_canvas(
                    canvas_tensor, patch_mask, state.world_model.ae_optimizer
                )
                state.world_model.ae_scheduler.step()

                loss_history.append(loss)
                samples_seen_list.append((batch_num + 1) * samples_this_batch)

        total_time = time.time() - total_start

        # Store results
        final_losses = loss_history[-max(1, len(loss_history)//10):]
        results['batch_size'].append(batch_size)
        results['total_time'].append(total_time)
        results['num_batches'].append(len(loss_history))
        results['final_loss'].append(loss_history[-1] if loss_history else None)
        results['final_loss_std'].append(np.std(final_losses) if len(final_losses) > 1 else 0.0)
        results['loss_history'].append(loss_history)
        results['samples_seen'].append(samples_seen_list)

    # Restore initial state
    state.world_model.autoencoder.load_state_dict(initial_state['model'])
    state.world_model.ae_optimizer.load_state_dict(initial_state['optimizer'])
    state.world_model.ae_scheduler.load_state_dict(initial_state['scheduler'])

    # Generate visualizations
    import pandas as pd

    # 1. Time comparison
    fig_time, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(results['batch_size'])), results['total_time'],
                   color='skyblue', edgecolor='navy', linewidth=1.5)
    ax.set_xticks(range(len(results['batch_size'])))
    ax.set_xticklabels([f"BS={bs}" for bs in results['batch_size']])
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Total Training Time (seconds)', fontsize=12)
    ax.set_title(f'Training Time Comparison ({total_samples} samples)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, time_val in zip(bars, results['total_time']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()

    # 2. Quality comparison
    fig_quality, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(results['batch_size'])), results['final_loss'],
           yerr=results['final_loss_std'], color='lightcoral',
           edgecolor='darkred', linewidth=1.5, capsize=5)
    ax.set_xticks(range(len(results['batch_size'])))
    ax.set_xticklabels([f"BS={bs}" for bs in results['batch_size']])
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Final Loss (Hybrid)', fontsize=12)
    ax.set_title(f'Final Loss Comparison ({total_samples} samples)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # 3. Convergence comparison
    fig_convergence, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results['batch_size'])))
    for i, bs in enumerate(results['batch_size']):
        ax.plot(results['samples_seen'][i], results['loss_history'][i],
                label=f'BS={bs}', color=colors[i], linewidth=2, alpha=0.8)
    ax.set_xlabel('Samples Seen', fontsize=12)
    ax.set_ylabel('Loss (Hybrid)', fontsize=12)
    ax.set_title('Loss Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # 4. Summary table
    table_data = {
        'Batch Size': results['batch_size'],
        'Total Time (s)': [f"{t:.2f}" for t in results['total_time']],
        'Num Batches': results['num_batches'],
        'Samples/Sec': [f"{total_samples/t:.2f}" for t in results['total_time']],
        'Final Loss': [format_loss(l) for l in results['final_loss']],
        'Loss Std': [f"{s:.6f}" for s in results['final_loss_std']],
    }
    df = pd.DataFrame(table_data)

    # Summary markdown
    best_time_idx = np.argmin(results['total_time'])
    best_loss_idx = np.argmin(results['final_loss'])

    summary = f"## Results Summary\n\n"
    summary += f"**Total samples trained:** {total_samples}\n\n"
    summary += f"**Fastest:** Batch size {results['batch_size'][best_time_idx]} "
    summary += f"({results['total_time'][best_time_idx]:.2f}s)\n\n"
    summary += f"**Best loss:** Batch size {results['batch_size'][best_loss_idx]} "
    summary += f"({format_loss(results['final_loss'][best_loss_idx])})\n\n"

    if 1 in results['batch_size']:
        bs1_idx = results['batch_size'].index(1)
        speedup = results['total_time'][bs1_idx] / results['total_time'][best_time_idx]
        summary += f"**Speedup vs BS=1:** {speedup:.2f}x\n\n"

    summary += f"**Throughput range:** "
    summary += f"{min(total_samples/t for t in results['total_time']):.1f} - "
    summary += f"{max(total_samples/t for t in results['total_time']):.1f} samples/sec\n"

    final_status = "✅ Comparison complete!"
    yield final_status, summary, fig_time, fig_quality, fig_convergence, df
