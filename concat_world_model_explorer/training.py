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
from .evaluation import evaluate_full_session
from .visualization import (
    generate_batch_training_update,
    create_loss_vs_samples_plot,
    create_loss_vs_recent_checkpoints_plot,
)

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


def save_best_model_checkpoint(current_loss, samples_seen, world_model, auto_saved_checkpoints, num_best_models_to_keep):
    """
    Save a best model checkpoint and manage the list to keep only N best models.

    Args:
        current_loss: Current loss value
        samples_seen: Number of samples seen at this checkpoint
        world_model: The world model instance
        auto_saved_checkpoints: List of (loss, filepath) tuples for auto-saved checkpoints
        num_best_models_to_keep: Maximum number of best models to keep

    Returns:
        Tuple of (success: bool, message: str, updated_checkpoints_list)
    """
    checkpoint_dir = state.get_checkpoint_dir_for_session(state.session_state["session_dir"])
    checkpoint_name = f"best_model_auto_loss_{current_loss:.6f}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    try:
        # Save the new checkpoint
        checkpoint = {
            'model_state_dict': world_model.autoencoder.state_dict(),
            'optimizer_state_dict': world_model.ae_optimizer.state_dict(),
            'scheduler_state_dict': world_model.ae_scheduler.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'samples_seen': samples_seen,
            'loss': current_loss,
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


def run_world_model_batch(total_samples, batch_size, current_observation_idx, update_interval=100,
                          window_size=10, num_random_obs=5, num_best_models_to_keep=3,
                          enable_wandb=False, wandb_run_name=""):
    """
    Run batch training with periodic full-session evaluation.

    Args:
        total_samples: Total number of training samples
        batch_size: Batch size for training
        current_observation_idx: Currently selected observation to refresh during updates
        update_interval: Evaluate every N samples (default: 100)
        window_size: Number of recent checkpoints for rolling window plot (default: 10)
        num_random_obs: Number of random observations to visualize (default: 5)
        num_best_models_to_keep: Number of best model checkpoints to keep (default: 3)
        enable_wandb: Whether to log metrics to Weights & Biases (default: False)
        wandb_run_name: Optional custom name for wandb run (default: "")

    Yields:
        Tuple of (status, loss_vs_samples_plot, loss_vs_recent_plot, eval_loss_plot, eval_dist_plot,
                  obs_status, obs_combined_fig)
    """

    print(f"[DEBUG] run_world_model_batch called with: total_samples={total_samples}, batch_size={batch_size}, update_interval={update_interval}, num_best_models_to_keep={num_best_models_to_keep}")

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

    # Sample with replacement to reach total_samples (allows looping through session)
    all_valid_indices = list(range(min_frames_needed - 1, len(observations)))
    sampled_indices = random.choices(all_valid_indices, k=total_samples)
    print(f"[DEBUG] Sampled {len(sampled_indices)} indices from {len(all_valid_indices)} valid observations")
    print(f"[DEBUG] Expected batches: {len(sampled_indices) // batch_size} (with batch_size={batch_size})")

    # Apply linear scaling rule to learning rate and recreate optimizer/scheduler
    import math
    num_gradient_updates = math.ceil(total_samples / batch_size)
    base_batch_size = 1  # Base LR is defined for batch size 1
    base_lr = config.AutoencoderConcatPredictorWorldModelConfig.AUTOENCODER_LR
    scaled_lr = base_lr * (batch_size / base_batch_size)

    print(f"[DEBUG] Linear LR scaling: base_lr={base_lr:.6f} (BS={base_batch_size}) -> scaled_lr={scaled_lr:.6f} (BS={batch_size})")

    # Scale warmup steps inversely with batch size (to see same amount of data)
    base_warmup_steps = config.AutoencoderConcatPredictorWorldModelConfig.WARMUP_STEPS
    scaled_warmup_steps = max(1, int(base_warmup_steps / batch_size))  # Minimum 1 step
    warmup_samples_seen = scaled_warmup_steps * batch_size

    print(f"[DEBUG] Warmup scaling: base_warmup={base_warmup_steps} steps (BS={base_batch_size}) -> scaled_warmup={scaled_warmup_steps} steps (BS={batch_size}), samples_seen={warmup_samples_seen}")

    # Recreate optimizer with scaled LR
    import torch.optim as optim
    param_groups = [
        {"params": state.world_model.autoencoder.parameters()},
    ]
    state.world_model.ae_optimizer = optim.AdamW(param_groups, lr=scaled_lr)

    # Recreate scheduler with updated total steps and scaled warmup
    state.world_model.ae_scheduler = world_model_utils.create_warmup_cosine_scheduler(
        state.world_model.ae_optimizer,
        warmup_steps=scaled_warmup_steps,
        total_steps=num_gradient_updates,
        lr_min_ratio=config.AutoencoderConcatPredictorWorldModelConfig.LR_MIN_RATIO,
    )
    print(f"[DEBUG] Scheduler configured for {num_gradient_updates} steps (warmup: {scaled_warmup_steps}) with LR range {scaled_lr:.6f} -> {scaled_lr * config.AutoencoderConcatPredictorWorldModelConfig.LR_MIN_RATIO:.6f}")

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

    # Initialize metrics
    cumulative_metrics = {
        'samples_seen': [],
        'loss_at_sample': [],
    }
    samples_seen = 0
    UPDATE_INTERVAL = int(update_interval)  # Evaluate every N samples
    last_eval_samples = 0

    # Track best loss for automatic checkpoint saving
    best_loss = float('inf')

    # Track auto-saved checkpoints for this training run (list of tuples: (loss, filepath))
    auto_saved_checkpoints = []

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
                        "total_samples": total_samples,
                        "batch_size": batch_size,
                        "base_lr": base_lr,
                        "scaled_lr": scaled_lr,
                        "warmup_steps": scaled_warmup_steps,
                        "total_gradient_updates": num_gradient_updates,
                        "update_interval": update_interval,
                        "session_name": state.session_state.get("session_name", "unknown"),
                        "frame_size": config.AutoencoderConcatPredictorWorldModelConfig.FRAME_SIZE,
                        "separator_width": config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
                        "canvas_history_size": config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE,
                        "patch_size": config.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE,
                        "weight_decay": config.AutoencoderConcatPredictorWorldModelConfig.WEIGHT_DECAY,
                        "lr_min_ratio": config.AutoencoderConcatPredictorWorldModelConfig.LR_MIN_RATIO,
                        "focal_beta": config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_BETA,
                        "focal_loss_alpha": config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_LOSS_ALPHA,
                    }
                )
                print(f"[WANDB] Initialized run: {wandb.run.name}")
            except Exception as e:
                print(f"[WANDB WARNING] Failed to initialize wandb: {str(e)}")
                print("[WANDB WARNING] Continuing training without wandb logging")
                enable_wandb = False  # Disable wandb for rest of training

    # Training loop
    state.world_model.autoencoder.train()

    batch_count = 0
    try:
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

            print(f"[DEBUG] Starting training loop: next_canvas is {'not None' if next_canvas is not None else 'None'}, samples_seen={samples_seen}, total_samples={total_samples}")
            while next_canvas is not None and samples_seen < total_samples:
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
                    print(f"[DEBUG] Batch {batch_count}: samples_seen={samples_seen}/{total_samples}, loss={loss:.6f}")

                # Periodic progress update (no evaluation)
                if samples_seen - last_eval_samples >= UPDATE_INTERVAL:
                    last_eval_samples = samples_seen
                    elapsed_time = time.time() - training_start_time

                    # Track training loss for progress plots
                    cumulative_metrics['samples_seen'].append(samples_seen)
                    cumulative_metrics['loss_at_sample'].append(loss)

                    # Save best model if loss improved
                    if loss < best_loss:
                        best_loss = loss
                        success, save_msg, auto_saved_checkpoints = save_best_model_checkpoint(
                            loss, samples_seen, state.world_model, auto_saved_checkpoints, num_best_models_to_keep
                        )
                        if success:
                            print(f"[AUTO-SAVE] {save_msg}")

                    # Log gradient diagnostics and additional metrics to wandb
                    if enable_wandb and grad_diagnostics:
                        try:
                            wandb.log({
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
                            }, step=samples_seen)
                        except Exception as e:
                            print(f"[WANDB WARNING] Failed to log gradient diagnostics: {str(e)}")

                    # Status message
                    current_lr = state.world_model.ae_optimizer.param_groups[0]['lr']
                    status_msg = f"📊 **Training Progress**\n\n"
                    status_msg += f"- Samples: {samples_seen:,} / {total_samples:,}\n"
                    status_msg += f"- Current batch loss: {loss:.6f}\n"
                    status_msg += f"- Best loss: {best_loss:.6f}\n"
                    status_msg += f"- Learning rate: {current_lr:.6e}\n"

                    # Yield update (no evaluation plots)
                    print(f"[DEBUG] Yielding update at {samples_seen} samples (batch {batch_count})")
                    update_result = generate_batch_training_update(
                        samples_seen, total_samples, cumulative_metrics,
                        status_msg, None, None, current_observation_idx,
                        window_size, num_random_obs, completed=False, elapsed_time=elapsed_time
                    )

                    # Log observation samples visualization to wandb
                    if enable_wandb and len(update_result) > 6 and update_result[6] is not None:
                        try:
                            wandb.log({
                                "visualizations/observation_samples": wandb.Image(update_result[6]),
                            }, step=samples_seen)
                        except Exception as e:
                            print(f"[WANDB WARNING] Failed to log observation samples: {str(e)}")

                    yield update_result
                    print(f"[DEBUG] Resumed after yield, continuing training...")

            print(f"[DEBUG] Exited training loop: batches_processed={batch_count}, samples_seen={samples_seen}, total_samples={total_samples}")
            print(f"[DEBUG] Exit reason: next_canvas is {'None' if next_canvas is None else 'not None'}, samples_seen >= total_samples: {samples_seen >= total_samples}")

        else:
            # Fallback: no stream pipelining
            print(f"[DEBUG] Using fallback mode (no stream pipelining)")
            for canvas_tensor, patch_mask, _ in dataloader:
                batch_count += 1
                if samples_seen >= total_samples:
                    print(f"[DEBUG] Breaking: samples_seen ({samples_seen}) >= total_samples ({total_samples})")
                    break

                # Train on batch
                loss, grad_diagnostics = state.world_model.autoencoder.train_on_canvas(
                    canvas_tensor, patch_mask, state.world_model.ae_optimizer
                )
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
                    print(f"[DEBUG] Batch {batch_count}: samples_seen={samples_seen}/{total_samples}, loss={loss:.6f}")

                # Periodic progress update (no evaluation)
                if samples_seen - last_eval_samples >= UPDATE_INTERVAL:
                    last_eval_samples = samples_seen
                    elapsed_time = time.time() - training_start_time

                    # Track training loss for progress plots
                    cumulative_metrics['samples_seen'].append(samples_seen)
                    cumulative_metrics['loss_at_sample'].append(loss)

                    # Save best model if loss improved
                    if loss < best_loss:
                        best_loss = loss
                        success, save_msg, auto_saved_checkpoints = save_best_model_checkpoint(
                            loss, samples_seen, state.world_model, auto_saved_checkpoints, num_best_models_to_keep
                        )
                        if success:
                            print(f"[AUTO-SAVE] {save_msg}")

                    # Log gradient diagnostics and additional metrics to wandb
                    if enable_wandb and grad_diagnostics:
                        try:
                            wandb.log({
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
                            }, step=samples_seen)
                        except Exception as e:
                            print(f"[WANDB WARNING] Failed to log gradient diagnostics: {str(e)}")

                    # Status message
                    current_lr = state.world_model.ae_optimizer.param_groups[0]['lr']
                    status_msg = f"📊 **Training Progress**\n\n"
                    status_msg += f"- Samples: {samples_seen:,} / {total_samples:,}\n"
                    status_msg += f"- Current batch loss: {loss:.6f}\n"
                    status_msg += f"- Best loss: {best_loss:.6f}\n"
                    status_msg += f"- Learning rate: {current_lr:.6e}\n"

                    # Yield update (no evaluation plots)
                    print(f"[DEBUG] Yielding update at {samples_seen} samples (batch {batch_count})")
                    update_result = generate_batch_training_update(
                        samples_seen, total_samples, cumulative_metrics,
                        status_msg, None, None, current_observation_idx,
                        window_size, num_random_obs, completed=False, elapsed_time=elapsed_time
                    )

                    # Log observation samples visualization to wandb
                    if enable_wandb and len(update_result) > 6 and update_result[6] is not None:
                        try:
                            wandb.log({
                                "visualizations/observation_samples": wandb.Image(update_result[6]),
                            }, step=samples_seen)
                        except Exception as e:
                            print(f"[WANDB WARNING] Failed to log observation samples: {str(e)}")

                    yield update_result
                    print(f"[DEBUG] Resumed after yield, continuing training...")

            print(f"[DEBUG] Exited training loop (fallback): batches_processed={batch_count}, samples_seen={samples_seen}, total_samples={total_samples}")

        # Final evaluation after training complete
        print(f"[DEBUG] Running final evaluation...")
        status_msg, fig_loss, fig_dist, stats_text, stats = evaluate_full_session()
        cumulative_metrics['samples_seen'].append(samples_seen)
        current_loss = stats['hybrid']['mean']
        cumulative_metrics['loss_at_sample'].append(current_loss)

        # Save best model automatically if final loss is the best
        if current_loss < best_loss:
            best_loss = current_loss
            success, save_msg, auto_saved_checkpoints = save_best_model_checkpoint(
                current_loss, samples_seen, state.world_model, auto_saved_checkpoints, num_best_models_to_keep
            )
            if success:
                status_msg += f"\n\n{save_msg}"

        # Add final learning rate to status
        final_lr = state.world_model.ae_optimizer.param_groups[0]['lr']
        status_msg += f"\n\n📊 **Final Learning Rate**: {final_lr:.6e}"

        # Log full-session evaluation metrics to wandb
        if enable_wandb and stats:
            try:
                wandb.log({
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
                    "eval/loss_standard_mean": stats['standard']['mean'],
                    "eval/loss_standard_median": stats['standard']['median'],
                    "eval/loss_standard_std": stats['standard']['std'],
                    "eval/final_lr": final_lr,
                }, step=samples_seen)

                # Log evaluation visualizations
                if fig_loss is not None:
                    wandb.log({
                        "visualizations/eval_loss_over_time": wandb.Image(fig_loss),
                    }, step=samples_seen)
                if fig_dist is not None:
                    wandb.log({
                        "visualizations/eval_loss_distribution": wandb.Image(fig_dist),
                    }, step=samples_seen)
            except Exception as e:
                print(f"[WANDB WARNING] Failed to log evaluation metrics: {str(e)}")

        # Calculate total elapsed time
        total_elapsed_time = time.time() - training_start_time

        print(f"[DEBUG] Final yield: samples_seen={samples_seen}, total_samples={total_samples}, elapsed_time={total_elapsed_time:.2f}s")
        yield generate_batch_training_update(
            samples_seen, total_samples, cumulative_metrics,
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
