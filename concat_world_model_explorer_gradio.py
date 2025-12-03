"""
Concat World Model Explorer Gradio App

A web-based UI for running AutoencoderConcatPredictorWorldModel on recorded robot sessions.
Allows running the world model for N iterations and visualizing training and prediction results.
"""

import os
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gradio as gr
from datetime import datetime
from pathlib import Path

import config
import world_model_utils
from autoencoder_concat_predictor_world_model import AutoencoderConcatPredictorWorldModel
from recording_reader import RecordingReader
from replay_robot import ReplayRobot
from session_explorer_lib import (
    list_session_dirs,
    load_session_metadata,
    load_session_events,
    extract_observations,
    extract_actions,
    load_frame_image,
    format_timestamp,
    prebuild_all_canvases,
)
from recorded_policy import create_recorded_action_selector
from models.autoencoder_concat_predictor import (
    canvas_to_tensor,
    compute_patch_mask_for_last_slot,
    compute_randomized_patch_mask_for_last_slot,
    compute_randomized_patch_mask_for_last_slot_gpu,
    compute_hybrid_loss_on_masked_patches,
)
from models.canvas_dataset import create_canvas_dataloader
from attention_viz import (
    compute_patch_centers,
    draw_attention_connections,
    create_attention_statistics,
    create_attention_heatmap,
)

# Session directories
TOROIDAL_DOT_SESSIONS_DIR = config.TOROIDAL_DOT_RECORDING_DIR
TOROIDAL_DOT_CHECKPOINT_DIR = config.TOROIDAL_DOT_CHECKPOINT_DIR

# Ensure checkpoint directory exists
os.makedirs(TOROIDAL_DOT_CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global state
session_state = {}
world_model = None
replay_robot = None
current_checkpoint_name = None

def format_loss(loss_value):
    """Format loss value for display"""
    if loss_value is None:
        return "--"
    if loss_value < 0.001:
        return f"{loss_value:.2e}"
    else:
        return f"{loss_value:.6f}"

def format_grad_diagnostics(grad_diag):
    """Format gradient diagnostics for display"""
    if grad_diag is None:
        return "No diagnostics available"

    lines = []
    lines.append("**Gradient Flow Diagnostics:**\n")
    lines.append(f"- Learning Rate: {grad_diag['lr']:.2e}")
    lines.append(f"- Decoder Head Weight Grad Norm: {format_loss(grad_diag['head_weight_norm'])}")
    lines.append(f"- Decoder Head Bias Grad Norm: {format_loss(grad_diag['head_bias_norm'])}")
    lines.append(f"- Mask Token Grad Norm: {format_loss(grad_diag['mask_token_norm'])}")
    lines.append(f"- First Decoder QKV Weight Grad Norm: {format_loss(grad_diag['qkv_weight_norm'])}")

    lines.append("\n**Loss Metrics:**\n")
    focal_beta = grad_diag.get('focal_beta', 10.0)
    focal_alpha = grad_diag.get('focal_alpha', 0.2)
    lines.append(f"- **Hybrid Loss (training)**: {format_loss(grad_diag.get('loss_hybrid'))} *[α={focal_alpha:.2f}]*")
    lines.append(f"  - Plain MSE Component: {format_loss(grad_diag.get('loss_plain'))}")
    lines.append(f"  - Focal MSE Component: {format_loss(grad_diag.get('loss_focal'))} *[β={focal_beta:.1f}]*")
    lines.append(f"- Standard Loss (for comparison): {format_loss(grad_diag.get('loss_standard'))}")

    lines.append("\n**Focal Weight Statistics:**\n")
    lines.append(f"- Mean Focal Weight: {grad_diag.get('focal_weight_mean', 1.0):.3f}")
    lines.append(f"- Max Focal Weight: {grad_diag.get('focal_weight_max', 1.0):.3f}")
    lines.append("  *(Focal weights adaptively upweight high-error pixels)*")

    lines.append("\n**Loss Dilution Diagnostics:**\n")
    lines.append(f"- Loss on Non-Black Pixels: {format_loss(grad_diag.get('loss_nonblack'))}")
    lines.append(f"- Black Baseline (if model predicted black): {format_loss(grad_diag.get('black_baseline'))}")
    lines.append(f"- Fraction of Non-Black Pixels: {grad_diag.get('frac_nonblack', 0):.6f} ({grad_diag.get('frac_nonblack', 0)*100:.2f}%)")

    # Add interpretation hint
    if grad_diag.get('loss_nonblack') is not None and grad_diag.get('black_baseline') is not None:
        loss_nonblack = grad_diag['loss_nonblack']
        baseline = grad_diag['black_baseline']
        if loss_nonblack is not None and baseline is not None:
            improvement_pct = (1 - loss_nonblack / baseline) * 100
            lines.append(f"\n**Dot Learning Progress:** {improvement_pct:.1f}% improvement over black baseline")

            if loss_nonblack >= baseline * 0.95:  # Loss is close to or above baseline
                lines.append("⚠️ **Loss on non-black pixels ≈ black baseline** → Model is NOT learning the dot (loss dilution)")
            elif loss_nonblack < baseline * 0.5:  # Clear improvement
                lines.append("✅ **Loss dropping below baseline** → Model IS learning the dot!")
            else:
                lines.append("📊 **Some progress** → Keep training to see if dot emerges")

    return "\n".join(lines)

def refresh_sessions():
    """Refresh session list"""
    sessions = list_session_dirs()
    # Filter for toroidal dot sessions only
    toroidal_sessions = [s for s in sessions if "toroidal" in s.lower()]
    choices = [os.path.basename(s) + " - " + s for s in toroidal_sessions]
    return gr.Dropdown(choices=choices, value=choices[-1] if choices else None)

def load_session(session_choice):
    """Load a session from dropdown selection"""
    global session_state, world_model, replay_robot

    if not session_choice:
        return "No session selected", None, "", 0, 0

    # Extract session_dir from choice
    session_dir = session_choice.split(" - ")[-1]

    metadata = load_session_metadata(session_dir)
    events = load_session_events(session_dir)
    observations = extract_observations(events, session_dir)
    actions = extract_actions(events)

    # Pre-build all canvases for fast batch training
    print("\n" + "="*60)
    print("Pre-building all canvases for batch training...")
    print("="*60)
    prebuild_start = time.time()
    canvas_cache = prebuild_all_canvases(
        session_dir,
        observations,
        actions,
        config.AutoencoderConcatPredictorWorldModelConfig
    )
    prebuild_time = time.time() - prebuild_start
    print(f"Canvas pre-building completed in {prebuild_time:.2f}s")
    print(f"Memory usage: ~{len(canvas_cache) * 224 * 688 * 3 / (1024**2):.1f} MB")
    print("="*60 + "\n")

    session_state.update({
        "session_name": os.path.basename(session_dir),
        "session_dir": session_dir,
        "metadata": metadata,
        "events": events,
        "observations": observations,
        "actions": actions,
        "canvas_cache": canvas_cache,
    })

    # Build session info
    if not observations:
        info = f"**{session_state['session_name']}** has no observation frames."
        return info, None, "", 0, 0

    details = [
        f"**Session:** {session_state['session_name']}",
        f"**Total events:** {len(events)}",
        f"**Observations:** {len(observations)}",
        f"**Actions:** {len(actions)}",
    ]
    if metadata:
        start_time = metadata.get("start_time")
        if start_time:
            details.append(f"**Start:** {start_time}")
        robot_type_display = metadata.get("robot_type")
        if robot_type_display:
            details.append(f"**Robot:** {robot_type_display}")

    info = "\n\n".join(details)

    # Load first frame
    first_frame = load_frame_image(observations[0]["full_path"])
    frame_info = f"**Observation 1 / {len(observations)}**\n\nStep: {observations[0]['step']}\n\nTimestamp: {format_timestamp(observations[0]['timestamp'])}"

    max_frames = len(observations) - 1

    # Initialize world model and replay robot
    action_space = metadata.get("action_space", [])

    # Create recording reader
    reader = RecordingReader(session_dir)

    # Create replay robot
    replay_robot = ReplayRobot(reader, action_space)

    # TODO: Change action selectors throughout the codebase to only return an action
    # Currently recorded_action_selector returns (action, all_action_predictions) but
    # AutoencoderConcatPredictorWorldModel expects just action to be returned

    # Create world model with recorded action selector
    recorded_selector = create_recorded_action_selector(reader)

    def action_selector_adapter(observation, action_space):
        action, _ = recorded_selector()
        return action

    world_model = AutoencoderConcatPredictorWorldModel(
        replay_robot,
        action_selector=action_selector_adapter,
        device=device,
    )

    info += "\n\n**World model initialized and ready to run**"

    return info, first_frame, frame_info, 0, max_frames

def update_frame(frame_idx):
    """Update frame display"""
    if not session_state.get("observations"):
        return None, ""

    observations = session_state["observations"]
    if frame_idx >= len(observations):
        frame_idx = len(observations) - 1

    obs = observations[frame_idx]
    frame = load_frame_image(obs["full_path"])
    frame_info = f"**Observation {frame_idx + 1} / {len(observations)}**\n\nStep: {obs['step']}\n\nTimestamp: {format_timestamp(obs['timestamp'])}"

    return frame, frame_info

def build_canvas_from_frame(frame_idx):
    """Build canvas from selected frame and its history (helper function)"""
    global session_state

    observations = session_state["observations"]
    actions = session_state["actions"]

    # Validate frame index
    frame_idx = int(frame_idx)
    if frame_idx >= len(observations):
        return None, f"Frame index {frame_idx} out of range (max: {len(observations)-1})", None, None

    # Check if we have enough history
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    if frame_idx < min_frames_needed - 1:
        return None, f"Need at least {min_frames_needed} frames of history. Selected frame {frame_idx+1} doesn't have enough history.", None, None

    # Try to use pre-built canvas cache first (Phase 1 optimization)
    canvas_cache = session_state.get("canvas_cache", {})
    if frame_idx in canvas_cache:
        cached_data = canvas_cache[frame_idx]
        return cached_data['canvas'], None, cached_data['start_idx'], cached_data['interleaved']

    # Fallback: build canvas on-demand (for backward compatibility or if cache not available)
    # Extract frames for canvas (need CANVAS_HISTORY_SIZE frames)
    start_idx = frame_idx - (min_frames_needed - 1)
    selected_frames = []
    for idx in range(start_idx, frame_idx + 1):
        frame_img = load_frame_image(observations[idx]["full_path"])
        selected_frames.append(np.array(frame_img))

    # Extract actions between those frames
    selected_actions = []
    for idx in range(start_idx, frame_idx):
        # Find action that corresponds to transition from frame idx to idx+1
        # Actions list should align with observations
        if idx < len(actions):
            selected_actions.append(actions[idx])
        else:
            # Fallback: use default action if not enough actions recorded
            selected_actions.append({"action": 0})  # Default toroidal dot action

    # Build interleaved history
    interleaved = [selected_frames[0]]
    for i in range(len(selected_actions)):
        interleaved.append(selected_actions[i])
        if i + 1 < len(selected_frames):
            interleaved.append(selected_frames[i + 1])

    # Build training canvas
    from models.autoencoder_concat_predictor import build_canvas
    training_canvas = build_canvas(
        interleaved,
        frame_size=config.AutoencoderConcatPredictorWorldModelConfig.FRAME_SIZE,
        sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
    )

    return training_canvas, None, start_idx, interleaved

def infer_on_single_canvas(frame_idx):
    """Run inference only on a single canvas built from selected frame and its history"""
    global world_model, session_state

    if world_model is None:
        return "Please load a session first", "", None, None, None, None

    if not session_state.get("observations") or not session_state.get("actions"):
        return "No session data available", "", None, None, None, None

    # Build canvas
    training_canvas, error, start_idx, interleaved = build_canvas_from_frame(frame_idx)
    if training_canvas is None:
        return error, "", None, None, None, None

    frame_idx = int(frame_idx)

    # Store canvas for visualization
    world_model.last_training_canvas = training_canvas

    # Compute patch mask for last slot
    canvas_tensor = canvas_to_tensor(training_canvas).to(device)
    canvas_height, canvas_width = canvas_tensor.shape[-2:]
    num_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE

    import config as cfg
    patch_mask = compute_randomized_patch_mask_for_last_slot(
        img_size=(canvas_height, canvas_width),
        patch_size=cfg.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE,
        num_frame_slots=num_frames,
        sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
        mask_ratio_min=cfg.MASK_RATIO_MIN,
        mask_ratio_max=cfg.MASK_RATIO_MAX,
    ).to(device)

    # Store the mask for visualization
    world_model.last_training_mask = patch_mask

    # Run inference (no training)
    world_model.autoencoder.eval()
    with torch.no_grad():
        pred_patches, _ = world_model.autoencoder.forward_with_patch_mask(canvas_tensor, patch_mask)
        img_pred = world_model.autoencoder.unpatchify(pred_patches)

        # Compute loss using shared helper function from model module
        target_patches = world_model.autoencoder.patchify(canvas_tensor)  # [1, num_patches, patch_dim]

        # Select masked patches
        masked_pred = pred_patches[patch_mask]
        masked_target = target_patches[patch_mask]

        # Compute loss with same function as training
        import config as cfg
        loss_dict = compute_hybrid_loss_on_masked_patches(
            masked_pred,
            masked_target,
            focal_alpha=cfg.AutoencoderConcatPredictorWorldModelConfig.FOCAL_LOSS_ALPHA,
            focal_beta=cfg.AutoencoderConcatPredictorWorldModelConfig.FOCAL_BETA
        )

        # Convert tensor losses to scalars
        loss_dict = {
            'loss_hybrid': loss_dict['loss_hybrid'].item() if torch.is_tensor(loss_dict['loss_hybrid']) else loss_dict['loss_hybrid'],
            'loss_plain': loss_dict['loss_plain'].item() if torch.is_tensor(loss_dict['loss_plain']) else loss_dict['loss_plain'],
            'loss_focal': loss_dict['loss_focal'].item() if torch.is_tensor(loss_dict['loss_focal']) else loss_dict['loss_focal'],
            'loss_standard': loss_dict['loss_standard'].item() if torch.is_tensor(loss_dict['loss_standard']) else loss_dict['loss_standard'],
            'focal_alpha': cfg.AutoencoderConcatPredictorWorldModelConfig.FOCAL_LOSS_ALPHA,
            'focal_beta': cfg.AutoencoderConcatPredictorWorldModelConfig.FOCAL_BETA,
            'focal_weight_mean': loss_dict['focal_weight_mean'],
            'focal_weight_max': loss_dict['focal_weight_max'],
        }

    # Generate visualizations
    status_msg = f"**Inference on canvas from frames {start_idx+1}-{frame_idx+1}**\n\n"
    status_msg += f"(No training performed - weights unchanged)\n\n"
    status_msg += f"Hybrid loss: {format_loss(loss_dict['loss_hybrid'])}\n"
    status_msg += f"Standard loss: {format_loss(loss_dict['loss_standard'])}"

    # Inference info (matches training display format)
    inference_info = f"**Reconstruction Loss (Hybrid):** {format_loss(loss_dict['loss_hybrid'])} *[α={loss_dict['focal_alpha']:.2f}]*\n\n"
    inference_info += f"**Standard Loss (unweighted):** {format_loss(loss_dict['loss_standard'])}\n\n"
    inference_info += f"*Note: Using same loss calculation as training (hybrid: α * plain_mse + (1-α) * focal_mse with β={loss_dict['focal_beta']:.1f})*\n\n"
    inference_info += f"*Model weights are unchanged (inference only)*"

    # Training canvas visualizations
    fig_inference_canvas = None
    fig_inference_canvas_masked = None
    fig_inference_inpainting_full = None
    fig_inference_inpainting_composite = None

    if world_model.last_training_canvas is not None:
        # 1. Original canvas
        fig_inference_canvas, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.imshow(world_model.last_training_canvas)
        ax.set_title(f"Inference Canvas (Frames {start_idx+1}-{frame_idx+1})")
        ax.axis("off")
        plt.tight_layout()

        # Generate additional visualizations if mask is available
        if world_model.last_training_mask is not None:
            # 2. Canvas with mask overlay
            canvas_with_mask = world_model.get_canvas_with_mask_overlay(
                world_model.last_training_canvas,
                world_model.last_training_mask
            )
            fig_inference_canvas_masked, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(canvas_with_mask)
            ax.set_title("Inference Canvas with Mask (Red = Masked Patches)")
            ax.axis("off")
            plt.tight_layout()

            # 3. Full model output
            inpainting_full = world_model.get_canvas_inpainting_full_output(
                world_model.last_training_canvas,
                world_model.last_training_mask
            )
            fig_inference_inpainting_full, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(inpainting_full)
            ax.set_title("Inference - Full Model Output")
            ax.axis("off")
            plt.tight_layout()

            # 4. Composite
            inpainting_composite = world_model.get_canvas_inpainting_composite(
                world_model.last_training_canvas,
                world_model.last_training_mask
            )
            fig_inference_inpainting_composite, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(inpainting_composite)
            ax.set_title("Inference - Composite")
            ax.axis("off")
            plt.tight_layout()

    return status_msg, inference_info, fig_inference_canvas, fig_inference_canvas_masked, fig_inference_inpainting_full, fig_inference_inpainting_composite

def evaluate_full_session():
    """Evaluate model loss on all observations in the session"""
    global world_model, session_state

    if world_model is None:
        return "Please load a session first", None, None, ""

    if not session_state.get("observations") or not session_state.get("actions"):
        return "No session data available", None, None, ""

    observations = session_state["observations"]
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
    world_model.autoencoder.eval()
    for frame_idx in range(min_frames_needed - 1, len(observations)):
        # Build canvas
        training_canvas, error, start_idx, interleaved = build_canvas_from_frame(frame_idx)
        if training_canvas is None:
            continue

        # Compute patch mask
        canvas_tensor = canvas_to_tensor(training_canvas).to(device)
        canvas_height, canvas_width = canvas_tensor.shape[-2:]
        num_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE

        import config as cfg
        patch_mask = compute_randomized_patch_mask_for_last_slot(
            img_size=(canvas_height, canvas_width),
            patch_size=cfg.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE,
            num_frame_slots=num_frames,
            sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
            mask_ratio_min=cfg.MASK_RATIO_MIN,
            mask_ratio_max=cfg.MASK_RATIO_MAX,
        ).to(device)

        # Run inference
        with torch.no_grad():
            pred_patches, _ = world_model.autoencoder.forward_with_patch_mask(canvas_tensor, patch_mask)
            target_patches = world_model.autoencoder.patchify(canvas_tensor)

            # Select masked patches
            masked_pred = pred_patches[patch_mask]
            masked_target = target_patches[patch_mask]

            # Compute loss
            loss_dict = compute_hybrid_loss_on_masked_patches(
                masked_pred,
                masked_target,
                focal_alpha=cfg.AutoencoderConcatPredictorWorldModelConfig.FOCAL_LOSS_ALPHA,
                focal_beta=cfg.AutoencoderConcatPredictorWorldModelConfig.FOCAL_BETA
            )

            # Store results
            results['observation_indices'].append(frame_idx)
            results['loss_hybrid'].append(loss_dict['loss_hybrid'].item() if torch.is_tensor(loss_dict['loss_hybrid']) else loss_dict['loss_hybrid'])
            results['loss_standard'].append(loss_dict['loss_standard'].item() if torch.is_tensor(loss_dict['loss_standard']) else loss_dict['loss_standard'])
            results['loss_plain'].append(loss_dict['loss_plain'].item() if torch.is_tensor(loss_dict['loss_plain']) else loss_dict['loss_plain'])
            results['loss_focal'].append(loss_dict['loss_focal'].item() if torch.is_tensor(loss_dict['loss_focal']) else loss_dict['loss_focal'])
            results['timestamps'].append(observations[frame_idx]['timestamp'])

    if len(results['observation_indices']) == 0:
        return "No observations could be evaluated", None, None, ""

    # Compute statistics
    import numpy as np
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

def create_loss_vs_samples_plot(cumulative_metrics):
    """Create plot of loss vs samples seen"""
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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
    global world_model, session_state
    import matplotlib.pyplot as plt

    if world_model is None:
        return "Please load session and train model first", None

    if not observation_indices:
        return "No observations to visualize", None

    observations = session_state.get("observations", [])
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
        canvas_tensor = canvas_to_tensor(training_canvas).to(device)
        canvas_height, canvas_width = canvas_tensor.shape[-2:]
        num_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE

        patch_mask = compute_randomized_patch_mask_for_last_slot(
            img_size=(canvas_height, canvas_width),
            patch_size=config.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE,
            num_frame_slots=num_frames,
            sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
            mask_ratio_min=config.MASK_RATIO_MIN,
            mask_ratio_max=config.MASK_RATIO_MAX,
        ).to(device)

        # Run inference
        world_model.autoencoder.eval()
        with torch.no_grad():
            pred_patches, _ = world_model.autoencoder.forward_with_patch_mask(
                canvas_tensor, patch_mask
            )

        # Get composite
        composite = world_model.get_canvas_inpainting_composite(training_canvas, patch_mask)

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
    global session_state

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
    observations = session_state.get("observations", [])
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    all_valid_indices = list(range(min_frames_needed - 1, len(observations)))

    if all_valid_indices:
        import random
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
    import math

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

def train_on_single_canvas(frame_idx, num_training_steps):
    """Train autoencoder on a single canvas built from selected frame and its history"""
    global world_model, session_state

    if world_model is None:
        return "Please load a session first", "", "", None, None, None, None, None

    if not session_state.get("observations") or not session_state.get("actions"):
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
        loss = world_model.train_autoencoder(training_canvas)
        loss_history.append(loss)

    # Store training canvas and mask for visualization
    world_model.last_training_canvas = training_canvas
    world_model.last_training_loss = loss_history[-1] if loss_history else None

    # Generate visualizations
    status_msg = f"**Trained on canvas from frames {start_idx+1}-{frame_idx+1}**\n\n"
    status_msg += f"Training steps: {num_training_steps}\n\n"
    status_msg += f"Final loss: {format_loss(loss_history[-1] if loss_history else None)}"

    # Training info with gradient diagnostics
    final_loss = loss_history[-1] if loss_history else None
    training_info = f"**Training Loss (Hybrid):** {format_loss(final_loss)}"

    # Add standard loss if available from diagnostics
    if world_model.last_grad_diagnostics and 'loss_standard' in world_model.last_grad_diagnostics:
        std_loss = world_model.last_grad_diagnostics['loss_standard']
        training_info += f"\n\n**Standard Loss (unweighted):** {format_loss(std_loss)}"

    # Gradient diagnostics
    grad_diag_info = format_grad_diagnostics(world_model.last_grad_diagnostics)

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

    if world_model.last_training_canvas is not None:
        # 1. Original training canvas
        fig_training_canvas, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.imshow(world_model.last_training_canvas)
        ax.set_title(f"Training Canvas (Frames {start_idx+1}-{frame_idx+1})")
        ax.axis("off")
        plt.tight_layout()

        # Generate additional visualizations if mask is available
        if world_model.last_training_mask is not None:
            # 2. Canvas with mask overlay
            canvas_with_mask = world_model.get_canvas_with_mask_overlay(
                world_model.last_training_canvas,
                world_model.last_training_mask
            )
            fig_training_canvas_masked, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(canvas_with_mask)
            ax.set_title("Training Canvas with Mask (Red = Masked Patches)")
            ax.axis("off")
            plt.tight_layout()

            # 3. Full model output
            inpainting_full = world_model.get_canvas_inpainting_full_output(
                world_model.last_training_canvas,
                world_model.last_training_mask
            )
            fig_training_inpainting_full, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(inpainting_full)
            ax.set_title("Training Inpainting - Full Model Output")
            ax.axis("off")
            plt.tight_layout()

            # 4. Composite
            inpainting_composite = world_model.get_canvas_inpainting_composite(
                world_model.last_training_canvas,
                world_model.last_training_mask
            )
            fig_training_inpainting_composite, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(inpainting_composite)
            ax.set_title("Training Inpainting - Composite")
            ax.axis("off")
            plt.tight_layout()

    return status_msg, training_info, grad_diag_info, fig_loss_history, fig_training_canvas, fig_training_canvas_masked, fig_training_inpainting_full, fig_training_inpainting_composite

def run_world_model(num_iterations):
    """Run the world model for N iterations with live metrics tracking and periodic UI updates"""
    global world_model, replay_robot

    if world_model is None:
        yield "Please load a session first", "", "", None, None, None, None, None, None, None, None, "", "--"
        return

    if num_iterations <= 0:
        yield "Number of iterations must be greater than 0", "", "", None, None, None, None, None, None, None, None, "", "--"
        return

    # Initialize or retrieve metrics history
    if not hasattr(world_model, '_metrics_history'):
        world_model._metrics_history = {
            'iteration': [],
            'training_loss': [],
            'prediction_error': [],
            'iteration_time': [],
        }

    # Helper function to generate all visualizations
    def generate_visualizations(loop_count, completed=False):
        """Generate all plots and metrics for current state"""

        # Get final state from world model's stored state
        last_prediction_np = world_model.last_prediction
        current_frame_np = None
        prediction_error = None

        # Get current frame from history
        if len(world_model.interleaved_history) > 0:
            for idx in range(len(world_model.interleaved_history) - 1, -1, -1):
                if idx % 2 == 0:  # Even indices are frames
                    current_frame_np = world_model.interleaved_history[idx]
                    break

        if current_frame_np is not None and last_prediction_np is not None:
            import world_model_utils
            pred_tensor = world_model_utils.to_model_tensor(last_prediction_np, device)
            curr_tensor = world_model_utils.to_model_tensor(current_frame_np, device)
            prediction_error = torch.nn.functional.mse_loss(pred_tensor, curr_tensor).item()

        # Create current metrics display
        current_metrics = ""
        if len(world_model._metrics_history['iteration']) > 0:
            current_metrics = f"**Latest Iteration Metrics:**\n\n"
            current_metrics += f"- Training Loss: {format_loss(world_model.last_training_loss)}\n"
            current_metrics += f"- Prediction Error: {format_loss(prediction_error)}\n"
            current_metrics += f"- Iteration Time: {world_model._metrics_history['iteration_time'][-1]:.2f}s\n"

        # Create metrics history plots (3 plots in a row)
        fig_metrics = None
        if len(world_model._metrics_history['iteration']) > 0:
            fig_metrics, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Training Loss over iterations
            valid_train_loss = [(i, loss) for i, loss in zip(world_model._metrics_history['iteration'],
                                                              world_model._metrics_history['training_loss']) if loss is not None]
            if valid_train_loss:
                iters, losses = zip(*valid_train_loss)
                axes[0].plot(iters, losses, 'b-o', linewidth=2, markersize=4)
                axes[0].set_xlabel('Iteration')
                axes[0].set_ylabel('Training Loss')
                axes[0].set_title('Training Loss Over Time')
                axes[0].grid(True, alpha=0.3)

            # Prediction Error over iterations
            valid_pred_error = [(i, err) for i, err in zip(world_model._metrics_history['iteration'],
                                                            world_model._metrics_history['prediction_error']) if err is not None]
            if valid_pred_error:
                iters, errors = zip(*valid_pred_error)
                axes[1].plot(iters, errors, 'r-o', linewidth=2, markersize=4)
                axes[1].set_xlabel('Iteration')
                axes[1].set_ylabel('Prediction Error (MSE)')
                axes[1].set_title('Prediction Error Over Time')
                axes[1].grid(True, alpha=0.3)

            # Iteration Time over iterations
            axes[2].plot(world_model._metrics_history['iteration'],
                           world_model._metrics_history['iteration_time'], 'purple', linewidth=2, marker='o', markersize=4)
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Time (seconds)')
            axes[2].set_title('Iteration Time Over Time')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

        # Create status message
        total_iters = len(world_model._metrics_history['iteration'])
        if completed:
            status_msg = f"**Completed {num_iterations} iterations** (Total: {total_iters} iterations)"
        else:
            status_msg = f"**Running... {total_iters}/{num_iterations} iterations complete**"

        if loop_count > 0:
            status_msg += f"\n\n*Session looped {loop_count} time{'s' if loop_count > 1 else ''}*"

        # Current frame and prediction
        fig_frames = None
        if current_frame_np is not None:
            if last_prediction_np is not None:
                fig_frames, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(current_frame_np)
                axes[0].set_title("Current Frame")
                axes[0].axis("off")
                axes[1].imshow(last_prediction_np)
                axes[1].set_title(f"Last Predicted Frame\nError: {format_loss(prediction_error)}")
                axes[1].axis("off")
                plt.tight_layout()
            else:
                fig_frames, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(current_frame_np)
                ax.set_title("Current Frame (no prediction)")
                ax.axis("off")
                plt.tight_layout()

        # Training canvas visualizations
        fig_training_canvas = None
        fig_training_canvas_masked = None
        fig_training_inpainting_full = None
        fig_training_inpainting_composite = None
        training_info = ""

        if world_model.last_training_canvas is not None:
            # 1. Original training canvas
            fig_training_canvas, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(world_model.last_training_canvas)
            ax.set_title(f"Training Canvas (Original)")
            ax.axis("off")
            plt.tight_layout()

            training_info = f"**Training Loss (Focal):** {format_loss(world_model.last_training_loss)}"

            # Add standard loss if available from diagnostics
            if world_model.last_grad_diagnostics and 'loss_standard' in world_model.last_grad_diagnostics:
                std_loss = world_model.last_grad_diagnostics['loss_standard']
                training_info += f"\n\n**Standard Loss (unweighted):** {format_loss(std_loss)}"

            # Generate additional visualizations if mask is available
            if world_model.last_training_mask is not None:
                # 2. Canvas with mask overlay (shows which patches are masked)
                canvas_with_mask = world_model.get_canvas_with_mask_overlay(
                    world_model.last_training_canvas,
                    world_model.last_training_mask
                )
                fig_training_canvas_masked, ax = plt.subplots(1, 1, figsize=(12, 4))
                ax.imshow(canvas_with_mask)
                ax.set_title("Training Canvas with Mask (Red = Masked Patches)")
                ax.axis("off")
                plt.tight_layout()

                # 3. Full model output (what model reconstructs for everything)
                inpainting_full = world_model.get_canvas_inpainting_full_output(
                    world_model.last_training_canvas,
                    world_model.last_training_mask
                )
                fig_training_inpainting_full, ax = plt.subplots(1, 1, figsize=(12, 4))
                ax.imshow(inpainting_full)
                ax.set_title("Training Inpainting - Full Model Output (All Patches)")
                ax.axis("off")
                plt.tight_layout()

                # 4. Composite (original + inpainted masked regions only)
                inpainting_composite = world_model.get_canvas_inpainting_composite(
                    world_model.last_training_canvas,
                    world_model.last_training_mask
                )
                fig_training_inpainting_composite, ax = plt.subplots(1, 1, figsize=(12, 4))
                ax.imshow(inpainting_composite)
                ax.set_title("Training Inpainting - Composite (Original + Inpainted Masked Regions)")
                ax.axis("off")
                plt.tight_layout()

        # Gradient diagnostics
        grad_diag_info = format_grad_diagnostics(world_model.last_grad_diagnostics)

        # Prediction canvas and predicted frame
        fig_prediction_canvas = None
        fig_predicted_frame = None

        if world_model.last_prediction_canvas is not None:
            fig_prediction_canvas, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(world_model.last_prediction_canvas)
            ax.set_title("Prediction Canvas (with blank slot)")
            ax.axis("off")
            plt.tight_layout()

        if last_prediction_np is not None:
            fig_predicted_frame, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(last_prediction_np)
            ax.set_title("Predicted Next Frame")
            ax.axis("off")
            plt.tight_layout()

        return status_msg, current_metrics, fig_metrics, fig_frames, fig_training_canvas, fig_training_canvas_masked, fig_training_inpainting_full, fig_training_inpainting_composite, fig_prediction_canvas, fig_predicted_frame, training_info, grad_diag_info, format_loss(prediction_error)

    try:
        import time
        loop_count = 0
        from config import AutoencoderConcatPredictorWorldModelConfig as Config
        UPDATE_INTERVAL = Config.GRADIO_UPDATE_INTERVAL  # Update UI every N iterations

        # Run world model with metric tracking and periodic UI updates
        for i in range(num_iterations):
            start_time = time.time()

            # Run one iteration, catching StopIteration to loop the session
            iteration_successful = False
            while not iteration_successful:
                try:
                    world_model.run(max_iterations=1)
                    iteration_successful = True
                except StopIteration:
                    # Session ended, reset reader to loop back to beginning
                    loop_count += 1
                    print(f"Session ended at iteration {i+1}/{num_iterations}, looping back to beginning (loop #{loop_count})...")
                    replay_robot.reader.reset()
                    world_model.interleaved_history = []
                    world_model.last_prediction = None
                    world_model.last_prediction_canvas = None

            iteration_time = time.time() - start_time

            # Get current metrics from world model's stored state
            last_prediction_np = world_model.last_prediction
            prediction_error = None

            # Get current frame from history if available
            if len(world_model.interleaved_history) > 0 and last_prediction_np is not None:
                for idx in range(len(world_model.interleaved_history) - 1, -1, -1):
                    if idx % 2 == 0:  # Even indices are frames
                        import world_model_utils
                        current_frame_np = world_model.interleaved_history[idx]
                        pred_tensor = world_model_utils.to_model_tensor(last_prediction_np, device)
                        curr_tensor = world_model_utils.to_model_tensor(current_frame_np, device)
                        prediction_error = torch.nn.functional.mse_loss(pred_tensor, curr_tensor).item()
                        break

            # Record metrics
            iter_num = len(world_model._metrics_history['iteration']) + 1
            world_model._metrics_history['iteration'].append(iter_num)
            world_model._metrics_history['training_loss'].append(world_model.last_training_loss if world_model.last_training_loss else None)
            world_model._metrics_history['prediction_error'].append(prediction_error)
            world_model._metrics_history['iteration_time'].append(iteration_time)

            # Periodically yield updates to refresh the UI
            if (i + 1) % UPDATE_INTERVAL == 0 or (i + 1) == num_iterations:
                yield generate_visualizations(loop_count, completed=(i + 1) == num_iterations)

        # Final update
        yield generate_visualizations(loop_count, completed=True)

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        yield error_msg, "", "", None, None, None, None, None, None, None, None, "", "", "--"

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
    checkpoint_name = f"best_model_auto_loss_{current_loss:.6f}.pth"
    checkpoint_path = os.path.join(TOROIDAL_DOT_CHECKPOINT_DIR, checkpoint_name)

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
                          window_size=10, num_random_obs=5, num_best_models_to_keep=3):
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

    Yields:
        Tuple of (status, loss_vs_samples_plot, loss_vs_recent_plot, eval_loss_plot, eval_dist_plot,
                  obs_status, obs_combined_fig)
    """
    global world_model, session_state
    import random
    import time

    print(f"[DEBUG] run_world_model_batch called with: total_samples={total_samples}, batch_size={batch_size}, update_interval={update_interval}, num_best_models_to_keep={num_best_models_to_keep}")

    # Validation
    if world_model is None:
        yield "Please load a session first", None, None, None, None, "", None
        return

    if not session_state.get("observations") or not session_state.get("actions"):
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

    observations = session_state["observations"]
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    max_samples_per_epoch = len(observations) - (min_frames_needed - 1)

    if max_samples_per_epoch <= 0:
        yield f"Session too small: need at least {min_frames_needed} observations", None, None, None, None, "", None
        return

    # Check if canvas cache exists
    canvas_cache = session_state.get("canvas_cache", {})
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
        {"params": world_model.autoencoder.parameters()},
    ]
    world_model.ae_optimizer = optim.AdamW(param_groups, lr=scaled_lr)

    # Recreate scheduler with updated total steps and scaled warmup
    world_model.ae_scheduler = world_model_utils.create_warmup_cosine_scheduler(
        world_model.ae_optimizer,
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
    use_stream_pipelining = (device == 'cuda' and torch.cuda.is_available())

    try:
        dataloader = create_canvas_dataloader(
            canvas_cache=canvas_cache,
            frame_indices=sampled_indices,
            batch_size=batch_size,
            config=config.AutoencoderConcatPredictorWorldModelConfig,
            device=device,
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

    # Training loop
    world_model.autoencoder.train()

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
                    next_canvas = next_canvas_cpu.to(device, non_blocking=True)
                    next_mask = next_mask_cpu.to(device, non_blocking=True)
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
                        next_canvas = next_canvas_cpu.to(device, non_blocking=True)
                        next_mask = next_mask_cpu.to(device, non_blocking=True)
                    except StopIteration:
                        next_canvas = None

                # Train on current batch
                loss, _ = world_model.autoencoder.train_on_canvas(
                    canvas_tensor, patch_mask, world_model.ae_optimizer
                )
                world_model.ae_scheduler.step()
                samples_seen += canvas_tensor.shape[0]

                if batch_count <= 5 or batch_count % 10 == 0:
                    print(f"[DEBUG] Batch {batch_count}: samples_seen={samples_seen}/{total_samples}, loss={loss:.6f}")

                # Periodic evaluation checkpoint
                if samples_seen - last_eval_samples >= UPDATE_INTERVAL:
                    # PAUSE training, run evaluation
                    status_msg, fig_loss, fig_dist, stats_text, stats = evaluate_full_session()

                    # Update metrics
                    cumulative_metrics['samples_seen'].append(samples_seen)
                    current_loss = stats['hybrid']['mean']
                    cumulative_metrics['loss_at_sample'].append(current_loss)
                    last_eval_samples = samples_seen

                    # Save best model automatically if loss improved
                    if current_loss < best_loss:
                        best_loss = current_loss
                        success, save_msg, auto_saved_checkpoints = save_best_model_checkpoint(
                            current_loss, samples_seen, world_model, auto_saved_checkpoints, num_best_models_to_keep
                        )
                        if success:
                            status_msg += f"\n\n{save_msg}"

                    # Add current learning rate to status
                    current_lr = world_model.ae_optimizer.param_groups[0]['lr']
                    status_msg += f"\n\n📊 **Current Learning Rate**: {current_lr:.6e}"

                    # Yield update (includes visualization for currently selected observation)
                    print(f"[DEBUG] Yielding update at {samples_seen} samples (batch {batch_count})")
                    elapsed_time = time.time() - training_start_time
                    yield generate_batch_training_update(
                        samples_seen, total_samples, cumulative_metrics,
                        status_msg, fig_loss, fig_dist, current_observation_idx,
                        window_size, num_random_obs, completed=False, elapsed_time=elapsed_time
                    )
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
                loss, _ = world_model.autoencoder.train_on_canvas(
                    canvas_tensor, patch_mask, world_model.ae_optimizer
                )
                world_model.ae_scheduler.step()
                samples_seen += canvas_tensor.shape[0]

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

                # Periodic evaluation checkpoint
                if samples_seen - last_eval_samples >= UPDATE_INTERVAL:
                    # PAUSE training, run evaluation
                    status_msg, fig_loss, fig_dist, stats_text, stats = evaluate_full_session()

                    # Update metrics
                    cumulative_metrics['samples_seen'].append(samples_seen)
                    current_loss = stats['hybrid']['mean']
                    cumulative_metrics['loss_at_sample'].append(current_loss)
                    last_eval_samples = samples_seen

                    # Save best model automatically if loss improved
                    if current_loss < best_loss:
                        best_loss = current_loss
                        success, save_msg, auto_saved_checkpoints = save_best_model_checkpoint(
                            current_loss, samples_seen, world_model, auto_saved_checkpoints, num_best_models_to_keep
                        )
                        if success:
                            status_msg += f"\n\n{save_msg}"

                    # Add current learning rate to status
                    current_lr = world_model.ae_optimizer.param_groups[0]['lr']
                    status_msg += f"\n\n📊 **Current Learning Rate**: {current_lr:.6e}"

                    # Yield update
                    print(f"[DEBUG] Yielding update at {samples_seen} samples (batch {batch_count})")
                    elapsed_time = time.time() - training_start_time
                    yield generate_batch_training_update(
                        samples_seen, total_samples, cumulative_metrics,
                        status_msg, fig_loss, fig_dist, current_observation_idx,
                        window_size, num_random_obs, completed=False, elapsed_time=elapsed_time
                    )
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
                current_loss, samples_seen, world_model, auto_saved_checkpoints, num_best_models_to_keep
            )
            if success:
                status_msg += f"\n\n{save_msg}"

        # Add final learning rate to status
        final_lr = world_model.ae_optimizer.param_groups[0]['lr']
        status_msg += f"\n\n📊 **Final Learning Rate**: {final_lr:.6e}"

        # Calculate total elapsed time
        total_elapsed_time = time.time() - training_start_time

        print(f"[DEBUG] Final yield: samples_seen={samples_seen}, total_samples={total_samples}, elapsed_time={total_elapsed_time:.2f}s")
        yield generate_batch_training_update(
            samples_seen, total_samples, cumulative_metrics,
            status_msg, fig_loss, fig_dist, current_observation_idx,
            window_size, num_random_obs, completed=True, elapsed_time=total_elapsed_time
        )
        print(f"[DEBUG] Training generator complete")

    except Exception as e:
        import traceback
        error_msg = f"Error during training: {str(e)}\n\n{traceback.format_exc()}"
        yield error_msg, None, None, None, None, "", None

def parse_manual_patch_indices(manual_patches_str):
    """Parse manual patch indices from string input.

    Supports formats like:
    - "0,5,10" (comma-separated)
    - "0-10" (range)
    - "0,5,10-15,20" (mixed)
    """
    if not manual_patches_str or not manual_patches_str.strip():
        return []

    indices = []
    parts = manual_patches_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range like "10-15"
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                indices.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            # Single index
            try:
                indices.append(int(part))
            except ValueError:
                continue

    return sorted(list(set(indices)))  # Remove duplicates and sort


def generate_attention_visualization(
    frame_idx,
    selection_mode,
    brightness_threshold,
    manual_patches,
    quantile,
    layer0_enabled,
    layer1_enabled,
    layer2_enabled,
    layer3_enabled,
    layer4_enabled,
    head0_enabled,
    head1_enabled,
    head2_enabled,
    head3_enabled,
    aggregation,
    selected_aggregation,
    viz_type
):
    """Generate decoder attention visualization FROM selected patches TO all patches using the currently selected frame"""
    global world_model, session_state
    from attention_viz import (
        detect_dot_patches, draw_attention_from_selected,
        create_attention_heatmap_overlay, compute_patch_centers
    )

    if world_model is None:
        return "Please load a session first", None, ""

    if not session_state.get("observations") or not session_state.get("actions"):
        return "No session data available", None, ""

    try:
        # Build canvas from selected frame
        training_canvas, error, start_idx, interleaved = build_canvas_from_frame(frame_idx)
        if training_canvas is None:
            return error, None, ""

        frame_idx = int(frame_idx)

        # Get current frame for visualization
        current_frame_path = session_state["observations"][frame_idx]["full_path"]
        current_frame = load_frame_image(current_frame_path)
        current_frame_np = np.array(current_frame)

        # Convert canvas to tensor and run inference to get attention
        canvas_tensor = canvas_to_tensor(training_canvas).to(device)

        # Run forward pass with attention capture (no masking for inference)
        with torch.no_grad():
            latent = world_model.autoencoder.encode(canvas_tensor)
            decoded, attn_weights_list = world_model.autoencoder.decode(latent, return_attn=True)

        # Get patch size
        patch_size = config.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE

        # Determine selected patches based on mode
        if selection_mode == "Automatic Dot Detection":
            selected_indices = detect_dot_patches(
                current_frame_np,
                patch_size=patch_size,
                brightness_threshold=brightness_threshold
            )
            if len(selected_indices) == 0:
                return f"No patches detected with brightness >= {brightness_threshold:.2f}. Try lowering the threshold.", None, ""
        else:  # Manual Selection
            selected_indices = parse_manual_patch_indices(manual_patches)
            if len(selected_indices) == 0:
                return "No valid patch indices provided. Please enter indices (e.g., 0,5,10 or 0-10)", None, ""
            # Validate indices
            img_height, img_width = current_frame_np.shape[:2]
            num_patches_h = img_height // patch_size
            num_patches_w = img_width // patch_size
            max_patch_idx = num_patches_h * num_patches_w - 1
            selected_indices = [idx for idx in selected_indices if 0 <= idx <= max_patch_idx]
            if len(selected_indices) == 0:
                return f"No valid patch indices. Max index is {max_patch_idx}", None, ""

        selected_indices = np.array(selected_indices)

        # Configure which layers to show
        enabled_layers = [
            layer0_enabled,
            layer1_enabled,
            layer2_enabled,
            layer3_enabled,
            layer4_enabled
        ]

        # Configure which heads to show
        head_checkboxes = [head0_enabled, head1_enabled, head2_enabled, head3_enabled]
        enabled_heads = [i for i, enabled in enumerate(head_checkboxes) if enabled]

        # Validate that at least one head is selected
        if len(enabled_heads) == 0:
            return "Error: At least one attention head must be selected", None, ""

        # Calculate canvas-adjusted selected indices
        # The selected patches are from the current frame (last frame in canvas)
        # We need to adjust their indices to canvas coordinates
        frame_size = config.AutoencoderConcatPredictorWorldModelConfig.FRAME_SIZE[0]  # 224
        sep_width = config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH  # 16
        canvas_history_size = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE  # 3

        canvas_height, canvas_width = training_canvas.shape[:2]
        num_patches_w_canvas = canvas_width // patch_size
        num_patches_w_frame = frame_size // patch_size  # 14

        # The last frame starts at this pixel/patch offset
        last_frame_pixel_offset = (canvas_history_size - 1) * (frame_size + sep_width)
        last_frame_patch_col_offset = last_frame_pixel_offset // patch_size

        # Convert frame-based selected indices to canvas coordinates
        canvas_selected_indices = []
        for frame_patch_idx in selected_indices:
            # Convert frame patch index to (row, col) in frame
            frame_row = frame_patch_idx // num_patches_w_frame
            frame_col = frame_patch_idx % num_patches_w_frame

            # Convert to canvas coordinates
            canvas_col = last_frame_patch_col_offset + frame_col
            canvas_patch_idx = frame_row * num_patches_w_canvas + canvas_col
            canvas_selected_indices.append(canvas_patch_idx)

        canvas_selected_indices = np.array(canvas_selected_indices)

        # Generate visualization based on type
        if viz_type == "Patch-to-Patch Lines":
            # Use canvas for visualization
            img_height, img_width = training_canvas.shape[:2]
            patch_centers = compute_patch_centers(img_height, img_width, patch_size)

            fig = draw_attention_from_selected(
                canvas_img=training_canvas,
                patch_centers=patch_centers,
                attn_weights_list=attn_weights_list,
                selected_indices=canvas_selected_indices,  # Use canvas-adjusted indices
                quantile=quantile,
                enabled_layers=enabled_layers,
                enabled_heads=enabled_heads,
                aggregation=aggregation,
                selected_patch_aggregation=selected_aggregation,
                alpha=0.6
            )

        elif viz_type == "Heatmap Matrix":
            # For matrix heatmap, show the first enabled layer
            layer_idx = 0
            for i, enabled in enumerate(enabled_layers):
                if enabled:
                    layer_idx = i
                    break

            # Use existing heatmap function but modify for selected patches
            from attention_viz import aggregate_attention_heads
            attn_aggregated = aggregate_attention_heads(
                attn_weights_list[layer_idx],
                aggregation=aggregation,
                enabled_heads=enabled_heads
            )

            # Extract attention from selected patches to all patches
            num_patches = attn_aggregated.shape[0] - 1  # Exclude CLS token
            selected_token_indices = selected_indices + 1
            all_patch_indices = np.arange(num_patches) + 1

            attn_matrix = attn_aggregated[selected_token_indices][:, all_patch_indices]

            # Create heatmap
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            im = ax.imshow(attn_matrix, aspect='auto', cmap='viridis', interpolation='nearest')

            ax.set_xlabel('Target Patch Index', fontsize=12)
            ax.set_ylabel('Selected Patch Index', fontsize=12)

            heads_str = "All" if enabled_heads is None or len(enabled_heads) == 4 else f"[{','.join(map(str, enabled_heads))}]"
            ax.set_title(
                f'Decoder Layer {layer_idx} Attention Heatmap\n'
                f'FROM Selected Patches ({len(selected_indices)}) TO All Patches ({num_patches})\n'
                f'Heads: {heads_str} | Aggregation: {aggregation}',
                fontsize=12,
                pad=10
            )

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Attention Weight', fontsize=12)

            ax.set_xticks(np.arange(num_patches)[::10])
            ax.set_yticks(np.arange(len(selected_indices)))
            ax.set_yticklabels([str(idx) for idx in selected_indices])
            ax.grid(False)

            plt.tight_layout()

        else:  # Heatmap Overlay on Frame
            # For overlay, show the first enabled layer
            layer_idx = 0
            for i, enabled in enumerate(enabled_layers):
                if enabled:
                    layer_idx = i
                    break

            fig = create_attention_heatmap_overlay(
                frame_img=training_canvas,  # Use canvas instead of just the frame
                attn_weights_list=attn_weights_list,
                selected_indices=canvas_selected_indices,  # Use canvas-adjusted indices
                patch_size=patch_size,
                layer_idx=layer_idx,
                aggregation=aggregation,
                enabled_heads=enabled_heads,
                selected_patch_aggregation=selected_aggregation,
                alpha=0.6
            )

        # Generate statistics message
        heads_display = str(enabled_heads) if len(enabled_heads) < 4 else "All (0,1,2,3)"

        stats_text = f"**Attention Visualization Statistics:**\n\n"
        stats_text += f"- **Selection mode:** {selection_mode}\n"
        stats_text += f"- **Selected patches:** {len(selected_indices)}\n"
        stats_text += f"- **Selected patch indices:** {list(selected_indices)}\n"
        stats_text += f"- **Visualization type:** {viz_type}\n"
        stats_text += f"- **Head aggregation:** {aggregation}\n"
        stats_text += f"- **Selected patch aggregation:** {selected_aggregation}\n"
        stats_text += f"- **Enabled heads:** {heads_display}\n"
        if viz_type == "Patch-to-Patch Lines":
            stats_text += f"- **Quantile:** {quantile:.1f}% (showing top {100-quantile:.1f}% of connections)\n"

        status_msg = f"**Attention visualization generated successfully**\n\n"
        status_msg += f"Using frame {frame_idx + 1} from session\n"
        status_msg += f"Frame size: {current_frame_np.shape[0]}x{current_frame_np.shape[1]} pixels\n"
        status_msg += f"Patch size: {patch_size}x{patch_size} pixels\n"
        status_msg += f"Selected {len(selected_indices)} patches using {selection_mode}"

        return status_msg, fig, stats_text

    except Exception as e:
        import traceback
        error_msg = f"Error generating attention visualization:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, ""

def run_batch_comparison(batch_sizes_str, total_samples):
    """
    Compare training efficiency across different batch sizes.

    Trains model with each batch size over the same total number of samples,
    measuring time, final loss, and convergence.
    """
    global world_model, session_state

    # Validation
    if world_model is None:
        yield "Please load a session first", "", None, None, None, None
        return

    if not session_state.get("observations") or not session_state.get("actions"):
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
    observations = session_state["observations"]
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    valid_frame_count = len(observations) - (min_frames_needed - 1)

    if valid_frame_count < max(batch_sizes):
        yield f"Session too small: {valid_frame_count} valid frames, need {max(batch_sizes)}", "", None, None, None, None
        return

    # Save initial model state
    initial_state = {
        'model': world_model.autoencoder.state_dict(),
        'optimizer': world_model.ae_optimizer.state_dict(),
        'scheduler': world_model.ae_scheduler.state_dict(),
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
        world_model.autoencoder.load_state_dict(initial_state['model'])
        world_model.ae_optimizer.load_state_dict(initial_state['optimizer'])
        world_model.ae_scheduler.load_state_dict(initial_state['scheduler'])

        status = f"Testing batch_size={batch_size} ({batch_idx+1}/{len(batch_sizes)})..."
        yield status, "", None, None, None, None

        # Training loop with DataLoader (Phase 2 optimization)
        loss_history = []
        samples_seen_list = []
        total_start = time.time()
        world_model.autoencoder.train()

        # Check if we have pre-built canvas cache
        canvas_cache = session_state.get("canvas_cache", {})

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
            use_stream_pipelining = (device == 'cuda' and torch.cuda.is_available())

            dataloader = create_canvas_dataloader(
                canvas_cache=canvas_cache,
                frame_indices=sampled_all_indices,
                batch_size=batch_size,
                config=config.AutoencoderConcatPredictorWorldModelConfig,
                device=device,
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
                        next_canvas = next_canvas_cpu.to(device, non_blocking=True)
                        next_mask = next_mask_cpu.to(device, non_blocking=True)
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
                            next_canvas = next_canvas_cpu.to(device, non_blocking=True)
                            next_mask = next_mask_cpu.to(device, non_blocking=True)
                        except StopIteration:
                            next_canvas = None

                    # Train on current batch while next batch transfers in background
                    loss, _ = world_model.autoencoder.train_on_canvas(
                        canvas_tensor, patch_mask, world_model.ae_optimizer
                    )
                    world_model.ae_scheduler.step()

                    samples_seen += canvas_tensor.shape[0]
                    loss_history.append(loss)
                    samples_seen_list.append(samples_seen)
                    batch_num += 1
            else:
                # Fallback: CPU or no CUDA - no stream pipelining (automatic transfer in collate_fn)
                for batch_num, (canvas_tensor, patch_mask, _) in enumerate(dataloader):
                    # Train
                    loss, _ = world_model.autoencoder.train_on_canvas(
                        canvas_tensor, patch_mask, world_model.ae_optimizer
                    )
                    world_model.ae_scheduler.step()

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
                canvas_tensor = canvas_to_tensor(canvas_batch, batch_size=len(batch_canvases)).to(device)

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
                    device=device,
                )

                # Train
                loss, _ = world_model.autoencoder.train_on_canvas(
                    canvas_tensor, patch_mask, world_model.ae_optimizer
                )
                world_model.ae_scheduler.step()

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
    world_model.autoencoder.load_state_dict(initial_state['model'])
    world_model.ae_optimizer.load_state_dict(initial_state['optimizer'])
    world_model.ae_scheduler.load_state_dict(initial_state['scheduler'])

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

def list_available_checkpoints():
    """List all available checkpoint files in the checkpoint directory"""
    checkpoint_dir = Path(TOROIDAL_DOT_CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return []

    # Find all .pth files
    checkpoint_files = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.name for f in checkpoint_files]

def refresh_checkpoints():
    """Refresh checkpoint dropdown list"""
    checkpoints = list_available_checkpoints()
    choices = checkpoints if checkpoints else ["No checkpoints available"]
    return gr.Dropdown(choices=choices, value=choices[0] if checkpoints else None)

def save_model_weights(checkpoint_name):
    """Save current model weights to a checkpoint file"""
    global world_model, current_checkpoint_name

    if world_model is None:
        return "Error: No model loaded. Please load a session first.", gr.Dropdown()

    if not checkpoint_name or checkpoint_name.strip() == "":
        return "Error: Please provide a checkpoint name.", gr.Dropdown()

    checkpoint_name = checkpoint_name.strip()

    # Add .pth extension if not present
    if not checkpoint_name.endswith('.pth'):
        checkpoint_name += '.pth'

    checkpoint_path = os.path.join(TOROIDAL_DOT_CHECKPOINT_DIR, checkpoint_name)

    try:
        # Save model state dict, optimizer state, scheduler state, and metadata
        checkpoint = {
            'model_state_dict': world_model.autoencoder.state_dict(),
            'optimizer_state_dict': world_model.ae_optimizer.state_dict(),
            'scheduler_state_dict': world_model.ae_scheduler.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'frame_size': config.AutoencoderConcatPredictorWorldModelConfig.FRAME_SIZE,
                'separator_width': config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
                'canvas_history_size': config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE,
            }
        }

        # Add training metrics if available
        if hasattr(world_model, '_metrics_history') and world_model._metrics_history['iteration']:
            checkpoint['metrics'] = {
                'total_iterations': len(world_model._metrics_history['iteration']),
                'final_training_loss': world_model._metrics_history['training_loss'][-1] if world_model._metrics_history['training_loss'] else None,
                'final_prediction_error': world_model._metrics_history['prediction_error'][-1] if world_model._metrics_history['prediction_error'] else None,
            }

        torch.save(checkpoint, checkpoint_path)
        current_checkpoint_name = checkpoint_name

        status_msg = f"✅ **Model weights saved successfully!**\n\n"
        status_msg += f"**Checkpoint:** {checkpoint_name}\n"
        status_msg += f"**Location:** {checkpoint_path}\n"
        status_msg += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        if 'metrics' in checkpoint:
            status_msg += f"\n**Training Metrics:**\n"
            status_msg += f"- Total iterations: {checkpoint['metrics']['total_iterations']}\n"
            if checkpoint['metrics']['final_training_loss'] is not None:
                status_msg += f"- Final training loss: {format_loss(checkpoint['metrics']['final_training_loss'])}\n"
            if checkpoint['metrics']['final_prediction_error'] is not None:
                status_msg += f"- Final prediction error: {format_loss(checkpoint['metrics']['final_prediction_error'])}\n"

        return status_msg, refresh_checkpoints()

    except Exception as e:
        import traceback
        error_msg = f"❌ **Error saving checkpoint:**\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, gr.Dropdown()

def load_model_weights(checkpoint_name):
    """Load model weights from a checkpoint file"""
    global world_model, current_checkpoint_name

    if world_model is None:
        return "Error: No model loaded. Please load a session first."

    if not checkpoint_name or checkpoint_name == "No checkpoints available":
        return "Error: Please select a valid checkpoint."

    checkpoint_path = os.path.join(TOROIDAL_DOT_CHECKPOINT_DIR, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        return f"Error: Checkpoint file not found: {checkpoint_path}"

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load model state
        if 'model_state_dict' in checkpoint:
            world_model.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback: assume entire checkpoint is the state dict
            world_model.autoencoder.load_state_dict(checkpoint)

        # Track which components were loaded
        optimizer_loaded = False
        scheduler_loaded = False
        warnings = []

        # Try to load optimizer state if available (may fail if parameter groups differ)
        if 'optimizer_state_dict' in checkpoint:
            try:
                world_model.ae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_loaded = True
            except (ValueError, KeyError) as e:
                warnings.append(f"⚠️ Optimizer state not loaded (parameter group mismatch): {str(e)}")
                print(f"[LOAD WARNING] Skipping optimizer state: {str(e)}")

        # Try to load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            try:
                world_model.ae_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scheduler_loaded = True
            except (ValueError, KeyError) as e:
                warnings.append(f"⚠️ Scheduler state not loaded: {str(e)}")
                print(f"[LOAD WARNING] Skipping scheduler state: {str(e)}")

        current_checkpoint_name = checkpoint_name

        status_msg = f"✅ **Model weights loaded successfully!**\n\n"
        status_msg += f"**Checkpoint:** {checkpoint_name}\n"
        status_msg += f"**Location:** {checkpoint_path}\n"

        # Show what was loaded
        status_msg += f"\n**Loaded Components:**\n"
        status_msg += f"- Model weights: ✅\n"
        status_msg += f"- Optimizer state: {'✅' if optimizer_loaded else '❌ (skipped)'}\n"
        status_msg += f"- Scheduler state: {'✅' if scheduler_loaded else '❌ (skipped)'}\n"

        # Add warnings if any
        if warnings:
            status_msg += f"\n**Warnings:**\n"
            for warning in warnings:
                status_msg += f"{warning}\n"
            status_msg += f"\n*Note: Model weights loaded successfully. Optimizer/scheduler will be reinitialized on next training run.*\n"

        if 'timestamp' in checkpoint:
            status_msg += f"\n**Saved at:** {checkpoint['timestamp']}\n"

        # Show loss and samples_seen for auto-saved best models
        if 'loss' in checkpoint:
            status_msg += f"**Loss at save:** {checkpoint['loss']:.6f}\n"
        if 'samples_seen' in checkpoint:
            status_msg += f"**Samples seen:** {checkpoint['samples_seen']:,}\n"

        if 'config' in checkpoint:
            status_msg += f"\n**Model Configuration:**\n"
            status_msg += f"- Frame size: {checkpoint['config'].get('frame_size')}\n"
            status_msg += f"- Separator width: {checkpoint['config'].get('separator_width')}\n"
            status_msg += f"- Canvas history size: {checkpoint['config'].get('canvas_history_size')}\n"

        if 'metrics' in checkpoint:
            status_msg += f"\n**Training Metrics (at save time):**\n"
            status_msg += f"- Total iterations: {checkpoint['metrics']['total_iterations']}\n"
            if checkpoint['metrics']['final_training_loss'] is not None:
                status_msg += f"- Final training loss: {format_loss(checkpoint['metrics']['final_training_loss'])}\n"
            if checkpoint['metrics']['final_prediction_error'] is not None:
                status_msg += f"- Final prediction error: {format_loss(checkpoint['metrics']['final_prediction_error'])}\n"

        return status_msg

    except Exception as e:
        import traceback
        error_msg = f"❌ **Error loading checkpoint:**\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg

def get_checkpoint_info():
    """Get current checkpoint status information"""
    global current_checkpoint_name

    if current_checkpoint_name:
        return f"**Current checkpoint:** {current_checkpoint_name}"
    else:
        return "**Current checkpoint:** None (using fresh model)"

# Build Gradio interface
with gr.Blocks(title="Concat World Model Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Concat World Model Explorer")
    gr.Markdown("Run AutoencoderConcatPredictorWorldModel on recorded toroidal dot sessions.")

    # Session Selection
    with gr.Row():
        session_dropdown = gr.Dropdown(label="Session", choices=[], interactive=True)
        refresh_btn = gr.Button("🔄 Refresh", size="sm")
        load_session_btn = gr.Button("Load Session", variant="primary")

    session_info = gr.Markdown("No session loaded.")

    gr.Markdown("---")

    # Model Checkpoint Management
    gr.Markdown("## Model Checkpoint Management")
    gr.Markdown("Save and load trained model weights.")

    checkpoint_status = gr.Markdown("**Current checkpoint:** None (using fresh model)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Save Model Weights")
            save_checkpoint_name = gr.Textbox(
                label="Checkpoint Name",
                placeholder="e.g., my_model_checkpoint",
                info="Extension .pth will be added automatically"
            )
            save_checkpoint_btn = gr.Button("💾 Save Weights", variant="primary")
            save_checkpoint_status = gr.Markdown("")

        with gr.Column(scale=1):
            gr.Markdown("### Load Model Weights")
            checkpoint_dropdown = gr.Dropdown(
                label="Select Checkpoint",
                choices=[],
                interactive=True
            )
            with gr.Row():
                refresh_checkpoints_btn = gr.Button("🔄 Refresh", size="sm")
                load_checkpoint_btn = gr.Button("📂 Load Weights", variant="primary", scale=2)
            load_checkpoint_status = gr.Markdown("")

    gr.Markdown("---")

    # Frame Viewer
    gr.Markdown("## Current Frame")
    with gr.Row():
        with gr.Column(scale=2):
            frame_image = gr.Image(label="Current Frame", type="pil", interactive=False)
        with gr.Column(scale=1):
            frame_info = gr.Markdown("Load a session to view frames.")

    # Frame Navigation
    with gr.Row():
        frame_slider = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Frame", interactive=True)

    with gr.Row():
        frame_number_input = gr.Number(value=0, label="Jump to Frame", precision=0)
        jump_btn = gr.Button("Jump", size="sm")

    gr.Markdown("---")

    # Single Canvas Inference
    gr.Markdown("## Inference on Single Canvas")
    gr.Markdown("Run inference (no training) on a canvas built from the selected frame and its history to see what the model predicts.")

    with gr.Row():
        single_canvas_infer_btn = gr.Button("🔍 Run Inference on Selected Frame", variant="secondary")

    single_canvas_infer_status = gr.Markdown("")

    # Single Canvas Inference Visualizations (collapsible)
    with gr.Accordion("Single Canvas Inference Results", open=False):
        single_canvas_inference_info = gr.Markdown("")
        single_canvas_inference_canvas = gr.Plot(label="1. Inference Canvas")
        single_canvas_inference_masked = gr.Plot(label="2. Canvas with Mask Overlay")
        single_canvas_inference_full = gr.Plot(label="3. Full Inpainting Output")
        single_canvas_inference_composite = gr.Plot(label="4. Composite Reconstruction")

    gr.Markdown("---")

    # Full Session Evaluation
    gr.Markdown("## Full Session Evaluation")
    gr.Markdown("Evaluate model performance across all observations in the session to get objective metrics for model comparison.")

    with gr.Row():
        evaluate_session_btn = gr.Button("📊 Evaluate Model on Full Session", variant="primary")

    eval_status = gr.Markdown("")

    # Full Session Evaluation Results (collapsible)
    with gr.Accordion("Full Session Evaluation Results", open=False):
        eval_statistics = gr.Markdown("")
        eval_loss_over_time = gr.Plot(label="Loss Over Observations")
        eval_distribution = gr.Plot(label="Loss Distribution")

    gr.Markdown("---")

    # Single Canvas Training
    gr.Markdown("## Train on Single Canvas")
    gr.Markdown("Train the autoencoder on a canvas built from the selected frame and its history.")

    with gr.Row():
        single_canvas_training_steps = gr.Number(value=10, label="Training Steps", precision=0, minimum=1)
        single_canvas_train_btn = gr.Button("Train on Selected Frame", variant="primary")

    single_canvas_status = gr.Markdown("")

    # Single Canvas Training Visualizations (collapsible)
    with gr.Accordion("Single Canvas Training Results", open=False):
        single_canvas_training_info = gr.Markdown("")
        single_canvas_grad_diag = gr.Markdown("")
        single_canvas_loss_history = gr.Plot(label="Training Loss History")
        single_canvas_training_canvas = gr.Plot(label="1. Training Canvas")
        single_canvas_training_masked = gr.Plot(label="2. Canvas with Mask Overlay")
        single_canvas_inpainting_full = gr.Plot(label="3. Full Inpainting Output")
        single_canvas_inpainting_composite = gr.Plot(label="4. Composite Reconstruction")

    gr.Markdown("---")

    # ========== BATCH TRAINING SECTION ==========
    gr.Markdown("---")
    gr.Markdown("## Run World Model (Batch Training)")
    gr.Markdown("Train the model on batches of observations with periodic full-session evaluation.")

    with gr.Row():
        total_samples_input = gr.Number(
            value=10000000,
            label="Total Samples",
            precision=0,
            minimum=1,
            interactive=True,
            info="Number of training samples (loops through session if needed)"
        )
        batch_size_input = gr.Number(
            value=64,
            label="Batch Size",
            precision=0,
            minimum=1,
            interactive=True,
            info="Samples per batch"
        )
        update_interval_input = gr.Number(
            value=2000,
            label="Update Interval",
            precision=0,
            minimum=10,
            interactive=True,
            info="Evaluate every N samples (lower = more frequent updates)"
        )
        window_size_input = gr.Number(
            value=500,
            label="Rolling Window Size",
            precision=0,
            minimum=1,
            interactive=True,
            info="Number of recent checkpoints to show in rolling window graph"
        )
        num_random_obs_input = gr.Number(
            value=2,
            label="Random Observations to Visualize",
            precision=0,
            minimum=1,
            maximum=20,
            interactive=True,
            info="Number of random observations to sample and visualize on each update"
        )
        num_best_models_input = gr.Number(
            value=3,
            label="Best Models to Keep",
            precision=0,
            minimum=1,
            maximum=10,
            interactive=True,
            info="Maximum number of best model checkpoints to keep (auto-deletes worse models)"
        )

    # Training info display (dynamically updated)
    training_info_display = gr.Markdown("")

    with gr.Row():
        run_batch_btn = gr.Button("🚀 Run Batch Training", variant="primary")

    batch_training_status = gr.Markdown("")

    gr.Markdown("---")

    # Training Progress
    gr.Markdown("## Training Progress")

    with gr.Row():
        with gr.Column(scale=1):
            loss_vs_samples_plot = gr.Plot(label="Loss vs Samples Seen (Full History)")
        with gr.Column(scale=1):
            loss_vs_recent_plot = gr.Plot(label="Loss vs Recent Checkpoints (Rolling Window)")

    gr.Markdown("---")

    # Training Observation Samples
    gr.Markdown("---")

    gr.Markdown("## Training Observation Samples")
    gr.Markdown("Random observations + current frame, re-sampled on each training update.")

    observation_samples_status = gr.Markdown("")
    observation_samples_plot = gr.Plot(label="Observation Samples (Original + Composite Grid)")

    gr.Markdown("---")

    # Latest Evaluation Results
    gr.Markdown("## Latest Evaluation Results")
    gr.Markdown("Most recent full-session evaluation from training.")

    latest_eval_loss_plot = gr.Plot(label="Loss Over Session")
    latest_eval_dist_plot = gr.Plot(label="Loss Distribution")

    gr.Markdown("---")

    # Attention Visualization
    gr.Markdown("## Decoder Attention Visualization")
    gr.Markdown("Visualize decoder attention FROM selected patches TO all other patches.")
    gr.Markdown("*Note: Uses the currently selected frame to build a canvas and run inference to get attention.*")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Controls")

            # Patch selection mode
            attn_selection_mode = gr.Radio(
                choices=["Automatic Dot Detection", "Manual Selection"],
                value="Automatic Dot Detection",
                label="Patch Selection Mode"
            )

            # Brightness threshold for automatic detection
            attn_brightness_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Brightness Threshold (for automatic detection)",
                info="Minimum avg brightness (0-1) to detect dot patches"
            )

            # Manual patch selection
            attn_manual_patches = gr.Textbox(
                label="Manual Patch Indices (for manual selection)",
                placeholder="e.g., 0,5,10 or 0-10",
                info="Comma-separated indices or ranges (e.g., 0,5,10-15)"
            )

            # Visualization type
            attn_viz_type = gr.Radio(
                choices=["Patch-to-Patch Lines", "Heatmap Matrix", "Heatmap Overlay on Frame"],
                value="Patch-to-Patch Lines",
                label="Visualization Type"
            )

            # Aggregation method
            attn_aggregation = gr.Radio(
                choices=["mean", "max", "sum"],
                value="mean",
                label="Head Aggregation Method"
            )

            # Selected patch aggregation method
            attn_selected_aggregation = gr.Radio(
                choices=["mean", "max", "sum"],
                value="mean",
                label="Selected Patch Aggregation Method",
                info="How to aggregate attention from multiple selected patches"
            )

            # Quantile slider
            attn_quantile = gr.Slider(
                minimum=0.0,
                maximum=100.0,
                value=95.0,
                step=1.0,
                label="Attention Quantile (%)",
                info="Show top N% of connections (e.g., 95 = show strongest 5%)"
            )

            # Layer toggles
            gr.Markdown("**Select Layers to Display:**")
            attn_layer0 = gr.Checkbox(label="Layer 0", value=True)
            attn_layer1 = gr.Checkbox(label="Layer 1", value=True)
            attn_layer2 = gr.Checkbox(label="Layer 2", value=True)
            attn_layer3 = gr.Checkbox(label="Layer 3", value=True)
            attn_layer4 = gr.Checkbox(label="Layer 4", value=True)

            # Head toggles
            gr.Markdown("**Select Attention Heads to Display:**")
            attn_head0 = gr.Checkbox(label="Head 0", value=True)
            attn_head1 = gr.Checkbox(label="Head 1", value=True)
            attn_head2 = gr.Checkbox(label="Head 2", value=True)
            attn_head3 = gr.Checkbox(label="Head 3", value=True)

            # Generate button
            generate_attn_btn = gr.Button("Generate Attention Visualization", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Visualization")
            attn_status = gr.Markdown("")
            attn_plot = gr.Plot(label="Attention Visualization")
            attn_stats = gr.Markdown("")

    gr.Markdown("---")

    # Batch Size Comparison Testing
    gr.Markdown("## Batch Size Comparison")
    gr.Markdown(
        "Compare training efficiency across different batch sizes. "
        "Each test trains over the same total number of samples for fair comparison."
    )

    with gr.Row():
        with gr.Column(scale=1):
            batch_sizes_input = gr.Textbox(
                value="1,2,4,8,16",
                label="Batch Sizes to Test",
                info="Comma-separated (e.g., 1,2,4,8,16)"
            )
            comparison_total_samples_input = gr.Number(
                value=1000,
                label="Total Samples Per Test",
                precision=0,
                minimum=100,
                info="Same for all batch sizes (fair comparison)"
            )
            run_comparison_btn = gr.Button("🔬 Run Batch Comparison", variant="primary")

        with gr.Column(scale=1):
            comparison_status = gr.Markdown("")

    with gr.Accordion("Batch Comparison Results", open=False):
        comparison_summary = gr.Markdown("")
        comparison_time_plot = gr.Plot(label="Total Training Time by Batch Size")
        comparison_quality_plot = gr.Plot(label="Final Loss by Batch Size")
        comparison_convergence_plot = gr.Plot(label="Loss Convergence (All Batch Sizes)")
        comparison_table = gr.Dataframe(label="Detailed Results")

    gr.Markdown("---")

    # Event handlers
    refresh_btn.click(
        fn=refresh_sessions,
        inputs=[],
        outputs=[session_dropdown]
    )

    load_session_btn.click(
        fn=load_session,
        inputs=[session_dropdown],
        outputs=[session_info, frame_image, frame_info, frame_slider, frame_slider]
    )

    # Checkpoint management event handlers
    save_checkpoint_btn.click(
        fn=save_model_weights,
        inputs=[save_checkpoint_name],
        outputs=[save_checkpoint_status, checkpoint_dropdown]
    )

    refresh_checkpoints_btn.click(
        fn=refresh_checkpoints,
        inputs=[],
        outputs=[checkpoint_dropdown]
    )

    load_checkpoint_btn.click(
        fn=load_model_weights,
        inputs=[checkpoint_dropdown],
        outputs=[load_checkpoint_status]
    )

    frame_slider.change(
        fn=update_frame,
        inputs=[frame_slider],
        outputs=[frame_image, frame_info]
    )

    def jump_to_frame(frame_num):
        frame_num = int(frame_num)
        img, info = update_frame(frame_num)
        return img, info, frame_num

    jump_btn.click(
        fn=jump_to_frame,
        inputs=[frame_number_input],
        outputs=[frame_image, frame_info, frame_slider]
    )

    single_canvas_infer_btn.click(
        fn=infer_on_single_canvas,
        inputs=[frame_slider],
        outputs=[
            single_canvas_infer_status,
            single_canvas_inference_info,
            single_canvas_inference_canvas,
            single_canvas_inference_masked,
            single_canvas_inference_full,
            single_canvas_inference_composite,
        ]
    )

    # Note: evaluate_full_session now returns 5 values, but we only use 4 in the UI
    # The 5th value (stats dict) is used internally by run_world_model_batch
    evaluate_session_btn.click(
        fn=lambda: evaluate_full_session()[:4],  # Take only first 4 return values
        inputs=[],
        outputs=[
            eval_status,
            eval_loss_over_time,
            eval_distribution,
            eval_statistics,
        ]
    )

    single_canvas_train_btn.click(
        fn=train_on_single_canvas,
        inputs=[frame_slider, single_canvas_training_steps],
        outputs=[
            single_canvas_status,
            single_canvas_training_info,
            single_canvas_grad_diag,
            single_canvas_loss_history,
            single_canvas_training_canvas,
            single_canvas_training_masked,
            single_canvas_inpainting_full,
            single_canvas_inpainting_composite,
        ]
    )

    # Dynamic training info update handlers
    total_samples_input.change(
        fn=calculate_training_info,
        inputs=[total_samples_input, batch_size_input],
        outputs=[training_info_display]
    )

    batch_size_input.change(
        fn=calculate_training_info,
        inputs=[total_samples_input, batch_size_input],
        outputs=[training_info_display]
    )

    # Batch training handler (takes current slider value as input)
    run_batch_btn.click(
        fn=run_world_model_batch,
        inputs=[total_samples_input, batch_size_input, frame_slider, update_interval_input,
                window_size_input, num_random_obs_input, num_best_models_input],
        outputs=[
            batch_training_status,
            loss_vs_samples_plot,
            loss_vs_recent_plot,
            latest_eval_loss_plot,
            latest_eval_dist_plot,
            observation_samples_status,
            observation_samples_plot,
        ]
    )

    generate_attn_btn.click(
        fn=generate_attention_visualization,
        inputs=[
            frame_slider,
            attn_selection_mode,
            attn_brightness_threshold,
            attn_manual_patches,
            attn_quantile,
            attn_layer0,
            attn_layer1,
            attn_layer2,
            attn_layer3,
            attn_layer4,
            attn_head0,
            attn_head1,
            attn_head2,
            attn_head3,
            attn_aggregation,
            attn_selected_aggregation,
            attn_viz_type,
        ],
        outputs=[
            attn_status,
            attn_plot,
            attn_stats,
        ]
    )

    run_comparison_btn.click(
        fn=run_batch_comparison,
        inputs=[batch_sizes_input, comparison_total_samples_input],
        outputs=[
            comparison_status,
            comparison_summary,
            comparison_time_plot,
            comparison_quality_plot,
            comparison_convergence_plot,
            comparison_table,
        ]
    )

    # Initialize session dropdown and checkpoint dropdown on load
    def initialize_ui():
        sessions = refresh_sessions()
        checkpoints = refresh_checkpoints()
        # Initialize training info with default values
        training_info = calculate_training_info(10000000, 64)  
        return sessions, checkpoints, training_info

    demo.load(
        fn=initialize_ui,
        inputs=[],
        outputs=[session_dropdown, checkpoint_dropdown, training_info_display]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
