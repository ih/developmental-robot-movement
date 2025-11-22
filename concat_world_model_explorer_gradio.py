"""
Concat World Model Explorer Gradio App

A web-based UI for running AutoencoderConcatPredictorWorldModel on recorded robot sessions.
Allows running the world model for N iterations and visualizing training and prediction results.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import gradio as gr
from datetime import datetime
from pathlib import Path

import config
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
)
from recorded_policy import create_recorded_action_selector
from models.autoencoder_concat_predictor import (
    canvas_to_tensor,
    compute_patch_mask_for_last_slot,
    compute_randomized_patch_mask_for_last_slot,
    compute_hybrid_loss_on_masked_patches,
)
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

    session_state.update({
        "session_name": os.path.basename(session_dir),
        "session_dir": session_dir,
        "metadata": metadata,
        "events": events,
        "observations": observations,
        "actions": actions,
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

    return status_msg, fig_loss_over_time, fig_distribution, stats_text

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

def generate_attention_visualization(
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
    viz_type
):
    """Generate decoder attention visualization from the last training canvas"""
    global world_model

    if world_model is None:
        return "Please load a session and train/run the world model first", None, None

    if world_model.last_training_canvas is None or world_model.last_training_mask is None:
        return "No training canvas available. Please train or run the world model first.", None, None

    try:
        # Convert canvas to tensor
        canvas_tensor = canvas_to_tensor(world_model.last_training_canvas).to(device)
        patch_mask = world_model.last_training_mask.to(device)

        # Run forward pass with attention capture
        with torch.no_grad():
            _, _, attn_weights_list, patch_mask_out = world_model.autoencoder.forward_with_patch_mask(
                canvas_tensor,
                patch_mask,
                return_attn=True
            )

        # Get patch centers
        img_height, img_width = world_model.last_training_canvas.shape[:2]
        patch_size = config.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE
        patch_centers = compute_patch_centers(img_height, img_width, patch_size)

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

        # Generate visualization based on type
        if viz_type == "Patch-to-Patch Lines":
            fig = draw_attention_connections(
                canvas_img=world_model.last_training_canvas,
                patch_centers=patch_centers,
                attn_weights_list=attn_weights_list,
                patch_mask=patch_mask_out,
                quantile=quantile,
                enabled_layers=enabled_layers,
                enabled_heads=enabled_heads,
                aggregation=aggregation,
                alpha=0.6
            )
        else:  # Heatmap
            # For heatmap, show the first enabled layer
            layer_idx = 0
            for i, enabled in enumerate(enabled_layers):
                if enabled:
                    layer_idx = i
                    break

            fig = create_attention_heatmap(
                attn_weights_list=attn_weights_list,
                patch_mask=patch_mask_out,
                layer_idx=layer_idx,
                aggregation=aggregation,
                enabled_heads=enabled_heads
            )

        # Generate statistics
        stats = create_attention_statistics(
            attn_weights_list=attn_weights_list,
            patch_mask=patch_mask_out,
            quantile=quantile,
            aggregation=aggregation,
            enabled_heads=enabled_heads
        )

        # Format statistics for display
        heads_display = str(enabled_heads) if len(enabled_heads) < 4 else "All (0,1,2,3)"

        stats_text = f"**Attention Statistics:**\n\n"
        stats_text += f"- **Masked patches:** {stats['num_masked_patches']}\n"
        stats_text += f"- **Unmasked patches:** {stats['num_unmasked_patches']}\n"
        stats_text += f"- **Total connections:** {stats['total_connections']}\n"
        stats_text += f"- **Max attention weight:** {stats['max_layer_weight']:.4f}\n"
        stats_text += f"- **Quantile:** {stats['quantile']:.1f}%\n"
        stats_text += f"- **Avg threshold:** {stats['avg_threshold']:.4f}\n"
        stats_text += f"- **Aggregation:** {stats['aggregation']}\n"
        stats_text += f"- **Enabled heads:** {heads_display}\n\n"

        stats_text += "**Per-Layer Statistics:**\n\n"
        for layer_stats in stats['per_layer_stats']:
            layer_idx = layer_stats['layer']
            if enabled_layers[layer_idx]:
                stats_text += f"- **Layer {layer_idx}:**\n"
                stats_text += f"  - Connections: {layer_stats['num_connections']}\n"
                stats_text += f"  - Threshold: {layer_stats['threshold']:.4f}\n"
                stats_text += f"  - Max weight: {layer_stats['max_weight']:.4f}\n"
                stats_text += f"  - Mean weight: {layer_stats['mean_weight']:.4f}\n"
                stats_text += f"  - Total weight: {layer_stats['total_weight']:.4f}\n"

        status_msg = f"**Attention visualization generated successfully**\n\n"
        status_msg += f"Using last training canvas from world model\n"
        status_msg += f"Canvas size: {img_height}x{img_width} pixels\n"
        status_msg += f"Patch size: {patch_size}x{patch_size} pixels\n"
        status_msg += f"Quantile: {quantile:.1f}% (showing top {100-quantile:.1f}% of connections)"

        return status_msg, fig, stats_text

    except Exception as e:
        import traceback
        error_msg = f"Error generating attention visualization:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, ""

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

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            world_model.ae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            world_model.ae_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        current_checkpoint_name = checkpoint_name

        status_msg = f"✅ **Model weights loaded successfully!**\n\n"
        status_msg += f"**Checkpoint:** {checkpoint_name}\n"
        status_msg += f"**Location:** {checkpoint_path}\n"

        if 'timestamp' in checkpoint:
            status_msg += f"**Saved at:** {checkpoint['timestamp']}\n"

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

    # World Model Runner
    gr.Markdown("## Run World Model")
    gr.Markdown("Execute the world model for a specified number of iterations.")

    with gr.Row():
        num_iterations_input = gr.Number(value=10, label="Number of Iterations", precision=0, minimum=1)
        run_btn = gr.Button("Run World Model", variant="primary")

    run_status = gr.Markdown("")

    gr.Markdown("---")

    # Metrics Display
    gr.Markdown("## Iteration Metrics")
    current_metrics_display = gr.Markdown("")
    metrics_history_plot = gr.Plot(label="Metrics History")

    gr.Markdown("---")

    # Visualizations
    gr.Markdown("## Visualizations")

    gr.Markdown("### Current State")
    frames_plot = gr.Plot(label="Current Frame & Last Prediction")
    prediction_error_display = gr.Textbox(label="Prediction Error", value="--", interactive=False)

    gr.Markdown("### Training Results")
    training_info_display = gr.Markdown("")
    grad_diag_display = gr.Markdown("")
    training_canvas_plot = gr.Plot(label="1. Training Canvas (Original)")
    training_canvas_masked_plot = gr.Plot(label="2. Training Canvas with Mask Overlay")
    training_inpainting_full_plot = gr.Plot(label="3. Training Inpainting - Full Model Output")
    training_inpainting_composite_plot = gr.Plot(label="4. Training Inpainting - Composite")

    gr.Markdown("### Prediction Results")
    prediction_canvas_plot = gr.Plot(label="Prediction Canvas")
    predicted_frame_plot = gr.Plot(label="Predicted Next Frame")

    gr.Markdown("---")

    # Attention Visualization
    gr.Markdown("## Decoder Attention Visualization")
    gr.Markdown("Visualize decoder attention from masked patches to unmasked patches.")
    gr.Markdown("*Note: Uses the last training canvas from training or world model execution.*")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Controls")

            # Visualization type
            attn_viz_type = gr.Radio(
                choices=["Patch-to-Patch Lines", "Heatmap"],
                value="Patch-to-Patch Lines",
                label="Visualization Type"
            )

            # Aggregation method
            attn_aggregation = gr.Radio(
                choices=["mean", "max", "sum"],
                value="mean",
                label="Head Aggregation Method"
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

    evaluate_session_btn.click(
        fn=evaluate_full_session,
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

    run_btn.click(
        fn=run_world_model,
        inputs=[num_iterations_input],
        outputs=[
            run_status,
            current_metrics_display,
            metrics_history_plot,
            frames_plot,
            training_canvas_plot,
            training_canvas_masked_plot,
            training_inpainting_full_plot,
            training_inpainting_composite_plot,
            prediction_canvas_plot,
            predicted_frame_plot,
            training_info_display,
            grad_diag_display,
            prediction_error_display,
        ]
    )

    generate_attn_btn.click(
        fn=generate_attention_visualization,
        inputs=[
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
            attn_viz_type,
        ],
        outputs=[
            attn_status,
            attn_plot,
            attn_stats,
        ]
    )

    # Initialize session dropdown and checkpoint dropdown on load
    def initialize_ui():
        sessions = refresh_sessions()
        checkpoints = refresh_checkpoints()
        return sessions, checkpoints

    demo.load(
        fn=initialize_ui,
        inputs=[],
        outputs=[session_dropdown, checkpoint_dropdown]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
