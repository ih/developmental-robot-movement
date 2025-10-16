"""
Session Explorer Gradio App

A web-based UI for exploring recorded robot sessions, running autoencoder and predictor models,
and training models on selected frames/history. This is a Gradio port of session_explorer.ipynb.
"""

import os
import io
import json
import glob
import datetime
import asyncio
import time
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import gradio as gr
from tqdm import tqdm

import config
from models import MaskedAutoencoderViT, TransformerActionConditionedPredictor
from adaptive_world_model import AdaptiveWorldModel, normalize_action_dicts
from robot_interface import RobotInterface
from session_explorer_lib import *

# Session base directories - scan both robot types
SESSIONS_ROOT_DIR = os.path.join(config.AUX_DIR, "sessions")
JETBOT_SESSIONS_DIR = config.JETBOT_RECORDING_DIR
TOROIDAL_DOT_SESSIONS_DIR = config.TOROIDAL_DOT_RECORDING_DIR

# Checkpoint directories - will be set based on selected session's robot type
JETBOT_CHECKPOINT_DIR = config.JETBOT_CHECKPOINT_DIR
TOROIDAL_DOT_CHECKPOINT_DIR = config.TOROIDAL_DOT_CHECKPOINT_DIR

# Weights & Biases configuration
WANDB_PROJECT = ""  # Set to project name to enable wandb logging, or leave empty to disable

# Default model paths (updated when session is loaded based on robot type)
DEFAULT_AUTOENCODER_PATH = ""
DEFAULT_PREDICTOR_PATH = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global state
session_state = {}
training_signals = None
adaptive_world_model = None
current_robot_type = None
current_checkpoint_dir = None
stub_robot = DynamicStubRobotInterface()

# Training control state
training_control = {
    "autoencoder_paused": False,
    "predictor_paused": False,
    "autoencoder_task": None,
    "predictor_task": None,
}

class TrainerSignals:
    """Signals for pause/resume/stop training"""
    def __init__(self):
        self.autoencoder_pause = asyncio.Event()
        self.autoencoder_stop = asyncio.Event()
        self.autoencoder_is_paused = False
        self.predictor_pause = asyncio.Event()
        self.predictor_stop = asyncio.Event()
        self.predictor_is_paused = False

    def reset_autoencoder(self):
        self.autoencoder_pause.set()
        self.autoencoder_stop.clear()
        self.autoencoder_is_paused = False

    def reset_predictor(self):
        self.predictor_pause.set()
        self.predictor_stop.clear()
        self.predictor_is_paused = False

training_signals = TrainerSignals()
training_signals.reset_autoencoder()
training_signals.reset_predictor()

def format_loss(loss_value):
    """Format loss value for display"""
    if loss_value < 0.001:
        return f"{loss_value:.2e}"
    else:
        return f"{loss_value:.6f}"

def refresh_sessions():
    """Refresh session list"""
    sessions = list_session_dirs()
    choices = [os.path.basename(s) + " - " + s for s in sessions]
    return gr.Dropdown(choices=choices, value=choices[-1] if choices else None)

def load_session(session_choice):
    """Load a session from dropdown selection"""
    global adaptive_world_model, current_robot_type, current_checkpoint_dir
    global DEFAULT_AUTOENCODER_PATH, DEFAULT_PREDICTOR_PATH

    if not session_choice:
        return "No session selected", None, "", 0, 0, "", ""

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
    session_state["action_space"] = get_action_space(session_state)

    # Update stub robot and adaptive world model based on session's robot type
    robot_type = metadata.get("robot_type", "unknown")
    action_space = metadata.get("action_space", [])

    stub_robot.set_action_space(action_space)

    # Determine checkpoint directory based on robot type
    if "toroidal" in robot_type.lower():
        checkpoint_dir = TOROIDAL_DOT_CHECKPOINT_DIR
        config.ACTION_CHANNELS = config.ToroidalDotConfig.ACTION_CHANNELS_DOT
        config.ACTION_RANGES = config.ToroidalDotConfig.ACTION_RANGES_DOT
    elif "jetbot" in robot_type.lower():
        checkpoint_dir = JETBOT_CHECKPOINT_DIR
    else:
        # Unknown robot type - try to infer from action space
        if action_space and "action" in action_space[0]:
            checkpoint_dir = TOROIDAL_DOT_CHECKPOINT_DIR
            config.ACTION_CHANNELS = config.ToroidalDotConfig.ACTION_CHANNELS_DOT
            config.ACTION_RANGES = config.ToroidalDotConfig.ACTION_RANGES_DOT
        else:
            checkpoint_dir = JETBOT_CHECKPOINT_DIR

    # Update default model paths
    DEFAULT_AUTOENCODER_PATH = os.path.join(checkpoint_dir, "autoencoder.pth")
    DEFAULT_PREDICTOR_PATH = os.path.join(checkpoint_dir, "predictor_0.pth")

    # Recreate AdaptiveWorldModel if needed
    if adaptive_world_model is None or current_robot_type != robot_type or current_checkpoint_dir != checkpoint_dir:
        current_robot_type = robot_type
        current_checkpoint_dir = checkpoint_dir
        adaptive_world_model = AdaptiveWorldModel(
            stub_robot,
            wandb_project=WANDB_PROJECT,
            checkpoint_dir=checkpoint_dir,
            interactive=False
        )

    # Build session info
    if not observations:
        info = f"**{session_state['session_name']}** has no observation frames."
        return info, None, "", 0, 0, DEFAULT_AUTOENCODER_PATH, DEFAULT_PREDICTOR_PATH

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
    frame_info = f"**Observation 1 / {len(observations)}**\nStep: {observations[0]['step']}\nTimestamp: {format_timestamp(observations[0]['timestamp'])}"

    max_frames = len(observations) - 1

    return info, first_frame, frame_info, 0, max_frames, DEFAULT_AUTOENCODER_PATH, DEFAULT_PREDICTOR_PATH

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

def load_models(autoencoder_path, predictor_path):
    """Load model checkpoints"""
    messages = []

    if autoencoder_path and os.path.exists(autoencoder_path):
        try:
            loaded_autoencoder = load_autoencoder_model(autoencoder_path, device)
            adaptive_world_model.autoencoder = loaded_autoencoder
            messages.append(f"✓ Autoencoder loaded from `{autoencoder_path}`.")
        except Exception as exc:
            messages.append(f"✗ Failed to load autoencoder: {exc}")
    elif autoencoder_path:
        messages.append(f"✗ Autoencoder path not found: {autoencoder_path}")
    else:
        messages.append("Autoencoder path is empty; skipping load.")

    if predictor_path and os.path.exists(predictor_path):
        try:
            loaded_predictor = load_predictor_model(predictor_path, device)
            if len(adaptive_world_model.predictors) == 0:
                adaptive_world_model.predictors.append(loaded_predictor)
            else:
                adaptive_world_model.predictors[0] = loaded_predictor
            messages.append(f"✓ Predictor loaded from `{predictor_path}`.")
        except Exception as exc:
            messages.append(f"✗ Failed to load predictor: {exc}")
    elif predictor_path:
        messages.append(f"✗ Predictor path not found: {predictor_path}")
    else:
        messages.append("Predictor path is empty; skipping load.")

    return "\n\n".join(messages)

def run_autoencoder(frame_idx):
    """Run autoencoder inference"""
    if adaptive_world_model.autoencoder is None:
        return None, "Load the autoencoder checkpoint first."

    if not session_state.get("observations"):
        return None, "Load a session first."

    observations = session_state["observations"]
    if frame_idx >= len(observations):
        return None, "Invalid frame index."

    obs = observations[frame_idx]
    frame_tensor = get_frame_tensor(session_state["session_dir"], obs["frame_path"]).unsqueeze(0).to(device)

    adaptive_world_model.autoencoder.eval()
    with torch.no_grad():
        reconstructed = adaptive_world_model.autoencoder.reconstruct(frame_tensor)
    mse = F.mse_loss(reconstructed, frame_tensor).item()

    original_img = tensor_to_numpy_image(frame_tensor)
    reconstructed_img = tensor_to_numpy_image(reconstructed)

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(reconstructed_img)
    axes[1].set_title(f"Reconstruction\nMSE: {format_loss(mse)}")
    axes[1].axis("off")
    plt.tight_layout()

    return fig, f"**Reconstruction Loss:** {format_loss(mse)}"

def run_predictor(frame_idx, history_length):
    """Run predictor inference"""
    if adaptive_world_model.autoencoder is None or not adaptive_world_model.predictors:
        return None, None, "Load both autoencoder and predictor checkpoints first."

    if not session_state.get("observations"):
        return None, None, "Load a session first."

    predictor = adaptive_world_model.predictors[0]
    autoencoder = adaptive_world_model.autoencoder

    target_idx = frame_idx
    selected_obs, action_dicts, error = build_predictor_sequence(session_state, target_idx, history_length)

    if error:
        return None, None, f"**Cannot run predictor:** {error}"

    if len(selected_obs) < 2:
        return None, None, "**Cannot run predictor:** Need at least 2 frames."

    past_obs = selected_obs[:-1]
    current_obs = selected_obs[-1]

    if len(action_dicts) < 1:
        return None, None, "**Cannot run predictor:** Need at least one action in history."

    past_actions = action_dicts[:-1] if len(action_dicts) > 1 else []
    recorded_current_action = action_dicts[-1]

    # Encode past frames
    past_feature_history = []
    for obs in past_obs:
        tensor = get_frame_tensor(session_state["session_dir"], obs["frame_path"]).unsqueeze(0).to(device)
        autoencoder.eval()
        with torch.no_grad():
            encoded = autoencoder.encode(tensor).detach()
        past_feature_history.append(encoded)

    # Get actual current frame
    current_tensor = get_frame_tensor(session_state["session_dir"], current_obs["frame_path"]).unsqueeze(0).to(device)

    # Predict current frame
    actions_for_current = past_actions + [clone_action(recorded_current_action)]
    current_action_norm = normalize_action_dicts([recorded_current_action]).to(device)
    last_past_features = past_feature_history[-1] if past_feature_history else None

    predictor.eval()
    with torch.no_grad():
        predicted_current_features = predictor(
            past_feature_history,
            actions_for_current,
            action_normalized=current_action_norm,
            last_features=last_past_features
        )
        predicted_current_tensor = decode_features_to_image(autoencoder, predicted_current_features)
        current_mse = F.mse_loss(predicted_current_tensor, current_tensor).item()

    predicted_current_img = tensor_to_numpy_image(predicted_current_tensor)
    current_img = tensor_to_numpy_image(current_tensor)

    # Create comparison plot
    fig1, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(current_img)
    axes[0].set_title(f"Actual Current (Frame {target_idx+1})")
    axes[0].axis("off")
    axes[1].imshow(predicted_current_img)
    axes[1].set_title(f"Predicted Current from Past\nMSE: {current_mse:.6f}")
    axes[1].axis("off")
    plt.tight_layout()

    # Action sweep
    action_space = get_action_space(session_state)

    if not action_space:
        return fig1, None, f"**Prediction Error (MSE):** {format_loss(current_mse)}\n\n**Past context:** {len(past_obs)} frames"

    counterfactual_predictions = []

    for action in action_space:
        actions_variant = past_actions + [clone_action(action)]
        variant_action_norm = normalize_action_dicts([action]).to(device)

        if actions_equal(action, recorded_current_action):
            cf_tensor = predicted_current_tensor
            cf_img = predicted_current_img
            cf_mse = current_mse
        else:
            with torch.no_grad():
                cf_features = predictor(
                    past_feature_history,
                    actions_variant,
                    action_normalized=variant_action_norm,
                    last_features=last_past_features
                )
                cf_tensor = decode_features_to_image(autoencoder, cf_features)
            cf_img = tensor_to_numpy_image(cf_tensor)
            with torch.no_grad():
                cf_mse = F.mse_loss(cf_tensor, current_tensor).item()

        counterfactual_predictions.append({
            "action": action,
            "image": cf_img,
            "label": format_action_label(action),
            "mse": cf_mse,
            "is_recorded": actions_equal(action, recorded_current_action)
        })

    # Grid of counterfactual predictions
    cols = min(4, len(counterfactual_predictions))
    rows = math.ceil(len(counterfactual_predictions) / cols)
    fig2, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(rows, cols) if rows > 1 or cols > 1 else np.array([[axes]])

    for idx, pred in enumerate(counterfactual_predictions):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(pred["image"])
        title = pred["label"]
        if pred["is_recorded"]:
            title += " (recorded)"
        title += f"\nMSE: {pred['mse']:.6f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(len(counterfactual_predictions), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.tight_layout()

    info = f"**Prediction Error (MSE):** {format_loss(current_mse)}\n\n**Past context:** {len(past_obs)} frames"

    return fig1, fig2, info

def train_autoencoder_step_wrapper(frame_tensor):
    """Single autoencoder training step using AdaptiveWorldModel"""
    frame_numpy = tensor_to_numpy_image(frame_tensor)
    loss = adaptive_world_model.train_autoencoder(frame_numpy)
    return loss

def train_autoencoder_threshold(frame_idx, threshold, max_steps, progress=gr.Progress()):
    """Train autoencoder to threshold"""
    if adaptive_world_model.autoencoder is None:
        return None, "Load the autoencoder checkpoint first.", "Error", "--"

    if not session_state.get("observations"):
        return None, "No session loaded.", "Error", "--"

    observations = session_state["observations"]
    if frame_idx >= len(observations):
        return None, "Invalid frame index.", "Error", "--"

    obs = observations[frame_idx]
    frame_tensor = get_frame_tensor(session_state["session_dir"], obs["frame_path"]).unsqueeze(0).to(device)

    losses = []
    training_signals.reset_autoencoder()

    progress(0, desc="Training autoencoder...")

    for step in range(max_steps):
        adaptive_world_model.autoencoder.train()
        loss = train_autoencoder_step_wrapper(frame_tensor)
        losses.append(loss)

        progress((step + 1) / max_steps, desc=f"Step {step + 1}/{max_steps}, Loss: {format_loss(loss)}")

        if loss <= threshold:
            break

    final_loss = losses[-1] if losses else None

    # Create training plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(losses)
    ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.7, label=f"Target: {format_loss(threshold)}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_title("Autoencoder Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    status = f"Completed in {len(losses)} steps"
    loss_display = format_loss(final_loss) if final_loss is not None else "--"
    info = f"**Training completed after {len(losses)} steps**\n\nFinal reconstruction loss: {format_loss(final_loss)}"

    if final_loss <= threshold:
        info += f"\n\n**Target threshold {format_loss(threshold)} achieved!**"
    else:
        info += f"\n\n**Target threshold {format_loss(threshold)} not reached after {max_steps} steps.**"

    return fig, info, status, loss_display

def train_autoencoder_steps(frame_idx, num_steps, progress=gr.Progress()):
    """Train autoencoder for fixed number of steps"""
    return train_autoencoder_threshold(frame_idx, 0.0, num_steps, progress)

def train_predictor_threshold(frame_idx, history_length, threshold, max_steps, progress=gr.Progress()):
    """Train predictor to threshold"""
    if adaptive_world_model.autoencoder is None or not adaptive_world_model.predictors:
        return None, "Load both autoencoder and predictor checkpoints first.", "Error", "--"

    if not session_state.get("observations"):
        return None, "No session loaded.", "Error", "--"

    observations = session_state["observations"]
    target_idx = frame_idx

    selected_obs, action_dicts, error = build_predictor_sequence(session_state, target_idx, history_length)
    if error:
        return None, f"**Cannot train predictor:** {error}", "Error", "--"

    if target_idx + 1 >= len(observations):
        return None, "**Cannot train predictor:** No next frame available.", "Error", "--"

    next_obs = observations[target_idx + 1]
    target_tensor = get_frame_tensor(session_state["session_dir"], next_obs["frame_path"]).unsqueeze(0).to(device)

    feature_history = []
    for obs in selected_obs:
        tensor = get_frame_tensor(session_state["session_dir"], obs["frame_path"]).unsqueeze(0).to(device)
        adaptive_world_model.autoencoder.eval()
        with torch.no_grad():
            encoded = adaptive_world_model.autoencoder.encode(tensor).detach()
        feature_history.append(encoded)

    recorded_future_action, _ = get_future_action_for_prediction(session_state, target_idx)
    if recorded_future_action is None:
        recorded_future_action = {}

    history_actions_with_future = [clone_action(action) for action in action_dicts]
    history_actions_with_future.append(clone_action(recorded_future_action))

    losses = []
    training_signals.reset_predictor()
    predictor = adaptive_world_model.predictors[0]

    progress(0, desc="Training predictor...")

    for step in range(max_steps):
        adaptive_world_model.predictors[0].train()
        adaptive_world_model.autoencoder.train()

        try:
            if history_actions_with_future:
                action_normalized = normalize_action_dicts([history_actions_with_future[-1]]).to(device)
            else:
                action_normalized = torch.zeros(1, len(config.ACTION_CHANNELS), device=device)

            last_features = feature_history[-1] if feature_history else None

            predicted_features = predictor(
                feature_history,
                history_actions_with_future,
                action_normalized=action_normalized,
                last_features=last_features
            )
            loss = adaptive_world_model.train_predictor(
                level=0,
                current_frame_tensor=target_tensor,
                predicted_features=predicted_features,
                history_features=feature_history,
                history_actions=history_actions_with_future,
            )
        except Exception as exc:
            return None, f"**Training error:** {exc}", "Error", "--"

        losses.append(loss)
        progress((step + 1) / max_steps, desc=f"Step {step + 1}/{max_steps}, Loss: {format_loss(loss)}")

        if loss <= threshold:
            break

    final_loss = losses[-1] if losses else None

    # Create training plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(losses, label="Total Loss")
    ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.7, label=f"Target: {format_loss(threshold)}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Predictor Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    status = f"Completed in {len(losses)} steps"
    loss_display = format_loss(final_loss) if final_loss is not None else "--"
    info = f"**Training completed after {len(losses)} steps**\n\nFinal total loss: {format_loss(final_loss)}"

    if final_loss <= threshold:
        info += f"\n\n**Target threshold {format_loss(threshold)} achieved!**"
    else:
        info += f"\n\n**Target threshold {format_loss(threshold)} not reached after {max_steps} steps.**"

    return fig, info, status, loss_display

def train_predictor_steps(frame_idx, history_length, num_steps, progress=gr.Progress()):
    """Train predictor for fixed number of steps"""
    return train_predictor_threshold(frame_idx, history_length, 0.0, num_steps, progress)

# Build Gradio interface
with gr.Blocks(title="Session Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Session Explorer")
    gr.Markdown("Explore recorded robot sessions, run models, and train on selected frames.")

    # Session Selection
    with gr.Row():
        session_dropdown = gr.Dropdown(label="Session", choices=[], interactive=True)
        refresh_btn = gr.Button("🔄 Refresh", size="sm")
        load_session_btn = gr.Button("Load Session", variant="primary")

    session_info = gr.Markdown("No session loaded.")

    # Frame Viewer
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

    # Model Checkpoints
    gr.Markdown("## Model Checkpoints")
    with gr.Row():
        autoencoder_path = gr.Textbox(label="Autoencoder Path", value=DEFAULT_AUTOENCODER_PATH, interactive=True)
    with gr.Row():
        predictor_path = gr.Textbox(label="Predictor Path", value=DEFAULT_PREDICTOR_PATH, interactive=True)

    load_models_btn = gr.Button("Load Models", variant="primary")
    model_status = gr.Markdown("Models not loaded.")

    gr.Markdown("---")

    # Autoencoder Inference
    gr.Markdown("## Autoencoder Inference")
    gr.Markdown("Uses the currently selected frame.")

    run_autoencoder_btn = gr.Button("Run Autoencoder", variant="success")
    autoencoder_output_plot = gr.Plot(label="Autoencoder Output")
    autoencoder_output_info = gr.Markdown("")

    gr.Markdown("---")

    # Predictor Inference
    gr.Markdown("## Predictor Inference")
    gr.Markdown("History uses frames leading up to the current selection to predict the next observation.")

    history_slider = gr.Slider(minimum=2, maximum=8, value=3, step=1, label="History Length")
    run_predictor_btn = gr.Button("Run Predictor", variant="primary")
    predictor_output_plot1 = gr.Plot(label="Prediction Comparison")
    predictor_output_plot2 = gr.Plot(label="Action Sweep")
    predictor_output_info = gr.Markdown("")

    gr.Markdown("---")

    # Autoencoder Training
    gr.Markdown("## Autoencoder Training")
    gr.Markdown("Train the autoencoder using AdaptiveWorldModel with randomized masking.")

    with gr.Row():
        autoencoder_threshold = gr.Number(value=0.001, label="Threshold")
        autoencoder_max_steps = gr.Number(value=1000, label="Max Steps", precision=0)

    with gr.Row():
        train_autoencoder_threshold_btn = gr.Button("Train to Threshold", variant="success")
        autoencoder_fixed_steps = gr.Number(value=100, label="Fixed Steps", precision=0)
        train_autoencoder_steps_btn = gr.Button("Train N Steps", variant="success")

    with gr.Row():
        autoencoder_status = gr.Textbox(label="Status", value="Idle", interactive=False)
        autoencoder_loss = gr.Textbox(label="Loss", value="--", interactive=False)

    autoencoder_training_plot = gr.Plot(label="Training Progress")
    autoencoder_training_info = gr.Markdown("")

    gr.Markdown("---")

    # Predictor Training
    gr.Markdown("## Predictor Training")
    gr.Markdown("Train the predictor using AdaptiveWorldModel with joint autoencoder training.")

    with gr.Row():
        predictor_threshold = gr.Number(value=0.001, label="Threshold")
        predictor_max_steps = gr.Number(value=1000, label="Max Steps", precision=0)

    with gr.Row():
        train_predictor_threshold_btn = gr.Button("Train to Threshold", variant="success")
        predictor_fixed_steps = gr.Number(value=100, label="Fixed Steps", precision=0)
        train_predictor_steps_btn = gr.Button("Train N Steps", variant="success")

    with gr.Row():
        predictor_status = gr.Textbox(label="Status", value="Idle", interactive=False)
        predictor_loss = gr.Textbox(label="Loss", value="--", interactive=False)

    predictor_training_plot = gr.Plot(label="Training Progress")
    predictor_training_info = gr.Markdown("")

    # Event handlers
    refresh_btn.click(
        fn=refresh_sessions,
        inputs=[],
        outputs=[session_dropdown]
    )

    load_session_btn.click(
        fn=load_session,
        inputs=[session_dropdown],
        outputs=[session_info, frame_image, frame_info, frame_slider, frame_slider, autoencoder_path, predictor_path]
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

    load_models_btn.click(
        fn=load_models,
        inputs=[autoencoder_path, predictor_path],
        outputs=[model_status]
    )

    run_autoencoder_btn.click(
        fn=run_autoencoder,
        inputs=[frame_slider],
        outputs=[autoencoder_output_plot, autoencoder_output_info]
    )

    run_predictor_btn.click(
        fn=run_predictor,
        inputs=[frame_slider, history_slider],
        outputs=[predictor_output_plot1, predictor_output_plot2, predictor_output_info]
    )

    train_autoencoder_threshold_btn.click(
        fn=train_autoencoder_threshold,
        inputs=[frame_slider, autoencoder_threshold, autoencoder_max_steps],
        outputs=[autoencoder_training_plot, autoencoder_training_info, autoencoder_status, autoencoder_loss]
    )

    train_autoencoder_steps_btn.click(
        fn=train_autoencoder_steps,
        inputs=[frame_slider, autoencoder_fixed_steps],
        outputs=[autoencoder_training_plot, autoencoder_training_info, autoencoder_status, autoencoder_loss]
    )

    train_predictor_threshold_btn.click(
        fn=train_predictor_threshold,
        inputs=[frame_slider, history_slider, predictor_threshold, predictor_max_steps],
        outputs=[predictor_training_plot, predictor_training_info, predictor_status, predictor_loss]
    )

    train_predictor_steps_btn.click(
        fn=train_predictor_steps,
        inputs=[frame_slider, history_slider, predictor_fixed_steps],
        outputs=[predictor_training_plot, predictor_training_info, predictor_status, predictor_loss]
    )

    # Initialize session dropdown on load
    demo.load(
        fn=refresh_sessions,
        inputs=[],
        outputs=[session_dropdown]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
