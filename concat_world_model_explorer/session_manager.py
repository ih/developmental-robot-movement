"""
Session management functions for loading and navigating robot sessions.
"""

import os
import time
import gradio as gr

import config
from . import state
from . import checkpoint_manager
from .inference import get_counterfactual_action_choices
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


def get_robot_types():
    """Return list of available robot types."""
    return ["toroidal_dot", "jetbot", "so101"]


def get_sessions_dir_for_type(robot_type):
    """Return the sessions directory for a given robot type."""
    if robot_type == "jetbot":
        return config.JETBOT_RECORDING_DIR
    elif robot_type == "so101":
        return config.SO101_RECORDING_DIR
    else:  # default to toroidal_dot
        return config.TOROIDAL_DOT_RECORDING_DIR


def refresh_sessions_for_type(robot_type):
    """Refresh session list for a specific robot type."""
    from session_explorer_lib import list_session_dirs_from_base

    sessions_dir = get_sessions_dir_for_type(robot_type)
    sessions = list_session_dirs_from_base(sessions_dir)

    # Store the selected robot type in state
    state.selected_robot_type = robot_type

    # Create choices with just session names (since all are from same directory)
    choices = [os.path.basename(s) + " - " + s for s in sessions]
    return gr.Dropdown(choices=choices, value=choices[-1] if choices else None)


def refresh_sessions():
    """Refresh session list for currently selected robot type."""
    return refresh_sessions_for_type(state.selected_robot_type)


def get_validation_session_choices(robot_type=None):
    """Get session choices for validation dropdown with 'None' option at the start."""
    from session_explorer_lib import list_session_dirs_from_base

    if robot_type is None:
        robot_type = state.selected_robot_type

    sessions_dir = get_sessions_dir_for_type(robot_type)
    sessions = list_session_dirs_from_base(sessions_dir)

    # Add "None" as first option, then session choices
    choices = ["None - No validation"]
    choices.extend([os.path.basename(s) + " - " + s for s in sessions])
    return choices


def load_session(session_choice):
    """Load a session from dropdown selection"""
    if not session_choice:
        return "No session selected", None, "", gr.Dropdown(), gr.Radio()

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
    canvas_cache, detected_frame_size = prebuild_all_canvases(
        session_dir,
        observations,
        actions,
        config.AutoencoderConcatPredictorWorldModelConfig
    )
    prebuild_time = time.time() - prebuild_start
    print(f"Canvas pre-building completed in {prebuild_time:.2f}s")
    # Calculate canvas width based on detected frame size
    canvas_width = detected_frame_size[1] * 3 + config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH * 2
    print(f"Memory usage: ~{len(canvas_cache) * detected_frame_size[0] * canvas_width * 3 / (1024**2):.1f} MB")
    print("="*60 + "\n")

    # Store action space for robot-agnostic action handling
    action_space = metadata.get("action_space", [])

    # Reset checkpoint metadata when loading a new session (fresh start)
    state.reset_checkpoint_metadata()

    state.session_state.update({
        "session_name": os.path.basename(session_dir),
        "session_dir": session_dir,
        "metadata": metadata,
        "events": events,
        "observations": observations,
        "actions": actions,
        "canvas_cache": canvas_cache,
        "action_space": action_space,  # Added for JetBot support
        "detected_frame_size": detected_frame_size,  # (H, W) tuple from first loaded image
    })

    # Build session info
    if not observations:
        info = f"**{state.session_state['session_name']}** has no observation frames."
        return info, None, "", None, checkpoint_manager.refresh_checkpoints(), gr.Radio()

    details = [
        f"**Session:** {state.session_state['session_name']}",
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

    # Create recording reader
    reader = RecordingReader(session_dir)

    # Create replay robot
    replay_robot = ReplayRobot(reader, action_space)

    # Create world model with recorded action selector
    recorded_selector = create_recorded_action_selector(reader)

    def action_selector_adapter(observation, action_space):
        action, _ = recorded_selector()
        return action

    state.world_model = AutoencoderConcatPredictorWorldModel(
        replay_robot,
        action_selector=action_selector_adapter,
        device=state.device,
        frame_size=detected_frame_size,
    )

    info += "\n\n**World model initialized and ready to run**"

    # Initial canvas is None (frame 0 doesn't have enough history)
    min_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    frame_info += f"\n\n**Canvas:** Not available (need {min_frames} frames)"
    initial_canvas = None

    # Update counterfactual action radio based on session's action space
    cf_choices = get_counterfactual_action_choices()
    return info, first_frame, frame_info, initial_canvas, checkpoint_manager.refresh_checkpoints(), gr.update(choices=cf_choices, value=1)


def update_frame(frame_idx):
    """Update frame display and canvas ending at this frame"""
    if not state.session_state.get("observations"):
        return None, "", None

    frame_idx = int(frame_idx)  # Convert to int (Gradio sliders pass floats)

    observations = state.session_state["observations"]
    if frame_idx >= len(observations):
        frame_idx = len(observations) - 1

    obs = observations[frame_idx]
    frame = load_frame_image(obs["full_path"])
    frame_info = f"**Observation {frame_idx + 1} / {len(observations)}**\n\nStep: {obs['step']}\n\nTimestamp: {format_timestamp(obs['timestamp'])}"

    # Get canvas from cache if available
    canvas_cache = state.session_state.get("canvas_cache", {})
    canvas_image = None
    if frame_idx in canvas_cache:
        canvas_data = canvas_cache[frame_idx]
        canvas_image = canvas_data['canvas']  # numpy HxWx3 uint8
        start_idx = canvas_data['start_idx']
        frame_info += f"\n\n**Canvas:** frames {start_idx + 1} → {frame_idx + 1}"
    else:
        min_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
        if frame_idx < min_frames - 1:
            frame_info += f"\n\n**Canvas:** Not available (need {min_frames} frames)"

    return frame, frame_info, canvas_image


def load_validation_session(session_choice):
    """Load a session as validation set (pre-builds canvases but doesn't create world model)"""
    # Handle empty selection or "None" option
    if not session_choice or session_choice.startswith("None"):
        state.clear_validation_session()
        return "No validation session selected"

    # Extract session_dir from choice
    session_dir = session_choice.split(" - ")[-1]
    session_name = os.path.basename(session_dir)

    # Check if this is the same as training session
    if state.session_state.get("session_dir") == session_dir:
        state.clear_validation_session()
        return "⚠️ Cannot use same session for training and validation"

    metadata = load_session_metadata(session_dir)
    events = load_session_events(session_dir)
    observations = extract_observations(events, session_dir)
    actions = extract_actions(events)

    if not observations:
        state.clear_validation_session()
        return f"⚠️ Session **{session_name}** has no observation frames"

    # Pre-build all canvases for validation
    print("\n" + "="*60)
    print(f"Pre-building validation canvases for {session_name}...")
    print("="*60)
    prebuild_start = time.time()
    canvas_cache, detected_frame_size = prebuild_all_canvases(
        session_dir,
        observations,
        actions,
        config.AutoencoderConcatPredictorWorldModelConfig
    )
    prebuild_time = time.time() - prebuild_start
    print(f"Validation canvas pre-building completed in {prebuild_time:.2f}s")
    canvas_width = detected_frame_size[1] * 3 + config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH * 2
    print(f"Memory usage: ~{len(canvas_cache) * detected_frame_size[0] * canvas_width * 3 / (1024**2):.1f} MB")
    print("="*60 + "\n")

    # Store validation session state
    state.validation_session_state.update({
        "session_name": session_name,
        "session_dir": session_dir,
        "observations": observations,
        "actions": actions,
        "canvas_cache": canvas_cache,
        "detected_frame_size": detected_frame_size,
    })

    min_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    num_valid_canvases = len(observations) - (min_frames - 1)

    # Check frame size compatibility with training session
    train_frame_size = state.session_state.get("detected_frame_size")
    if train_frame_size and detected_frame_size != train_frame_size:
        return f"⚠️ **Validation:** {session_name} ({len(observations)} frames) - **Frame size mismatch!** Val: {detected_frame_size}, Train: {train_frame_size}. Validation will be skipped."

    return f"✅ **Validation:** {session_name} ({len(observations)} frames, {num_valid_canvases} canvases)"


def clear_validation_session_ui():
    """Clear validation session and return status message"""
    state.clear_validation_session()
    return "No validation session selected"
