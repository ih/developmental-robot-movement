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


def refresh_sessions():
    """Refresh session list - returns all sessions (JetBot and toroidal dot)"""
    sessions = list_session_dirs()  # Already returns both robot types
    choices = [os.path.basename(s) + " - " + s for s in sessions]
    return gr.Dropdown(choices=choices, value=choices[-1] if choices else None)


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
        return info, None, "", checkpoint_manager.refresh_checkpoints(), gr.Radio()

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

    # Update counterfactual action radio based on session's action space
    cf_choices = get_counterfactual_action_choices()
    return info, first_frame, frame_info, checkpoint_manager.refresh_checkpoints(), gr.update(choices=cf_choices, value=1)


def update_frame(frame_idx):
    """Update frame display"""
    if not state.session_state.get("observations"):
        return None, ""

    frame_idx = int(frame_idx)  # Convert to int (Gradio sliders pass floats)

    observations = state.session_state["observations"]
    if frame_idx >= len(observations):
        frame_idx = len(observations) - 1

    obs = observations[frame_idx]
    frame = load_frame_image(obs["full_path"])
    frame_info = f"**Observation {frame_idx + 1} / {len(observations)}**\n\nStep: {obs['step']}\n\nTimestamp: {format_timestamp(obs['timestamp'])}"

    return frame, frame_info
