"""
Canvas building operations for world model training and inference.
"""

import numpy as np
import matplotlib.pyplot as plt

import config
from . import state
from .utils import compute_canvas_figsize
from session_explorer_lib import load_frame_image
from models.autoencoder_concat_predictor import build_canvas


def build_canvas_from_frame(frame_idx):
    """Build canvas from selected frame and its history (helper function)"""
    observations = state.session_state["observations"]
    actions = state.session_state["actions"]

    # Validate frame index
    frame_idx = int(frame_idx)
    if frame_idx >= len(observations):
        return None, f"Frame index {frame_idx} out of range (max: {len(observations)-1})", None, None

    # Check if we have enough history
    min_frames_needed = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
    if frame_idx < min_frames_needed - 1:
        return None, f"Need at least {min_frames_needed} frames of history. Selected frame {frame_idx+1} doesn't have enough history.", None, None

    # Try to use pre-built canvas cache first (Phase 1 optimization)
    canvas_cache = state.session_state.get("canvas_cache", {})
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
            # Fallback: use session's action space for default (JetBot support)
            action_space = state.session_state.get("action_space", [])
            default_action = action_space[0] if action_space else {"action": 0}
            selected_actions.append(default_action)

    # Build interleaved history
    interleaved = [selected_frames[0]]
    for i in range(len(selected_actions)):
        interleaved.append(selected_actions[i])
        if i + 1 < len(selected_frames):
            interleaved.append(selected_frames[i + 1])

    # Build training canvas
    training_canvas = build_canvas(
        interleaved,
        frame_size=config.AutoencoderConcatPredictorWorldModelConfig.FRAME_SIZE,
        sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
    )

    return training_canvas, None, start_idx, interleaved


def build_counterfactual_canvas(frame_idx, counterfactual_action):
    """
    Build a canvas with a modified last action for counterfactual testing.

    Args:
        frame_idx: The frame index to build the canvas for
        counterfactual_action: The action to substitute for the actual last action

    Returns:
        Tuple of (true_canvas, counterfactual_canvas, error, start_idx, actual_last_action)
    """
    # Get the true canvas and interleaved history
    true_canvas, error, start_idx, interleaved = build_canvas_from_frame(frame_idx)
    if true_canvas is None:
        return None, None, error, None, None

    # Get actual last action (at index -2 in interleaved: [f0, a0, f1, a1, f2])
    action_space = state.session_state.get("action_space", [])
    default_action = action_space[0] if action_space else {"action": 0}
    actual_last_action = interleaved[-2] if len(interleaved) >= 3 else default_action

    # Create counterfactual interleaved by copying and modifying last action
    counterfactual_interleaved = interleaved.copy()
    counterfactual_interleaved[-2] = {"action": counterfactual_action}

    # Build counterfactual canvas
    counterfactual_canvas = build_canvas(
        counterfactual_interleaved,
        frame_size=config.AutoencoderConcatPredictorWorldModelConfig.FRAME_SIZE,
        sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
    )

    return true_canvas, counterfactual_canvas, None, start_idx, actual_last_action


def create_difference_heatmap(true_composite, cf_composite):
    """
    Create heatmap showing absolute pixel differences between two images.

    Args:
        true_composite: True inference composite (numpy array HxWx3)
        cf_composite: Counterfactual inference composite (numpy array HxWx3)

    Returns:
        Matplotlib figure with difference heatmap
    """
    diff = np.abs(cf_composite.astype(float) - true_composite.astype(float))
    diff_gray = np.mean(diff, axis=2)  # Average across RGB channels

    # Compute dynamic figsize based on composite dimensions
    canvas_h, canvas_w = diff_gray.shape[:2]
    figsize = compute_canvas_figsize(canvas_h, canvas_w)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(diff_gray, cmap='hot', vmin=0, vmax=255)
    ax.set_title("Prediction Difference Heatmap (Counterfactual - True)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Absolute Pixel Difference")
    plt.tight_layout()
    return fig
