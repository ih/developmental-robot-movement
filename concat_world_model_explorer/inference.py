"""
Counterfactual inference for testing model predictions with modified actions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from . import state
from .utils import format_loss, compute_canvas_figsize
from .canvas_ops import build_counterfactual_canvas, create_difference_heatmap
from models.autoencoder_concat_predictor import (
    canvas_to_tensor,
    compute_randomized_patch_mask_for_last_slot,
    compute_hybrid_loss_on_masked_patches,
)
from session_explorer_lib import describe_action


def get_action_description(action):
    """
    Get human-readable description of an action.

    Args:
        action: Action dictionary

    Returns:
        String description of the action
    """
    if isinstance(action, dict):
        if 'action' in action:
            # Discrete action style (toroidal dot, SO-101)
            action_val = action['action']
            if action_val == 0:
                return "Stay (RED)"
            elif action_val == 1:
                return "Move Positive (GREEN)"
            elif action_val == 2:
                return "Move Negative (BLUE)"
        elif 'motor_right' in action:
            # JetBot style
            motor_right = action['motor_right']
            if motor_right == 0:
                return "Stop (RED)"
            else:
                return f"Forward at {motor_right} (GREEN)"

    # Fallback to session_explorer_lib's describe_action
    return describe_action(action)


def get_counterfactual_action_choices():
    """
    Generate radio choices for counterfactual action based on session's action space.

    Returns:
        List of (label, value) tuples for Gradio Radio component
    """
    action_space = state.session_state.get("action_space", [])
    if len(action_space) >= 3:
        # SO-101 style: 3 actions
        return [
            ("Stay (action=0, RED)", 0),
            ("Move Positive (action=1, GREEN)", 1),
            ("Move Negative (action=2, BLUE)", 2),
        ]
    else:
        # Toroidal dot style: 2 actions
        return [
            ("Stay (action=0, RED)", 0),
            ("Move Right (action=1, GREEN)", 1),
        ]


def run_counterfactual_inference(frame_idx, counterfactual_action):
    """
    Run inference on true and counterfactual canvases and compare.

    Args:
        frame_idx: The frame index to test
        counterfactual_action: The counterfactual action

    Returns:
        Tuple of (status_msg, true_canvas_fig, cf_canvas_fig, true_inference_fig,
                  cf_inference_fig, diff_heatmap_fig, stats_md)
    """
    if state.world_model is None:
        return "Please load a session first", None, None, None, None, None, ""

    if not state.session_state.get("observations") or not state.session_state.get("actions"):
        return "No session data available", None, None, None, None, None, ""

    # Build both canvases
    true_canvas, cf_canvas, error, start_idx, actual_action = build_counterfactual_canvas(
        frame_idx, counterfactual_action
    )
    if error:
        return error, None, None, None, None, None, ""

    frame_idx = int(frame_idx)

    # Convert canvases to tensors
    true_tensor = canvas_to_tensor(true_canvas).to(state.device)
    cf_tensor = canvas_to_tensor(cf_canvas).to(state.device)

    # Compute patch mask for last slot (same mask for both)
    canvas_height, canvas_width = true_tensor.shape[-2:]
    num_frames = config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE

    patch_mask = compute_randomized_patch_mask_for_last_slot(
        img_size=(canvas_height, canvas_width),
        patch_size=config.AutoencoderConcatPredictorWorldModelConfig.PATCH_SIZE,
        num_frame_slots=num_frames,
        sep_width=config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
        mask_ratio_min=config.MASK_RATIO_MIN,
        mask_ratio_max=config.MASK_RATIO_MAX,
    ).to(state.device)

    # Run inference on both canvases
    state.world_model.autoencoder.eval()
    with torch.no_grad():
        # True canvas inference
        true_pred_patches, _ = state.world_model.autoencoder.forward_with_patch_mask(true_tensor, patch_mask)
        true_img_pred = state.world_model.autoencoder.unpatchify(true_pred_patches)

        # Counterfactual canvas inference
        cf_pred_patches, _ = state.world_model.autoencoder.forward_with_patch_mask(cf_tensor, patch_mask)
        cf_img_pred = state.world_model.autoencoder.unpatchify(cf_pred_patches)

        # Compute losses for both
        true_target_patches = state.world_model.autoencoder.patchify(true_tensor)
        cf_target_patches = state.world_model.autoencoder.patchify(cf_tensor)

        true_masked_pred = true_pred_patches[patch_mask]
        true_masked_target = true_target_patches[patch_mask]
        cf_masked_pred = cf_pred_patches[patch_mask]
        cf_masked_target = cf_target_patches[patch_mask]

        true_loss_dict = compute_hybrid_loss_on_masked_patches(
            true_masked_pred, true_masked_target,
            focal_alpha=config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_LOSS_ALPHA,
            focal_beta=config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_BETA
        )
        cf_loss_dict = compute_hybrid_loss_on_masked_patches(
            cf_masked_pred, cf_masked_target,
            focal_alpha=config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_LOSS_ALPHA,
            focal_beta=config.AutoencoderConcatPredictorWorldModelConfig.FOCAL_BETA
        )

        true_loss = true_loss_dict['loss_hybrid'].item() if torch.is_tensor(true_loss_dict['loss_hybrid']) else true_loss_dict['loss_hybrid']
        cf_loss = cf_loss_dict['loss_hybrid'].item() if torch.is_tensor(cf_loss_dict['loss_hybrid']) else cf_loss_dict['loss_hybrid']

    # Generate composite images for both
    state.world_model.last_training_canvas = true_canvas
    state.world_model.last_training_mask = patch_mask
    true_composite = state.world_model.get_canvas_inpainting_composite(true_canvas, patch_mask)

    state.world_model.last_training_canvas = cf_canvas
    cf_composite = state.world_model.get_canvas_inpainting_composite(cf_canvas, patch_mask)

    # Compute pixel difference statistics
    diff = np.abs(cf_composite.astype(float) - true_composite.astype(float))
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    diff_gray = np.mean(diff, axis=2)
    nonzero_pixels = np.sum(diff_gray > 5)  # Count pixels with > 5 difference

    # Get action descriptions (robot-agnostic)
    actual_action_desc = get_action_description(actual_action)
    cf_action_desc = get_action_description({"action": counterfactual_action})

    # Compute dynamic figsize based on canvas dimensions
    canvas_h, canvas_w = true_canvas.shape[:2]
    figsize = compute_canvas_figsize(canvas_h, canvas_w)

    # Generate visualizations
    # 1. True canvas figure
    fig_true_canvas, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(true_canvas)
    ax.set_title(f"True Canvas (Actual Last Action: {actual_action_desc})")
    ax.axis("off")
    plt.tight_layout()

    # 2. Counterfactual canvas figure
    fig_cf_canvas, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(cf_canvas)
    ax.set_title(f"Counterfactual Canvas (Last Action: {cf_action_desc})")
    ax.axis("off")
    plt.tight_layout()

    # 3. True inference composite
    fig_true_inference, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(true_composite)
    ax.set_title(f"True Inference Composite (Loss: {format_loss(true_loss)})")
    ax.axis("off")
    plt.tight_layout()

    # 4. Counterfactual inference composite
    fig_cf_inference, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(cf_composite)
    ax.set_title(f"Counterfactual Inference Composite (Loss: {format_loss(cf_loss)})")
    ax.axis("off")
    plt.tight_layout()

    # 5. Difference heatmap
    fig_diff_heatmap = create_difference_heatmap(true_composite, cf_composite)

    # 6. Statistics markdown
    stats_md = f"""
### Counterfactual Analysis Statistics

| Metric | Value |
|--------|-------|
| **Actual Last Action** | {actual_action_desc} |
| **Counterfactual Action** | {cf_action_desc} |
| **True Canvas Loss** | {format_loss(true_loss)} |
| **Counterfactual Canvas Loss** | {format_loss(cf_loss)} |
| **Mean Pixel Difference** | {mean_diff:.2f} |
| **Max Pixel Difference** | {max_diff:.2f} |
| **Changed Pixels (>5)** | {nonzero_pixels:,} |

*Note: If the model has learned the action-state relationship, changing the action
should result in different predictions for the masked region (last frame).*
"""

    # Status message
    status_msg = f"**Counterfactual inference on frame {frame_idx + 1}**\n\n"
    status_msg += f"Comparing: Actual={actual_action_desc} vs Counterfactual={cf_action_desc}"

    return status_msg, fig_true_canvas, fig_cf_canvas, fig_true_inference, fig_cf_inference, fig_diff_heatmap, stats_md
