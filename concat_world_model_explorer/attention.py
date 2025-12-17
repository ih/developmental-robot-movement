"""
Attention visualization operations for decoder attention analysis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from . import state
from .canvas_ops import build_canvas_from_frame
from session_explorer_lib import load_frame_image
from models.autoencoder_concat_predictor import canvas_to_tensor
from attention_viz import (
    detect_dot_patches,
    draw_attention_from_selected,
    create_attention_heatmap_overlay,
    compute_patch_centers,
    aggregate_attention_heads,
)


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

    if state.world_model is None:
        return "Please load a session first", None, ""

    if not state.session_state.get("observations") or not state.session_state.get("actions"):
        return "No session data available", None, ""

    try:
        # Build canvas from selected frame
        training_canvas, error, start_idx, interleaved = build_canvas_from_frame(frame_idx)
        if training_canvas is None:
            return error, None, ""

        frame_idx = int(frame_idx)

        # Get current frame for visualization
        current_frame_path = state.session_state["observations"][frame_idx]["full_path"]
        current_frame = load_frame_image(current_frame_path)
        current_frame_np = np.array(current_frame)

        # Convert canvas to tensor and run inference to get attention
        canvas_tensor = canvas_to_tensor(training_canvas).to(state.device)

        # Run forward pass with attention capture (no masking for inference)
        with torch.no_grad():
            latent = state.world_model.autoencoder.encode(canvas_tensor)
            decoded, attn_weights_list = state.world_model.autoencoder.decode(latent, return_attn=True)

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
