"""
Attention Visualization Utilities for Concat Predictor Decoder

This module provides utilities for visualizing decoder attention from masked patches
to unmasked patches in the AutoencoderConcatPredictorWorldModel.

Key functions:
- compute_patch_centers(): Calculate pixel coordinates for patch centers
- identify_masked_unmasked_patches(): Separate patches based on mask
- draw_attention_connections(): Draw lines from masked to unmasked patches
- create_attention_overlay(): Composite visualization over canvas image
"""

from typing import List, Tuple, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


def compute_patch_centers(
    img_height: int,
    img_width: int,
    patch_size: int
) -> np.ndarray:
    """
    Calculate pixel coordinates (x, y) for the center of each patch.

    Args:
        img_height: Image height in pixels
        img_width: Image width in pixels
        patch_size: Size of each square patch

    Returns:
        centers: array of shape [num_patches, 2] with (x, y) coordinates
    """
    num_patches_h = img_height // patch_size
    num_patches_w = img_width // patch_size

    centers = []
    for row in range(num_patches_h):
        for col in range(num_patches_w):
            # Center of patch in pixel coordinates
            center_y = row * patch_size + patch_size // 2
            center_x = col * patch_size + patch_size // 2
            centers.append([center_x, center_y])

    return np.array(centers)  # [num_patches, 2]


def identify_masked_unmasked_patches(
    patch_mask: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify which patches are masked vs unmasked.

    Args:
        patch_mask: [1, num_patches] boolean tensor where True = masked

    Returns:
        masked_indices: array of patch indices that are masked
        unmasked_indices: array of patch indices that are unmasked
    """
    # Convert to numpy and flatten
    mask_np = patch_mask.cpu().numpy().flatten()  # [num_patches]

    masked_indices = np.where(mask_np)[0]
    unmasked_indices = np.where(~mask_np)[0]

    return masked_indices, unmasked_indices


def aggregate_attention_heads(
    attn_layer: torch.Tensor,
    aggregation: str = 'mean',
    enabled_heads: List[int] = None
) -> np.ndarray:
    """
    Aggregate attention weights across multiple heads.

    Args:
        attn_layer: [B, num_heads, N, N] attention weights for one layer
        aggregation: 'mean', 'max', or 'sum'
        enabled_heads: list of head indices to include (e.g., [0, 1, 3]). If None, use all heads.

    Returns:
        attn_aggregated: [N, N] aggregated attention weights
    """
    # Squeeze batch dimension (assume B=1)
    attn = attn_layer[0].cpu().numpy()  # [num_heads, N, N]

    # Filter to only enabled heads
    if enabled_heads is not None:
        if len(enabled_heads) == 0:
            raise ValueError("At least one attention head must be enabled")
        attn = attn[enabled_heads]  # [num_enabled_heads, N, N]

    if aggregation == 'mean':
        return attn.mean(axis=0)
    elif aggregation == 'max':
        return attn.max(axis=0)
    elif aggregation == 'sum':
        return attn.sum(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def filter_attention_by_quantile(
    attn_weights: np.ndarray,
    masked_indices: np.ndarray,
    unmasked_indices: np.ndarray,
    quantile: float
) -> Tuple[List[Tuple[int, int, float]], float]:
    """
    Filter attention connections by quantile (percentile).

    Args:
        attn_weights: [N, N] aggregated attention weights (includes CLS token at index 0)
        masked_indices: patch indices that are masked
        unmasked_indices: patch indices that are unmasked
        quantile: percentile threshold (0-100). E.g., 95 means show top 5% of connections

    Returns:
        connections: list of (from_patch, to_patch, weight) tuples
        threshold: computed threshold value corresponding to the quantile
    """
    # First pass: collect all weights from masked to unmasked patches
    all_weights = []

    for masked_idx in masked_indices:
        masked_token_idx = masked_idx + 1
        for unmasked_idx in unmasked_indices:
            unmasked_token_idx = unmasked_idx + 1
            weight = attn_weights[masked_token_idx, unmasked_token_idx]
            all_weights.append(weight)

    # Compute threshold from quantile
    if len(all_weights) == 0:
        return [], 0.0

    threshold = np.quantile(all_weights, quantile / 100.0)

    # Second pass: filter connections by computed threshold
    connections = []

    for masked_idx in masked_indices:
        masked_token_idx = masked_idx + 1
        for unmasked_idx in unmasked_indices:
            unmasked_token_idx = unmasked_idx + 1
            weight = attn_weights[masked_token_idx, unmasked_token_idx]

            if weight >= threshold:
                connections.append((masked_idx, unmasked_idx, weight))

    return connections, threshold


def draw_attention_connections(
    canvas_img: np.ndarray,
    patch_centers: np.ndarray,
    attn_weights_list: List[torch.Tensor],
    patch_mask: torch.Tensor,
    quantile: float = 95.0,
    layer_colors: List[str] = None,
    enabled_layers: List[bool] = None,
    enabled_heads: List[int] = None,
    line_width_range: Tuple[float, float] = (0.5, 5.0),
    aggregation: str = 'mean',
    alpha: float = 0.6
) -> plt.Figure:
    """
    Draw attention connections from masked to unmasked patches.

    Args:
        canvas_img: [H, W, 3] RGB image (uint8)
        patch_centers: [num_patches, 2] array of (x, y) centers
        attn_weights_list: list of [B, num_heads, N, N] tensors (one per decoder layer)
        patch_mask: [1, num_patches] boolean tensor where True = masked
        quantile: percentile threshold (0-100). E.g., 95 means show top 5% of connections
        layer_colors: list of color names for each layer (default: matplotlib color cycle)
        enabled_layers: list of booleans indicating which layers to show
        enabled_heads: list of head indices to include (e.g., [0, 1, 3]). If None, use all heads.
        line_width_range: (min_width, max_width) for scaling line thickness by attention
        aggregation: 'mean', 'max', or 'sum' for aggregating across attention heads
        alpha: transparency for lines

    Returns:
        fig: matplotlib figure with visualization
    """
    num_layers = len(attn_weights_list)

    # Default layer colors using matplotlib color cycle
    if layer_colors is None:
        cmap = plt.cm.get_cmap('tab10')
        layer_colors = [cmap(i / num_layers) for i in range(num_layers)]

    # Default to all layers enabled
    if enabled_layers is None:
        enabled_layers = [True] * num_layers

    # Identify masked and unmasked patches
    masked_indices, unmasked_indices = identify_masked_unmasked_patches(patch_mask)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.imshow(canvas_img)
    ax.axis('off')

    # Track global min/max attention for normalization across all layers
    all_weights = []
    computed_thresholds = []

    # Process each layer
    for layer_idx in range(num_layers):
        if not enabled_layers[layer_idx]:
            continue

        # Aggregate attention across heads
        attn_aggregated = aggregate_attention_heads(
            attn_weights_list[layer_idx],
            aggregation=aggregation,
            enabled_heads=enabled_heads
        )

        # Filter connections by quantile
        connections, threshold = filter_attention_by_quantile(
            attn_aggregated,
            masked_indices,
            unmasked_indices,
            quantile
        )
        computed_thresholds.append(threshold)

        if len(connections) == 0:
            continue

        # Extract weights for normalization
        all_weights.extend([w for _, _, w in connections])

        # Draw lines for this layer
        layer_color = layer_colors[layer_idx]

        for from_patch, to_patch, weight in connections:
            from_center = patch_centers[from_patch]
            to_center = patch_centers[to_patch]

            # Scale line width by attention weight
            # Normalize within this layer's connections
            layer_weights = [w for _, _, w in connections]
            min_w = min(layer_weights)
            max_w = max(layer_weights)
            if max_w > min_w:
                normalized_weight = (weight - min_w) / (max_w - min_w)
            else:
                normalized_weight = 0.5

            line_width = (
                line_width_range[0] +
                normalized_weight * (line_width_range[1] - line_width_range[0])
            )

            # Draw line
            ax.plot(
                [from_center[0], to_center[0]],
                [from_center[1], to_center[1]],
                color=layer_color,
                linewidth=line_width,
                alpha=alpha,
                zorder=10 + layer_idx
            )

    # Create legend
    legend_patches = []
    for layer_idx in range(num_layers):
        if enabled_layers[layer_idx]:
            legend_patches.append(
                mpatches.Patch(color=layer_colors[layer_idx], label=f'Layer {layer_idx}')
            )

    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    # Add title with statistics
    total_connections = sum([
        len(filter_attention_by_quantile(
            aggregate_attention_heads(attn_weights_list[i], aggregation, enabled_heads),
            masked_indices,
            unmasked_indices,
            quantile
        )[0])
        for i in range(num_layers) if enabled_layers[i]
    ])

    max_attention = max(all_weights) if all_weights else 0.0
    avg_threshold = np.mean(computed_thresholds) if computed_thresholds else 0.0

    # Format head display
    if enabled_heads is None:
        heads_str = "All"
    else:
        heads_str = f"[{','.join(map(str, enabled_heads))}]"

    ax.set_title(
        f'Decoder Attention Visualization\n'
        f'Connections: {total_connections} | Quantile: {quantile:.1f}% (threshold ≈ {avg_threshold:.4f}) | '
        f'Heads: {heads_str} | Aggregation: {aggregation}',
        fontsize=12,
        pad=10
    )

    plt.tight_layout()
    return fig


def create_attention_statistics(
    attn_weights_list: List[torch.Tensor],
    patch_mask: torch.Tensor,
    quantile: float = 95.0,
    aggregation: str = 'mean',
    enabled_heads: List[int] = None
) -> Dict[str, any]:
    """
    Compute statistics about attention patterns.

    Args:
        attn_weights_list: list of [B, num_heads, N, N] tensors
        patch_mask: [1, num_patches] boolean tensor
        quantile: percentile threshold (0-100). E.g., 95 means count top 5% of connections
        aggregation: head aggregation method
        enabled_heads: list of head indices to include. If None, use all heads.

    Returns:
        stats: dictionary with attention statistics
    """
    masked_indices, unmasked_indices = identify_masked_unmasked_patches(patch_mask)

    stats = {
        'num_layers': len(attn_weights_list),
        'num_masked_patches': len(masked_indices),
        'num_unmasked_patches': len(unmasked_indices),
        'quantile': quantile,
        'aggregation': aggregation,
        'enabled_heads': enabled_heads if enabled_heads is not None else 'all',
        'per_layer_stats': []
    }

    computed_thresholds = []

    for layer_idx, attn_layer in enumerate(attn_weights_list):
        attn_aggregated = aggregate_attention_heads(attn_layer, aggregation, enabled_heads)
        connections, threshold = filter_attention_by_quantile(
            attn_aggregated,
            masked_indices,
            unmasked_indices,
            quantile
        )
        computed_thresholds.append(threshold)

        if connections:
            weights = [w for _, _, w in connections]
            layer_stats = {
                'layer': layer_idx,
                'num_connections': len(connections),
                'min_weight': min(weights),
                'max_weight': max(weights),
                'mean_weight': sum(weights) / len(weights),
                'total_weight': sum(weights),
                'threshold': threshold
            }
        else:
            layer_stats = {
                'layer': layer_idx,
                'num_connections': 0,
                'min_weight': 0.0,
                'max_weight': 0.0,
                'mean_weight': 0.0,
                'total_weight': 0.0,
                'threshold': threshold
            }

        stats['per_layer_stats'].append(layer_stats)

    # Overall statistics
    all_connections = sum([s['num_connections'] for s in stats['per_layer_stats']])
    all_weights = [w for s in stats['per_layer_stats'] for w in [s['mean_weight']] if s['num_connections'] > 0]

    stats['total_connections'] = all_connections
    stats['max_layer_weight'] = max([s['max_weight'] for s in stats['per_layer_stats']]) if stats['per_layer_stats'] else 0.0
    stats['avg_threshold'] = np.mean(computed_thresholds) if computed_thresholds else 0.0

    return stats


def create_attention_heatmap(
    attn_weights_list: List[torch.Tensor],
    patch_mask: torch.Tensor,
    layer_idx: int = 0,
    aggregation: str = 'mean',
    enabled_heads: List[int] = None
) -> plt.Figure:
    """
    Create a heatmap showing attention from masked to unmasked patches.

    Args:
        attn_weights_list: list of [B, num_heads, N, N] tensors
        patch_mask: [1, num_patches] boolean tensor
        layer_idx: which decoder layer to visualize
        aggregation: head aggregation method
        enabled_heads: list of head indices to include. If None, use all heads.

    Returns:
        fig: matplotlib figure with heatmap
    """
    masked_indices, unmasked_indices = identify_masked_unmasked_patches(patch_mask)

    # Aggregate attention across heads
    attn_aggregated = aggregate_attention_heads(
        attn_weights_list[layer_idx],
        aggregation=aggregation,
        enabled_heads=enabled_heads
    )

    # Extract attention from masked to unmasked patches
    # Add 1 to account for CLS token
    masked_token_indices = masked_indices + 1
    unmasked_token_indices = unmasked_indices + 1

    # Build attention matrix [num_masked, num_unmasked]
    attn_matrix = attn_aggregated[masked_token_indices][:, unmasked_token_indices]

    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    im = ax.imshow(attn_matrix, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Unmasked Patch Index', fontsize=12)
    ax.set_ylabel('Masked Patch Index', fontsize=12)

    # Format head display
    if enabled_heads is None:
        heads_str = "All"
    else:
        heads_str = f"[{','.join(map(str, enabled_heads))}]"

    ax.set_title(
        f'Decoder Layer {layer_idx} Attention Heatmap\n'
        f'Masked Patches ({len(masked_indices)}) → Unmasked Patches ({len(unmasked_indices)})\n'
        f'Heads: {heads_str} | Aggregation: {aggregation}',
        fontsize=12,
        pad=10
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=12)

    # Add grid
    ax.set_xticks(np.arange(len(unmasked_indices))[::10])
    ax.set_yticks(np.arange(len(masked_indices))[::10])
    ax.grid(False)

    plt.tight_layout()
    return fig
