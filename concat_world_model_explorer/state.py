"""
Global state management for the concat world model explorer.
"""

import torch
from typing import Dict, List, Tuple, Optional
import config

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global state variables
session_state = {}
validation_session_state = {}  # Stores validation session data (observations, actions, canvas_cache)
world_model = None
current_checkpoint_name = None
selected_robot_type = "toroidal_dot"  # Default robot type for session selection
instance_id = None  # Unique identifier for this instance (set by __main__.py based on port)

# Training control flag
training_stop_requested = False


def request_training_stop():
    """Request training to stop at the next checkpoint."""
    global training_stop_requested
    training_stop_requested = True


def reset_training_stop():
    """Reset the stop flag (call before starting training)."""
    global training_stop_requested
    training_stop_requested = False

# Checkpoint metadata (populated when loading a checkpoint)
# Used for resume functionality
loaded_checkpoint_metadata = {
    'samples_seen': 0,
    'loss': None,
    'learning_rate': None,
    'original_peak_lr': None,  # Peak LR from first training session (for global schedule)
    'scheduler_step': 0,
    'timestamp': None,
    'checkpoint_name': None,
}


def reset_checkpoint_metadata():
    """Reset checkpoint metadata to defaults (e.g., when starting fresh)."""
    global loaded_checkpoint_metadata
    loaded_checkpoint_metadata = {
        'samples_seen': 0,
        'loss': None,
        'learning_rate': None,
        'original_peak_lr': None,  # Peak LR from first training session (for global schedule)
        'scheduler_step': 0,
        'timestamp': None,
        'checkpoint_name': None,
    }


def clear_validation_session():
    """Clear validation session state."""
    global validation_session_state
    validation_session_state = {}


def get_checkpoint_dir_for_session(session_path: str) -> str:
    """
    Return appropriate checkpoint directory based on session's robot type.

    Args:
        session_path: Path to the session directory

    Returns:
        Path to robot-specific checkpoint directory
    """
    session_lower = session_path.lower()
    if "jetbot" in session_lower:
        return config.JETBOT_CHECKPOINT_DIR
    elif "so101" in session_lower:
        return config.SO101_CHECKPOINT_DIR
    return config.TOROIDAL_DOT_CHECKPOINT_DIR


# =============================================================================
# Loss-Weighted Sampling State
# =============================================================================

# Per-sample loss tracking for loss-weighted sampling
sample_losses: Dict[int, float] = {}  # frame_idx -> most recent loss


def update_sample_losses(batch_indices: List[int], losses: List[float]) -> None:
    """
    Update loss tracking for samples after a batch.

    Args:
        batch_indices: List of frame indices in the batch
        losses: List of per-sample losses (same length as batch_indices)
    """
    global sample_losses
    for idx, loss in zip(batch_indices, losses):
        sample_losses[idx] = loss


def compute_sample_weights(
    temperature: float,
    all_valid_indices: List[int],
    eps: float = 1e-8
) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute sampling weights from tracked losses using softmax with temperature.

    Higher loss -> higher weight (more likely to be sampled).
    Samples without loss data get the mean weight (neutral).

    Args:
        temperature: Controls weight sharpness (lower = more focus on high-loss)
        all_valid_indices: All valid frame indices that can be sampled
        eps: Small value for numerical stability

    Returns:
        Tuple of (weights tensor, indices list) where weights[i] corresponds to indices[i]
    """
    global sample_losses

    if not sample_losses:
        # No loss data yet - return uniform weights
        n = len(all_valid_indices)
        return torch.ones(n) / n, all_valid_indices

    # Get losses for all valid indices (use mean for unknown samples)
    known_losses = [sample_losses.get(idx) for idx in all_valid_indices]
    known_values = [l for l in known_losses if l is not None]

    if not known_values:
        # No known losses - return uniform
        n = len(all_valid_indices)
        return torch.ones(n) / n, all_valid_indices

    mean_loss = sum(known_values) / len(known_values)

    # Build loss tensor, substituting mean for unknown samples
    loss_values = torch.tensor([
        l if l is not None else mean_loss
        for l in known_losses
    ], dtype=torch.float32)

    # Normalize to [0, 1] range to handle different loss scales
    loss_min, loss_max = loss_values.min(), loss_values.max()
    if loss_max - loss_min > eps:
        normalized = (loss_values - loss_min) / (loss_max - loss_min)
    else:
        # All losses are the same - return uniform
        n = len(all_valid_indices)
        return torch.ones(n) / n, all_valid_indices

    # Apply temperature-scaled softmax: higher loss -> higher weight
    weights = torch.softmax(normalized / max(temperature, eps), dim=0)

    return weights, all_valid_indices


def reset_sample_losses() -> None:
    """Reset loss tracking (call on session change or fresh training)."""
    global sample_losses
    sample_losses = {}


def get_sample_loss_stats() -> Dict[str, float]:
    """
    Get statistics about tracked sample losses.

    Returns:
        Dict with loss statistics (count, mean, std, min, max)
    """
    global sample_losses

    if not sample_losses:
        return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}

    values = list(sample_losses.values())
    import numpy as np
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }
