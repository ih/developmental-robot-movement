"""
Global state management for the concat world model explorer.
"""

import torch
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
