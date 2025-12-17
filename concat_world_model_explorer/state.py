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
world_model = None
current_checkpoint_name = None


def get_checkpoint_dir_for_session(session_path: str) -> str:
    """
    Return appropriate checkpoint directory based on session's robot type.

    Args:
        session_path: Path to the session directory

    Returns:
        Path to robot-specific checkpoint directory
    """
    if "jetbot" in session_path.lower():
        return config.JETBOT_CHECKPOINT_DIR
    return config.TOROIDAL_DOT_CHECKPOINT_DIR
