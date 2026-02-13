#!/usr/bin/env python
"""Wrapper script that imports custom policies before running lerobot-record."""

import sys

# Patch camera backend to use DSHOW on Windows (before importing camera modules)
import platform
if platform.system() == "Windows":
    import cv2
    import lerobot.cameras.utils as cam_utils

    _original_get_cv2_backend = cam_utils.get_cv2_backend

    def _patched_get_cv2_backend():
        """Use DSHOW instead of MSMF on Windows for better camera compatibility."""
        if platform.system() == "Windows":
            return int(cv2.CAP_DSHOW)
        return _original_get_cv2_backend()

    cam_utils.get_cv2_backend = _patched_get_cv2_backend

    # Patch async_read to use synchronous read on Windows (threading issues with DSHOW)
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    _original_async_read = OpenCVCamera.async_read

    def _patched_async_read(self, timeout_ms: float = 200):
        """Use synchronous read on Windows to avoid threading issues with DSHOW."""
        return self.read()

    OpenCVCamera.async_read = _patched_async_read

# Import the custom policy to trigger registration before lerobot parses args
# Use explicit submodule imports to bypass namespace package shadowing when
# running from the project root (outer directory lacks __init__.py)
from lerobot_policy_simple_joint.configuration_simple_joint import SimpleJointConfig  # noqa: F401
from lerobot_policy_simple_joint.modeling_simple_joint import SimpleJointPolicy  # noqa: F401

# Now run lerobot-record
from lerobot.scripts.lerobot_record import record


def parse_arg(name: str) -> str | None:
    """Parse a command line argument value."""
    for i, arg in enumerate(sys.argv):
        if arg.startswith(f"--{name}="):
            return arg.split("=", 1)[1]
        elif arg == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def check_and_clean_dataset_cache():
    """Check if dataset cache exists and prompt user to remove it."""
    import shutil
    import time
    from pathlib import Path

    repo_id = parse_arg("dataset.repo_id")
    if not repo_id:
        return  # No repo_id specified, let lerobot handle it

    # Construct cache path (repo_id like "user/dataset" maps to subdirectories)
    cache_base = Path.home() / ".cache" / "huggingface" / "lerobot"
    cache_path = cache_base / repo_id  # pathlib handles forward slashes correctly

    if cache_path.exists():
        print(f"\nDataset cache already exists at:\n  {cache_path}\n")
        while True:
            response = input("Remove existing cache? [y/n]: ").strip().lower()
            if response in ("y", "yes"):
                shutil.rmtree(cache_path)
                # Wait for Windows filesystem to complete deletion (up to 5 seconds)
                for _ in range(50):
                    if not cache_path.exists():
                        break
                    time.sleep(0.1)
                if cache_path.exists():
                    print(f"ERROR: Failed to remove cache at {cache_path}")
                    print("Please close any programs using this directory and try again.")
                    sys.exit(1)
                print(f"Removed: {cache_path}\n")
                break
            elif response in ("n", "no"):
                print("Aborting. Please use a different --dataset.repo_id or remove the cache manually.")
                sys.exit(1)
            else:
                print("Please enter 'y' or 'n'")


def calculate_and_inject_episode_time():
    """Calculate episode time from action_sequence and action_duration, inject into args."""
    import ast

    action_sequence_str = parse_arg("policy.action_sequence")
    action_duration_str = parse_arg("policy.action_duration")

    if not action_sequence_str:
        return  # No action sequence, use default episode time

    # Parse action sequence
    try:
        action_sequence = ast.literal_eval(action_sequence_str)
        if not isinstance(action_sequence, list):
            return
    except (ValueError, SyntaxError):
        return

    # Parse action duration (default 0.5)
    action_duration = 0.5
    if action_duration_str:
        try:
            action_duration = float(action_duration_str)
        except ValueError:
            pass

    # Calculate episode time: (sequence length × action duration) + buffer for completion
    # Buffer accounts for camera warmup, calibration, and final action completion
    episode_time = len(action_sequence) * action_duration + 5.0

    # Check if episode_time_s is already specified
    if parse_arg("dataset.episode_time_s") is None:
        # Inject the calculated episode time
        sys.argv.append(f"--dataset.episode_time_s={episode_time}")
        print(f"Auto-calculated episode time: {episode_time:.1f}s "
              f"({len(action_sequence)} actions × {action_duration}s + 5s buffer)")
        print(f"  Note: Sequence will execute once and stop at action 0 (no wrapping)")


def inject_discrete_action_log_dir():
    """Inject --policy.discrete_action_log_dir into CLI args if not already set.

    Sets the log directory to the dataset's meta/ directory so logs are
    included when the dataset is pushed to HuggingFace Hub.
    The policy handles per-episode log file creation automatically in reset().
    """
    from pathlib import Path

    # Skip if already specified or no repo_id
    if parse_arg("policy.discrete_action_log_dir") is not None:
        return
    repo_id = parse_arg("dataset.repo_id")
    if not repo_id:
        return

    cache_base = Path.home() / ".cache" / "huggingface" / "lerobot"
    log_dir = cache_base / repo_id / "meta" / "discrete_action_logs"

    sys.argv.append(f"--policy.discrete_action_log_dir={log_dir}")
    print(f"Discrete action logs will be saved to: {log_dir}")


def return_arm_to_start(robot_port: str, starting_positions: dict):
    """Move the arm back to its starting position.

    Uses low-level motor bus access to avoid triggering calibration.
    """
    import time

    try:
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors import Motor, MotorNormMode

        print("\nReturning arm to starting position...")

        # Create motor bus directly (no calibration)
        bus = FeetechMotorsBus(
            port=robot_port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
        )
        bus.connect()

        # Get current positions
        current_positions = bus.sync_read("Present_Position")

        # Move gradually back to starting position
        steps = 50
        for step in range(steps + 1):
            alpha = step / steps
            # Interpolate between current and starting position
            target = {}
            for joint_name, start_pos in starting_positions.items():
                current_pos = current_positions[joint_name]
                target[joint_name] = current_pos + alpha * (start_pos - current_pos)

            bus.sync_write("Goal_Position", target)
            time.sleep(0.02)

        print("Arm returned to starting position.")
        bus.disconnect()

    except Exception as e:
        print(f"Warning: Could not return arm to start position: {e}")


def capture_starting_position(robot_port: str) -> dict:
    """Capture the arm's starting position before recording.

    Skipped for now - the return-to-start feature requires calibration.
    The main recording will handle calibration properly.
    """
    # Skip position capture to avoid calibration issues
    # The robot will be calibrated during the main recording
    print("Note: Starting position capture skipped (handled by main recording)")
    return {}


def print_command_line():
    """Print the full lerobot-record command for debugging."""
    # Build command: replace script name with lerobot-record, join all args
    args = ["lerobot-record"] + sys.argv[1:]
    command = " ".join(args)
    print("\n=== Executing lerobot-record ===")
    print(command)
    print("================================\n")


if __name__ == "__main__":
    check_and_clean_dataset_cache()
    calculate_and_inject_episode_time()
    inject_discrete_action_log_dir()

    # Get robot port for position capture/restore
    robot_port = parse_arg("robot.port")

    # Capture starting position
    starting_positions = {}
    if robot_port:
        starting_positions = capture_starting_position(robot_port)

    # Print full command for debugging
    print_command_line()

    # Run the recording
    try:
        record()
    finally:
        # Return arm to starting position after recording
        if robot_port and starting_positions:
            return_arm_to_start(robot_port, starting_positions)
