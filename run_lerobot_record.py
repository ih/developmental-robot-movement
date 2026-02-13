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
import lerobot_policy_simple_joint  # noqa: F401

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

    # Calculate episode time: sequence length × action_duration + buffer
    # action_duration should be calibrated to include servo settling time
    # Buffer accounts for camera warmup, calibration, and final action completion
    episode_time = len(action_sequence) * action_duration + 5.0

    # Check if episode_time_s is already specified
    if parse_arg("dataset.episode_time_s") is None:
        # Inject the calculated episode time
        sys.argv.append(f"--dataset.episode_time_s={episode_time}")
        print(f"Auto-calculated episode time: {episode_time:.1f}s "
              f"({len(action_sequence)} actions x {action_duration}s/action + 5s buffer)")
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


def _create_motor_bus(robot_port: str):
    """Create a FeetechMotorsBus with SO-101 motor configuration.

    Returns a connected bus. Caller is responsible for disconnecting.
    """
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

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
    return bus


# SO-101 joint names matching motor bus keys
SO101_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def _wait_for_settle(bus, joint_name: str, timeout: float = 5.0,
                     threshold: float = 0.5, stable_count: int = 5,
                     poll_interval: float = 0.010) -> float:
    """Poll Present_Position until the joint stops moving.

    Args:
        bus: Connected FeetechMotorsBus
        joint_name: Motor bus joint name (e.g. "shoulder_pan")
        timeout: Max wait time in seconds
        threshold: Position range (in motor units) to consider settled
        stable_count: Number of consecutive stable reads required
        poll_interval: Seconds between position polls

    Returns:
        Elapsed time until settled, or timeout value if not settled.
    """
    import time

    start = time.time()
    recent = []

    while time.time() - start < timeout:
        time.sleep(poll_interval)
        pos = bus.sync_read("Present_Position")[joint_name]
        recent.append(pos)

        if len(recent) > stable_count:
            recent.pop(0)

        if len(recent) == stable_count:
            if max(recent) - min(recent) < threshold:
                return time.time() - start

    return timeout


def calibrate_action_duration(robot_port: str, joint_name: str,
                              position_delta: float) -> float:
    """Measure servo settling time to determine optimal action_duration.

    Sends a test movement command and polls Present_Position to detect
    when the servo stops moving. Returns the measured time with a 20%
    safety margin.

    Args:
        robot_port: Serial port (e.g. "COM8")
        joint_name: Policy joint name (e.g. "shoulder_pan.pos")
        position_delta: Movement magnitude to test with

    Returns:
        Calibrated action_duration in seconds
    """
    import time

    # Map policy joint name (e.g. "shoulder_pan.pos") to motor bus name
    motor_name = joint_name.replace(".pos", "")
    if motor_name not in SO101_JOINT_NAMES:
        print(f"  Warning: unknown joint '{motor_name}', using default 0.5s")
        return 0.5

    print(f"\n=== Calibrating action_duration ===")
    print(f"  Joint: {joint_name} (motor: {motor_name})")
    print(f"  Position delta: {position_delta}")

    try:
        bus = _create_motor_bus(robot_port)

        # Read current position
        current_pos = bus.sync_read("Present_Position")[motor_name]
        print(f"  Current position: {current_pos:.1f}")

        # Send test movement (+position_delta)
        target_pos = current_pos + position_delta
        target_pos = max(-100, min(100, target_pos))
        print(f"  Moving to: {target_pos:.1f} (+{position_delta})")

        bus.sync_write("Goal_Position", {motor_name: target_pos})
        forward_time = _wait_for_settle(bus, motor_name)
        final_pos = bus.sync_read("Present_Position")[motor_name]
        print(f"  Forward settle time: {forward_time:.3f}s (reached {final_pos:.1f})")

        # Return to original position
        bus.sync_write("Goal_Position", {motor_name: current_pos})
        return_time = _wait_for_settle(bus, motor_name)
        print(f"  Return settle time: {return_time:.3f}s")

        bus.disconnect()

        # Use the longer of forward/return with 20% safety margin
        measured = max(forward_time, return_time)
        calibrated = round(measured * 1.2, 2)
        print(f"  Measured: {measured:.3f}s -> calibrated: {calibrated:.2f}s (1.2x margin)")
        print(f"=================================\n")
        return calibrated

    except Exception as e:
        print(f"  Calibration failed: {e}")
        print(f"  Using default action_duration=0.5s")
        print(f"=================================\n")
        return 0.5


def verify_calibration_with_canvases(robot_port: str, joint_name: str,
                                     position_delta: float,
                                     action_duration: float,
                                     camera_indices: list) -> bool:
    """Run a short test sequence and show sample canvases for verification.

    Executes [move+, stay, move-, stay] with the calibrated action_duration,
    captures camera frames at each decision boundary, and displays canvases
    with colored separators for visual verification.

    Args:
        robot_port: Serial port (e.g. "COM8")
        joint_name: Policy joint name (e.g. "shoulder_pan.pos")
        position_delta: Movement magnitude
        action_duration: Calibrated action duration to test
        camera_indices: List of camera indices to capture from

    Returns:
        True if user approves, False if user wants to abort/retry
    """
    import time
    import cv2
    import numpy as np

    motor_name = joint_name.replace(".pos", "")
    test_actions = [1, 0, 2, 0]  # move+, stay, move-, stay
    action_colors = {
        0: (0, 0, 255),    # RED (BGR) = stay
        1: (0, 255, 0),    # GREEN (BGR) = move+
        2: (255, 0, 0),    # BLUE (BGR) = move-
    }
    action_names = {0: "STAY", 1: "MOVE+", 2: "MOVE-"}
    sep_width = 4

    print(f"\n=== Calibration Verification ===")
    print(f"  Running test sequence: {[action_names[a] for a in test_actions]}")
    print(f"  Action duration: {action_duration}s")

    try:
        bus = _create_motor_bus(robot_port)

        # Open cameras
        caps = []
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
            if cap.isOpened():
                caps.append(cap)
            else:
                print(f"  Warning: Could not open camera {idx}")

        if not caps:
            print(f"  No cameras available, skipping visual verification")
            bus.disconnect()
            return True

        def capture_frame():
            """Capture and resize a frame from the first available camera."""
            for cap in caps:
                ret, frame = cap.read()
                if ret:
                    return cv2.resize(frame, (224, 224))
            return np.zeros((224, 224, 3), dtype=np.uint8)

        # Capture initial observation
        time.sleep(0.3)  # Let camera warm up
        frames = [capture_frame()]
        actions_taken = []

        # Execute test sequence
        current_pos = bus.sync_read("Present_Position")[motor_name]

        for action in test_actions:
            # Compute target
            if action == 1:
                target = current_pos + position_delta
            elif action == 2:
                target = current_pos - position_delta
            else:
                target = current_pos
            target = max(-100, min(100, target))

            # Execute action
            bus.sync_write("Goal_Position", {motor_name: target})
            actions_taken.append(action)

            # Wait for action_duration
            time.sleep(action_duration)

            # Capture result frame
            frames.append(capture_frame())
            current_pos = bus.sync_read("Present_Position")[motor_name]

        # Return to original position
        bus.sync_write("Goal_Position", {motor_name: bus.sync_read("Present_Position")[motor_name]})
        bus.disconnect()

        # Release cameras
        for cap in caps:
            cap.release()

        # Build verification canvas: frame | separator | frame | separator | ...
        canvas_parts = []
        for i, frame in enumerate(frames):
            canvas_parts.append(frame)
            if i < len(actions_taken):
                sep = np.zeros((224, sep_width, 3), dtype=np.uint8)
                color = action_colors[actions_taken[i]]
                sep[:, :] = color
                canvas_parts.append(sep)

        canvas = np.hstack(canvas_parts)

        # Add labels at the top
        label_height = 30
        labeled = np.zeros((224 + label_height, canvas.shape[1], 3), dtype=np.uint8)
        labeled[label_height:, :] = canvas

        # Add action labels above separators
        x = 224  # Start after first frame
        for i, action in enumerate(actions_taken):
            label = action_names[action]
            color = action_colors[action]
            cv2.putText(labeled, label, (x - 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            x += sep_width + 224

        # Display
        cv2.imshow("Calibration Verification", labeled)
        print(f"\n  GREEN/BLUE separators: arm should have MOVED between frames")
        print(f"  RED separators: arm should show NO movement between frames")
        print(f"\n  Press any key to close the preview window.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Ask user
        while True:
            response = input("\n  Does the timing look correct? [y/n/r] "
                             "(y=proceed, n=abort, r=retry +25%): ").strip().lower()
            if response in ("y", "yes"):
                print(f"=================================\n")
                return True
            elif response in ("n", "no"):
                print(f"  Aborting.")
                print(f"=================================\n")
                sys.exit(1)
            elif response in ("r", "retry"):
                print(f"=================================\n")
                return False  # Signal retry
            else:
                print("  Please enter 'y', 'n', or 'r'")

    except Exception as e:
        print(f"  Verification failed: {e}")
        print(f"  Proceeding with calibrated duration anyway.")
        print(f"=================================\n")
        return True


def return_arm_to_start(robot_port: str, starting_positions: dict):
    """Move the arm back to its starting position.

    Uses low-level motor bus access to avoid triggering calibration.
    """
    import time

    try:
        print("\nReturning arm to starting position...")
        bus = _create_motor_bus(robot_port)

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


def has_flag(name: str) -> bool:
    """Check if a boolean flag is present in CLI args and remove it.

    Removes the flag from sys.argv so it doesn't confuse lerobot-record.
    """
    flag = f"--{name}"
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


def _parse_camera_indices() -> list:
    """Extract camera indices from the robot.cameras CLI arg."""
    import re
    cameras_str = parse_arg("robot.cameras")
    if not cameras_str:
        return [0]  # Default to camera 0
    # Extract index_or_path values from the YAML-like camera config
    indices = re.findall(r'index_or_path:\s*(\d+)', cameras_str)
    return [int(i) for i in indices] if indices else [0]


if __name__ == "__main__":
    check_and_clean_dataset_cache()

    # Parse calibration flags (remove from argv before lerobot sees them)
    skip_calibration = has_flag("skip-calibration")
    skip_verification = has_flag("skip-verification")

    # Get robot port and policy parameters
    robot_port = parse_arg("robot.port")
    joint_name = parse_arg("policy.joint_name") or "shoulder_pan.pos"
    position_delta_str = parse_arg("policy.position_delta")
    action_duration_explicit = parse_arg("policy.action_duration")

    # Auto-calibrate action_duration if not explicitly set
    if (robot_port and not skip_calibration
            and action_duration_explicit is None and position_delta_str):
        position_delta = float(position_delta_str)

        # Calibration loop (supports retry with longer duration)
        calibrated_duration = calibrate_action_duration(
            robot_port, joint_name, position_delta
        )

        # Visual verification with sample canvases
        if not skip_verification:
            camera_indices = _parse_camera_indices()
            while True:
                approved = verify_calibration_with_canvases(
                    robot_port, joint_name, position_delta,
                    calibrated_duration, camera_indices
                )
                if approved:
                    break
                # Retry with 25% longer duration
                calibrated_duration = round(calibrated_duration * 1.25, 2)
                print(f"  Retrying with action_duration={calibrated_duration}s")

        # Inject calibrated duration and flag into CLI args
        sys.argv.append(f"--policy.action_duration={calibrated_duration}")
        sys.argv.append("--policy.calibrated_action_duration=true")
        print(f"Using calibrated action_duration={calibrated_duration}s")

    calculate_and_inject_episode_time()
    inject_discrete_action_log_dir()

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
