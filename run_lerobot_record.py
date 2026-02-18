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

# Import custom policies to trigger registration before lerobot parses args
# Use explicit submodule imports to bypass namespace package shadowing when
# running from the project root (outer directory lacks __init__.py)
from lerobot_policy_simple_joint.configuration_simple_joint import SimpleJointConfig  # noqa: F401
from lerobot_policy_simple_joint.modeling_simple_joint import SimpleJointPolicy  # noqa: F401
from lerobot_policy_simple_joint.configuration_multi_secondary_joint import MultiSecondaryJointConfig  # noqa: F401
from lerobot_policy_simple_joint.modeling_multi_secondary_joint import MultiSecondaryJointPolicy  # noqa: F401

# Patch lerobot record_loop to fix infinite loop during reset phase.
# Bug: when no policy/teleop is provided (reset between episodes), the `continue`
# on the "no policy" branch skips the timestamp update, so the while loop never
# exits. Fix: handle reset phase ourselves - call policy.get_reset_motor_targets(),
# command those motors via the robot's connected bus, then sleep.
import logging as _logging
import time as _time_module
import lerobot.scripts.lerobot_record as _record_mod

_original_record_loop = _record_mod.record_loop
_reset_state = {}  # Stores policy reference between episode and reset calls


def _patched_record_loop(*args, **kwargs):
    """Patch for lerobot record_loop infinite loop during reset phase.

    The bug: when policy=None and teleop=None (reset between episodes),
    `continue` at line 368 skips the timestamp update at line 398, causing
    `while timestamp < control_time_s` to loop forever.

    This patch:
    1. Stores the policy reference during episode recording calls
    2. During reset calls: calls policy.reset() to select new targets, then
       calls policy.get_reset_motor_targets() to get motors to command,
       sends those commands via the robot's connected bus, and sleeps
    3. The policy's _secondary_target_locked flag preserves the new target
       across the subsequent episode reset() call (prevents double-reset)
    """
    policy = kwargs.get('policy')
    teleop = kwargs.get('teleop')
    dataset = kwargs.get('dataset')
    control_time_s = kwargs.get('control_time_s', 0)

    # Episode recording call - store policy reference for reset phase
    if policy is not None:
        _reset_state['policy'] = policy

    # Reset phase (no policy, no teleop, no dataset) - handle ourselves
    if policy is None and teleop is None and dataset is None and control_time_s > 0:
        stored_policy = _reset_state.get('policy')
        robot = kwargs.get('robot')

        if stored_policy is not None and hasattr(stored_policy, 'get_reset_motor_targets'):
            old_secondary = stored_policy._current_secondary_target
            stored_policy.reset()
            motor_targets = stored_policy.get_reset_motor_targets()

            if motor_targets and robot is not None and hasattr(robot, 'bus'):
                try:
                    robot.bus.sync_write("Goal_Position", motor_targets)
                    new_secondary = stored_policy._current_secondary_target
                    delta_str = ""
                    if old_secondary is not None and new_secondary is not None:
                        signed = new_secondary - old_secondary
                        delta_str = f"  (delta={signed:+.1f})"
                    _logging.info(
                        f"Reset: commanding {motor_targets}{delta_str}, "
                        f"waiting {control_time_s:.1f}s to settle"
                    )
                except Exception as e:
                    _logging.warning(f"Reset servo command failed: {e}")

            # Lock target so next episode's reset() preserves it
            stored_policy._secondary_target_locked = True
        else:
            _logging.info(f"Reset phase: waiting {control_time_s:.1f}s...")

        _time_module.sleep(control_time_s)
        return

    return _original_record_loop(*args, **kwargs)


_record_mod.record_loop = _patched_record_loop

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


def inject_reset_time_for_multi_secondary():
    """For multi_secondary_joint policy, inject reset_time_s for secondary servo settling.

    The reset_time_s gives the secondary joint servo time to reach its new target
    position between episodes. Uses 3x action_duration as a conservative default.
    """
    policy_type = parse_arg("policy.type")
    if policy_type != "multi_secondary_joint":
        return

    if parse_arg("dataset.reset_time_s") is not None:
        return  # User already specified it

    action_duration_str = parse_arg("policy.action_duration")
    action_duration = float(action_duration_str) if action_duration_str else 0.5

    # Use 3x action_duration as conservative settling time for secondary joint
    reset_time = max(3.0, action_duration * 3)
    sys.argv.append(f"--dataset.reset_time_s={reset_time}")
    print(f"Auto-set reset_time_s={reset_time:.1f}s for secondary joint settling "
          f"between episodes")


def _create_motor_bus(robot_port: str, robot_id: str = None):
    """Create a FeetechMotorsBus with SO-101 motor configuration.

    If robot_id is provided, loads calibration from the default calibration
    directory so that sync_read/sync_write use normalized positions.

    Returns a connected bus. Caller is responsible for disconnecting.
    """
    import json
    from pathlib import Path
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode, MotorCalibration

    calibration = None
    if robot_id:
        cal_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so101_follower"
        cal_path = cal_dir / f"{robot_id}.json"
        if cal_path.is_file():
            with open(cal_path) as f:
                cal_dict = json.load(f)
            calibration = {
                motor: MotorCalibration(**cal_data)
                for motor, cal_data in cal_dict.items()
            }
            print(f"  Loaded calibration from {cal_path}")
        else:
            print(f"  Warning: No calibration file found at {cal_path}")

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
        calibration=calibration,
    )
    bus.connect()
    return bus


# SO-101 joint names matching motor bus keys
SO101_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def _wait_for_settle(bus, joint_name: str, start_pos: float,
                     timeout: float = 5.0,
                     threshold: float = 0.5, stable_count: int = 5,
                     poll_interval: float = 0.010,
                     departure_threshold: float = 0.5) -> float:
    """Poll Present_Position until the joint moves and then stops.

    Two-phase approach:
    1. Wait for departure: poll until position differs from start_pos by
       more than departure_threshold (confirms servo started moving).
    2. Wait for settle: poll until position range over stable_count
       consecutive reads is below threshold.

    Args:
        bus: Connected FeetechMotorsBus
        joint_name: Motor bus joint name (e.g. "shoulder_pan")
        start_pos: Position before the move command was sent
        timeout: Max wait time in seconds (applies across both phases)
        threshold: Position range (in motor units) to consider settled
        stable_count: Number of consecutive stable reads required
        poll_interval: Seconds between position polls
        departure_threshold: Min distance from start_pos to confirm movement started

    Returns:
        Elapsed time until settled, or timeout value if not settled.
    """
    import time

    start = time.time()
    departed = False
    recent = []

    while time.time() - start < timeout:
        time.sleep(poll_interval)
        pos = bus.sync_read("Present_Position")[joint_name]

        # Phase 1: detect that movement has started
        if not departed:
            if abs(pos - start_pos) > departure_threshold:
                departed = True
                recent = []  # reset for phase 2
            continue

        # Phase 2: wait for position to stabilize
        recent.append(pos)

        if len(recent) > stable_count:
            recent.pop(0)

        if len(recent) == stable_count:
            if max(recent) - min(recent) < threshold:
                return time.time() - start

    if not departed:
        print(f"    Warning: servo never departed from start_pos={start_pos:.1f} "
              f"(timeout={timeout}s)")
    return timeout


def calibrate_action_duration(robot_port: str, joint_name: str,
                              position_delta: float,
                              robot_id: str = None) -> float:
    """Measure servo settling time to determine optimal action_duration.

    Sends a test movement command and polls Present_Position to detect
    when the servo stops moving. Returns the measured time with a 20%
    safety margin.

    Args:
        robot_port: Serial port (e.g. "COM8")
        joint_name: Policy joint name (e.g. "shoulder_pan.pos")
        position_delta: Movement magnitude to test with
        robot_id: Robot ID for loading calibration file

    Returns:
        Calibrated action_duration in seconds
    """
    # Map policy joint name (e.g. "shoulder_pan.pos") to motor bus name
    motor_name = joint_name.replace(".pos", "")
    if motor_name not in SO101_JOINT_NAMES:
        print(f"  Warning: unknown joint '{motor_name}', using default 0.5s")
        return 0.5

    print(f"\n=== Calibrating action_duration ===")
    print(f"  Joint: {joint_name} (motor: {motor_name})")
    print(f"  Position delta: {position_delta}")

    try:
        bus = _create_motor_bus(robot_port, robot_id)

        # Read current position
        origin_pos = bus.sync_read("Present_Position")[motor_name]
        print(f"  Starting position: {origin_pos:.1f}")

        settle_times = []

        def _move_and_log(label, before_pos, target_pos):
            """Execute a movement, wait for settle, and log positions."""
            target_pos = max(-100, min(100, target_pos))
            expected_delta = target_pos - before_pos
            print(f"  [{label}] Before: {before_pos:.1f} | Target: {target_pos:.1f} "
                  f"(delta={expected_delta:+.1f})")
            bus.sync_write("Goal_Position", {motor_name: target_pos})
            settle_time = _wait_for_settle(bus, motor_name, before_pos)
            after_pos = bus.sync_read("Present_Position")[motor_name]
            actual_delta = after_pos - before_pos
            err = abs(actual_delta - expected_delta)
            match = "YES" if err < 1.0 else "NO"
            print(f"           After:  {after_pos:.1f} | Actual delta: {actual_delta:+.1f} "
                  f"| Match: {match} (err={err:.1f})")
            print(f"           Settle time: {settle_time:.3f}s")
            settle_times.append(settle_time)
            return after_pos

        # Test 1: Move positive (+position_delta)
        pos = _move_and_log("MOVE+", origin_pos, origin_pos + position_delta)

        # Test 2: Return to origin
        pos = _move_and_log("RETURN", pos, origin_pos)

        # Test 3: Move negative (-position_delta)
        pos = _move_and_log("MOVE-", pos, origin_pos - position_delta)

        # Test 4: Return to origin
        pos = _move_and_log("RETURN", pos, origin_pos)

        bus.disconnect()

        # Use the worst-case settle time with 20% safety margin
        measured = max(settle_times)
        calibrated = round(measured * 1.2, 2)
        print(f"  Settle times: {[f'{t:.3f}s' for t in settle_times]}")
        print(f"  Worst case: {measured:.3f}s -> calibrated: {calibrated:.2f}s (1.2x margin)")
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
                                     robot_id: str = None) -> bool:
    """Verify calibrated action_duration using the full recording pipeline.

    Runs the real pipeline end-to-end:
    1. Records a short test episode via lerobot-record subprocess
    2. Converts to explorer format via convert_lerobot_to_explorer.py
    3. Builds canvases using the same code path as staged_training.py
    4. Displays canvases for user approval

    Args:
        robot_port: Serial port (e.g. "COM8")
        joint_name: Policy joint name (e.g. "shoulder_pan.pos")
        position_delta: Movement magnitude
        action_duration: Calibrated action duration to test
        robot_id: Robot ID for loading calibration file

    Returns:
        True if user approves, False if user wants to retry with longer duration
    """
    import subprocess
    import tempfile
    import shutil
    import os
    import time
    import traceback
    import numpy as np
    from pathlib import Path

    cameras_str = parse_arg("robot.cameras")
    camera_keys = _parse_camera_keys()

    print(f"\n=== Calibration Verification ===")
    print(f"  Running full pipeline: record → convert → build canvases")
    print(f"  Action duration: {action_duration}s")
    print(f"  Test sequence: [MOVE+, STAY, MOVE-, STAY]")

    temp_repo = f"_temp/eval_calibration_verify_{int(time.time())}_{os.getpid()}"
    dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / temp_repo
    temp_session_dir = None

    try:
        # === Step A: Record via lerobot-record subprocess ===
        print(f"\n  Step 1/3: Recording test episode...")
        cmd = [
            sys.executable, os.path.abspath(__file__),
            f"--robot.type=so101_follower",
            f"--robot.port={robot_port}",
        ]
        if robot_id:
            cmd.append(f"--robot.id={robot_id}")
        if cameras_str:
            cmd.append(f"--robot.cameras={cameras_str}")
        cmd.extend([
            "--policy.type=simple_joint",
            f"--policy.joint_name={joint_name}",
            f"--policy.action_duration={action_duration}",
            f"--policy.position_delta={position_delta}",
            "--policy.action_sequence=[1, 0, 2, 0]",
            f"--dataset.repo_id={temp_repo}",
            "--dataset.num_episodes=1",
            "--dataset.single_task=Calibration verification",
            "--dataset.push_to_hub=false",
            "--skip-calibration",
            "--skip-verification",
        ])

        result = subprocess.run(cmd, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"lerobot-record subprocess failed with code {result.returncode}")

        if not dataset_path.exists():
            raise RuntimeError(f"Dataset not found at {dataset_path}")

        # === Step B: Convert via convert_lerobot_to_explorer.py ===
        print(f"\n  Step 2/3: Converting to explorer format...")
        temp_session_dir = tempfile.mkdtemp(prefix="cal_verify_session_")
        converter_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "convert_lerobot_to_explorer.py")
        convert_cmd = [
            sys.executable, converter_script,
            "--lerobot-path", str(dataset_path),
            "--output-dir", temp_session_dir,
            "--cameras", *camera_keys,
            "--joint-name", joint_name,
            "--action-duration", str(action_duration),
        ]
        if len(camera_keys) > 1:
            convert_cmd.extend(["--stack-cameras", "vertical"])

        result = subprocess.run(convert_cmd, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Converter subprocess failed with code {result.returncode}")

        # Find the generated session directory
        session_subdirs = [d for d in os.listdir(temp_session_dir)
                           if os.path.isdir(os.path.join(temp_session_dir, d))]
        if not session_subdirs:
            raise RuntimeError(f"No session directory found in {temp_session_dir}")
        session_path = os.path.join(temp_session_dir, session_subdirs[0])

        # === Step C: Build canvases via staged_training.py code path ===
        print(f"\n  Step 3/3: Building canvases...")
        from session_explorer_lib import (
            load_session_events, extract_observations, extract_actions,
            prebuild_all_canvases
        )
        import config

        events = load_session_events(session_path)
        observations = extract_observations(events, session_path)
        actions = extract_actions(events)

        canvas_cache, detected_frame_size = prebuild_all_canvases(
            session_path, observations, actions,
            config.AutoencoderConcatPredictorWorldModelConfig,
        )

        if not canvas_cache:
            raise RuntimeError("No canvases were built from the verification recording")

        # === Step D: Display canvases ===
        import cv2

        # Select canvases to show: first, middle, last
        sorted_indices = sorted(canvas_cache.keys())
        display_indices = [sorted_indices[0]]
        if len(sorted_indices) > 2:
            display_indices.append(sorted_indices[len(sorted_indices) // 2])
        if len(sorted_indices) > 1:
            display_indices.append(sorted_indices[-1])

        # Stack selected canvases vertically
        canvas_images = []
        for idx in display_indices:
            canvas = canvas_cache[idx]['canvas']
            # Convert RGB (from build_canvas) to BGR for cv2.imwrite
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            canvas_images.append(canvas_bgr)

        composite = np.vstack(canvas_images)

        # Add label at top
        label_height = 30
        labeled = np.zeros((composite.shape[0] + label_height, composite.shape[1], 3),
                           dtype=np.uint8)
        labeled[label_height:, :] = composite
        label_text = (f"action_duration={action_duration}s | "
                      f"{len(observations)} frames | "
                      f"showing canvas indices {display_indices}")
        cv2.putText(labeled, label_text, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Save and open
        preview_path = os.path.join(tempfile.gettempdir(), "calibration_verification.png")
        cv2.imwrite(preview_path, labeled)
        print(f"\n  Saved verification image to: {preview_path}")
        print(f"  GREEN separators = MOVE+ | BLUE separators = MOVE- | RED separators = STAY")
        print(f"  Movement should be visible across GREEN/BLUE separators, not RED")
        os.startfile(preview_path)

        # === Step E: Clean up temp data (before prompting user) ===
        shutil.rmtree(str(dataset_path), ignore_errors=True)
        # Also clean parent _temp/ dir if empty
        temp_parent = dataset_path.parent
        if temp_parent.exists() and not any(temp_parent.iterdir()):
            temp_parent.rmdir()
        shutil.rmtree(temp_session_dir, ignore_errors=True)
        temp_session_dir = None  # Mark as cleaned

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
        traceback.print_exc()
        # Clean up on failure
        shutil.rmtree(str(dataset_path), ignore_errors=True)
        if temp_session_dir:
            shutil.rmtree(temp_session_dir, ignore_errors=True)
        # Ask user what to do instead of proceeding silently
        while True:
            response = input("\n  Verification failed. [r/n] "
                             "(r=retry, n=abort): ").strip().lower()
            if response in ("r", "retry"):
                print(f"=================================\n")
                return False  # Signal retry (caller will increase duration 25%)
            elif response in ("n", "no"):
                print(f"  Aborting.")
                print(f"=================================\n")
                sys.exit(1)
            else:
                print("  Please enter 'r' or 'n'")


def return_arm_to_start(robot_port: str, starting_positions: dict,
                        robot_id: str = None):
    """Move the arm back to its starting position.

    Uses low-level motor bus access to avoid triggering calibration.
    """
    import time

    try:
        print("\nReturning arm to starting position...")
        bus = _create_motor_bus(robot_port, robot_id)

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


def capture_starting_position(robot_port: str, robot_id: str = None) -> dict:
    """Capture all motor positions before recording starts."""
    try:
        bus = _create_motor_bus(robot_port, robot_id)
        positions = bus.sync_read("Present_Position")
        bus.disconnect()
        print(f"  Captured starting positions: {positions}")
        return positions
    except Exception as e:
        print(f"  Warning: Could not capture starting position: {e}")
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


def _parse_camera_keys() -> list:
    """Extract camera key names from the robot.cameras CLI arg.

    Parses YAML-like camera config to get key names like 'base_0_rgb',
    'left_wrist_0_rgb'. These are needed by convert_lerobot_to_explorer.py.
    """
    import re
    cameras_str = parse_arg("robot.cameras")
    if not cameras_str:
        return ["base_0_rgb"]
    # Extract key names: "key_name: {type: ...}"
    keys = re.findall(r'(\w+)\s*:\s*\{', cameras_str)
    return keys if keys else ["base_0_rgb"]


if __name__ == "__main__":
    check_and_clean_dataset_cache()

    # Parse calibration flags (remove from argv before lerobot sees them)
    skip_calibration = has_flag("skip-calibration")
    skip_verification = has_flag("skip-verification")

    # Get robot port and policy parameters
    robot_port = parse_arg("robot.port")
    robot_id = parse_arg("robot.id")
    joint_name = parse_arg("policy.joint_name") or "shoulder_pan.pos"
    position_delta_str = parse_arg("policy.position_delta")

    # Auto-calibrate action_duration
    if robot_port and not skip_calibration and position_delta_str:
        position_delta = float(position_delta_str)

        # Calibration using motor position settling
        calibrated_duration = calibrate_action_duration(
            robot_port, joint_name, position_delta, robot_id
        )

        # Visual verification via full pipeline (record → convert → canvas)
        if not skip_verification:
            import time as _time
            _time.sleep(2)  # Let COM port settle after calibration
            input(f"\n  Calibrated action_duration={calibrated_duration}s. "
                  f"Press Enter to start verification...")
            while True:
                approved = verify_calibration_with_canvases(
                    robot_port, joint_name, position_delta,
                    calibrated_duration, robot_id
                )
                if approved:
                    break
                # Retry with 25% longer duration
                calibrated_duration = round(calibrated_duration * 1.25, 2)
                print(f"  Retrying with action_duration={calibrated_duration}s")

        # Inject calibrated duration and flag into CLI args
        # Remove any explicit --policy.action_duration from argv (calibration overrides it)
        sys.argv = [arg for arg in sys.argv if not arg.startswith("--policy.action_duration=")]
        sys.argv.append(f"--policy.action_duration={calibrated_duration}")
        sys.argv.append("--policy.calibrated_action_duration=true")
        print(f"Using calibrated action_duration={calibrated_duration}s")

    calculate_and_inject_episode_time()
    inject_discrete_action_log_dir()
    inject_reset_time_for_multi_secondary()

    # Capture starting position
    starting_positions = {}
    if robot_port:
        starting_positions = capture_starting_position(robot_port, robot_id)

    # Print full command for debugging
    print_command_line()

    # Run the recording
    try:
        record()
    finally:
        # Return arm to starting position after recording
        if robot_port and starting_positions:
            return_arm_to_start(robot_port, starting_positions, robot_id)
