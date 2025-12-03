"""
Filter Validation Set to Remove Training Overlap

Removes samples from validation session that have (x, y) positions
that were seen in the training session, ensuring true OOD validation.
"""

import argparse
import os
from datetime import datetime
from typing import Set, Tuple, List
import numpy as np
from tqdm import tqdm

from recording_reader import RecordingReader
from recording_writer import RecordingWriter
from toroidal_dot_env import ToroidalDotEnvironment
import config


def extract_samples_from_session(
    session_path: str,
    initial_x: int,
    initial_y: int
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Extract all (position, last_action) samples from a recorded session.

    Args:
        session_path: Path to recorded session
        initial_x: Initial x position
        initial_y: Initial y position

    Returns:
        List of ((x, y), last_action) tuples for each observation.
        First observation has last_action=None.
    """
    reader = RecordingReader(session_path)

    # Create environment with same initial position
    env = ToroidalDotEnvironment(
        initial_x=initial_x,
        initial_y=initial_y
    )

    samples = []
    last_action = None

    # Track (position, last_action) for each observation
    # Session structure: obs0, action0, obs1, action1, obs2, ...
    for event in reader.events:
        if event['type'] == 'observation':
            # Record current position with the action that led to it
            position = env.get_position()
            samples.append((position, last_action))
        elif event['type'] == 'action':
            # Execute action and remember it
            action_value = event['data']['action']
            env.step(action_value)
            last_action = action_value

    return samples


def filter_validation_session(
    train_session_path: str,
    val_session_path: str,
    train_initial_x: int,
    train_initial_y: int,
    val_initial_x: int,
    val_initial_y: int,
    output_dir: str = None
) -> None:
    """
    Create filtered validation session removing positions seen in training.

    Args:
        train_session_path: Path to training session
        val_session_path: Path to validation session
        train_initial_x: Training initial x position
        train_initial_y: Training initial y position
        val_initial_x: Validation initial x position
        val_initial_y: Validation initial y position
        output_dir: Directory for filtered session (default: same as val session)
    """
    print("=" * 70)
    print("Validation Set Overlap Filter")
    print("=" * 70)

    # Extract (position, last_action) samples from both sessions
    print(f"\n[1/4] Extracting samples from training session...")
    train_samples = extract_samples_from_session(
        train_session_path,
        train_initial_x,
        train_initial_y
    )
    # Exclude first sample (has no previous action) from training set
    train_samples_filtered = [s for s in train_samples if s[1] is not None]
    train_sample_set = set(train_samples_filtered)
    print(f"  Training session: {len(train_samples)} total observations")
    print(f"  Training samples (with last_action): {len(train_samples_filtered)}")
    print(f"  Unique (position, last_action) pairs: {len(train_sample_set)}")

    print(f"\n[2/4] Extracting samples from validation session...")
    val_samples = extract_samples_from_session(
        val_session_path,
        val_initial_x,
        val_initial_y
    )
    # Exclude first sample (has no previous action) from validation set
    val_samples_filtered = [s for s in val_samples if s[1] is not None]
    val_sample_set = set(val_samples_filtered)
    print(f"  Validation session: {len(val_samples)} total observations")
    print(f"  Validation samples (with last_action): {len(val_samples_filtered)}")
    print(f"  Unique (position, last_action) pairs: {len(val_sample_set)}")

    # Find overlap
    print(f"\n[3/4] Analyzing overlap...")
    overlap = train_sample_set & val_sample_set
    print(f"  Overlapping (position, last_action) pairs: {len(overlap)}")
    if len(val_sample_set) > 0:
        print(f"  Overlap percentage: {100 * len(overlap) / len(val_sample_set):.1f}%")
    else:
        print(f"  Overlap percentage: N/A (no validation samples)")

    if len(overlap) == 0:
        print("\n  No overlap found! Validation set is already OOD.")
        return

    # Create filtered validation session
    print(f"\n[4/4] Creating filtered validation session...")

    # Determine output directory and session name
    if output_dir is None:
        output_dir = os.path.dirname(val_session_path)

    val_session_name = os.path.basename(val_session_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filtered_session_name = f"{val_session_name}_filtered_{timestamp}"

    # Read validation session and filter
    val_reader = RecordingReader(val_session_path)
    writer = RecordingWriter(base_dir=output_dir, session_name=filtered_session_name)

    # Set metadata from original validation session
    writer.set_session_metadata(
        action_space=val_reader.session_meta.get('action_space', []),
        robot_type=val_reader.session_meta.get('robot_type', 'ToroidalDotRobot')
    )

    kept_count = 0
    removed_count = 0
    obs_idx = 0  # Track which observation we're on
    keep_next_action = False  # Flag to track if we should keep the next action

    for event in tqdm(val_reader.events, desc="  Filtering"):
        if event['type'] == 'observation':
            sample = val_samples[obs_idx]
            obs_idx += 1

            if sample not in train_sample_set:
                # Keep this observation (not in training set)
                frame_path = event['data']['frame_path']
                obs = val_reader._load_frame(frame_path)
                writer.record_observation(obs)
                kept_count += 1
                keep_next_action = True  # Keep the action following this observation
            else:
                removed_count += 1
                keep_next_action = False  # Skip the action following this observation

        elif event['type'] == 'action':
            # Only record action if the previous observation was kept
            if keep_next_action:
                writer.record_action(event['data'])
                keep_next_action = False  # Reset flag

    writer.finalize()

    # Report results
    print(f"\n" + "=" * 70)
    print("Filtering Complete!")
    print("=" * 70)
    print(f"Original validation observations:     {len(val_samples)}")
    print(f"Validation samples (with last_action): {len(val_samples_filtered)}")
    print(f"Samples removed (overlap):            {removed_count}")
    print(f"Samples kept (OOD):                   {kept_count}")
    if len(val_samples_filtered) > 0:
        print(f"Removal rate:                         {100 * removed_count / len(val_samples_filtered):.1f}%")
    print(f"\nFiltered session saved to:")
    print(f"  {os.path.join(output_dir, filtered_session_name)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Filter validation session to remove training overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter specific sessions
  python filter_validation_overlap.py \\
    --train-session saved/sessions/toroidal_dot/session_train \\
    --val-session saved/sessions/toroidal_dot/session_val \\
    --train-x 0 --train-y 112 \\
    --val-x 1 --val-y 112

  # Auto-detect latest random_duration sessions
  python filter_validation_overlap.py --auto-detect
        """
    )

    parser.add_argument(
        '--train-session',
        type=str,
        help='Path to training session directory'
    )
    parser.add_argument(
        '--val-session',
        type=str,
        help='Path to validation session directory'
    )
    parser.add_argument(
        '--train-x',
        type=int,
        help='Training session initial x position'
    )
    parser.add_argument(
        '--train-y',
        type=int,
        help='Training session initial y position'
    )
    parser.add_argument(
        '--val-x',
        type=int,
        help='Validation session initial x position'
    )
    parser.add_argument(
        '--val-y',
        type=int,
        help='Validation session initial y position'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for filtered session (default: same as validation)'
    )
    parser.add_argument(
        '--auto-detect',
        action='store_true',
        help='Auto-detect latest random_duration train/val sessions'
    )

    args = parser.parse_args()

    if args.auto_detect:
        # Find latest random_duration sessions
        import glob

        session_dir = config.TOROIDAL_DOT_RECORDING_DIR
        train_sessions = sorted(glob.glob(
            os.path.join(session_dir, "session_random_duration_*_train_*")
        ))
        val_sessions = sorted(glob.glob(
            os.path.join(session_dir, "session_random_duration_*_val_*")
        ))

        if not train_sessions or not val_sessions:
            print("Error: Could not find random_duration train/val sessions")
            return

        train_session = train_sessions[-1]
        val_session = val_sessions[-1]

        # Parse initial positions from session names
        # Format: session_random_duration_fixed_yXXX_trainxX_train_...
        train_name = os.path.basename(train_session)
        val_name = os.path.basename(val_session)

        # Extract y position
        import re
        train_y_match = re.search(r'fixed_y(\d+)', train_name)
        val_y_match = re.search(r'fixed_y(\d+)', val_name)

        if not train_y_match or not val_y_match:
            print("Error: Could not parse y positions from session names")
            return

        train_y = int(train_y_match.group(1))
        val_y = int(val_y_match.group(1))

        # Extract x position
        train_x_match = re.search(r'trainx(\d+)', train_name)
        val_x_match = re.search(r'valx(\d+)', val_name)

        if not train_x_match or not val_x_match:
            print("Error: Could not parse x positions from session names")
            return

        train_x = int(train_x_match.group(1))
        val_x = int(val_x_match.group(1))

        print(f"Auto-detected sessions:")
        print(f"  Training: {train_name}")
        print(f"    Position: x={train_x}, y={train_y}")
        print(f"  Validation: {val_name}")
        print(f"    Position: x={val_x}, y={val_y}")
        print()

        filter_validation_session(
            train_session,
            val_session,
            train_x,
            train_y,
            val_x,
            val_y,
            args.output_dir
        )
    else:
        # Manual mode - require all arguments
        if not all([
            args.train_session,
            args.val_session,
            args.train_x is not None,
            args.train_y is not None,
            args.val_x is not None,
            args.val_y is not None
        ]):
            parser.error(
                "All of --train-session, --val-session, --train-x, --train-y, "
                "--val-x, --val-y are required unless using --auto-detect"
            )

        filter_validation_session(
            args.train_session,
            args.val_session,
            args.train_x,
            args.train_y,
            args.val_x,
            args.val_y,
            args.output_dir
        )


if __name__ == "__main__":
    main()
