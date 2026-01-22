"""
Create Staged Training/Validation Splits

Creates progressively larger training and validation subsessions from an original session.
Starts with an initial size (default 10), doubles each stage until the original session
size is reached. Each stage creates a 70/30 train/validate split (configurable).

Usage:
    python create_staged_splits.py --session-path saved/sessions/so101/my_session
    python create_staged_splits.py --session-path saved/sessions/so101/my_session --initial-size 20 --train-ratio 0.8

Output naming convention:
    {session}_stage{N}_train_{train_size}
    {session}_stage{N}_validate_{validate_size}
"""

import argparse
import json
import math
import shutil
from pathlib import Path


def load_session_metadata(session_dir: str) -> dict:
    """Load session metadata from session_meta.json."""
    meta_path = Path(session_dir) / "session_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def load_session_events(session_dir: str) -> list:
    """Load all events from session event shards."""
    session_path = Path(session_dir)
    events = []

    # Find all event shard files
    shard_files = sorted(session_path.glob("events_shard_*.jsonl"))

    for shard_file in shard_files:
        with open(shard_file, "r") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

    return events


def get_observation_count(metadata: dict, events: list) -> int:
    """Get total number of observations in a session."""
    total_steps = metadata.get("total_steps", len(events))
    return total_steps // 2  # observations and actions alternate


def create_session_from_events(
    source_path: Path,
    dest_path: Path,
    events: list,
    original_metadata: dict,
    new_session_name: str,
    step_offset: int,
) -> None:
    """
    Create a new session directory with the given events.

    Args:
        source_path: Path to source session (for copying frames)
        dest_path: Path to destination session
        events: List of events for this split
        original_metadata: Original session metadata
        new_session_name: Name for the new session
        step_offset: Step offset to subtract for renumbering
    """
    # Create directory structure
    dest_path.mkdir(parents=True)
    frames_dir = dest_path / "frames"
    frames_dir.mkdir()

    # Renumber events and copy frames
    renumbered_events = []
    for event in events:
        new_event = event.copy()
        old_step = event["step"]
        new_step = old_step - step_offset
        new_event["step"] = new_step

        # If it's an observation, update frame path and copy frame
        if event.get("type") == "observation":
            old_frame_path = event.get("data", {}).get("frame_path", "")
            if old_frame_path:
                # Generate new frame filename based on new step
                new_frame_filename = f"frame_{new_step:06d}.jpg"
                new_frame_path = f"frames/{new_frame_filename}"

                # Copy frame file
                src_frame = source_path / old_frame_path
                dst_frame = dest_path / new_frame_path
                if src_frame.exists():
                    shutil.copy2(src_frame, dst_frame)
                else:
                    print(f"  Warning: Source frame not found: {src_frame}")

                # Update frame path in event
                new_event["data"] = event.get("data", {}).copy()
                new_event["data"]["frame_path"] = new_frame_path

        renumbered_events.append(new_event)

    # Calculate shard info
    shard_size = original_metadata.get("shard_size", 1000)
    total_steps = len(renumbered_events)
    total_shards = (total_steps + shard_size - 1) // shard_size

    # Write events to shards
    for shard_idx in range(total_shards):
        shard_start = shard_idx * shard_size
        shard_end = min(shard_start + shard_size, total_steps)
        shard_events = renumbered_events[shard_start:shard_end]

        shard_filename = f"events_shard_{shard_idx:03d}.jsonl"
        shard_path = dest_path / shard_filename

        with open(shard_path, "w") as f:
            for event in shard_events:
                f.write(json.dumps(event) + "\n")

    # Create new metadata
    new_metadata = original_metadata.copy()
    new_metadata["session_name"] = new_session_name
    new_metadata["total_steps"] = total_steps
    new_metadata["total_shards"] = total_shards
    new_metadata["split_from"] = original_metadata.get("session_name", str(source_path.name))
    new_metadata["split_step_offset"] = step_offset

    meta_path = dest_path / "session_meta.json"
    with open(meta_path, "w") as f:
        json.dump(new_metadata, f, indent=2)

    print(f"  Created {new_session_name}: {total_steps // 2} observations")


def create_staged_splits(
    session_path: str,
    initial_size: int = 10,
    train_ratio: float = 0.7,
) -> list[tuple[str, str]]:
    """
    Create staged training/validation splits from a session.

    Args:
        session_path: Path to the source session directory
        initial_size: Starting size for the first stage (default 10)
        train_ratio: Fraction of observations for training (default 0.7)

    Returns:
        List of (train_path, validate_path) tuples for each stage
    """
    session_path = Path(session_path).resolve()
    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    # Load metadata and events
    metadata = load_session_metadata(str(session_path))
    events = load_session_events(str(session_path))

    if not metadata:
        raise ValueError(f"Could not load metadata from {session_path}")
    if not events:
        raise ValueError(f"Could not load events from {session_path}")

    total_observations = get_observation_count(metadata, events)
    session_name = session_path.name
    parent_dir = session_path.parent

    print(f"Original session: {session_name}")
    print(f"Total observations: {total_observations}")
    print(f"Initial size: {initial_size}, Train ratio: {train_ratio}")
    print()

    if initial_size > total_observations:
        raise ValueError(
            f"Initial size ({initial_size}) exceeds total observations ({total_observations})"
        )

    created_splits = []
    stage = 1
    current_size = initial_size

    # Process stages until we've handled the full session
    while True:
        # Determine the size for this stage
        if current_size >= total_observations:
            stage_size = total_observations
        else:
            stage_size = current_size

        # Calculate train/validate sizes (round train up)
        train_size = math.ceil(stage_size * train_ratio)
        validate_size = stage_size - train_size

        print(f"Stage {stage} (size {stage_size}):")

        # Calculate step boundaries
        # observations are at steps 0, 2, 4, ... and actions at 1, 3, 5, ...
        # For N observations, we need steps 0 to (N*2 - 1)
        train_end_step = train_size * 2  # First step of validate
        stage_end_step = stage_size * 2  # First step after this stage

        # Extract events for train and validate
        train_events = [e for e in events if e["step"] < train_end_step]
        validate_events = [e for e in events if train_end_step <= e["step"] < stage_end_step]

        # Create output paths
        train_name = f"{session_name}_stage{stage}_train_{train_size}"
        validate_name = f"{session_name}_stage{stage}_validate_{validate_size}"

        train_path = parent_dir / train_name
        validate_path = parent_dir / validate_name

        # Check if output directories already exist
        if train_path.exists():
            raise FileExistsError(f"Output directory already exists: {train_path}")
        if validate_path.exists():
            raise FileExistsError(f"Output directory already exists: {validate_path}")

        # Create train session
        create_session_from_events(
            session_path,
            train_path,
            train_events,
            metadata,
            train_name,
            step_offset=0,
        )

        # Create validate session
        create_session_from_events(
            session_path,
            validate_path,
            validate_events,
            metadata,
            validate_name,
            step_offset=train_end_step,
        )

        created_splits.append((str(train_path), str(validate_path)))
        print()

        # Check if we've reached the full session
        if stage_size >= total_observations:
            break

        # Double the size for next stage
        stage += 1
        current_size *= 2

    return created_splits


def main():
    parser = argparse.ArgumentParser(
        description="Create staged training/validation splits from a session"
    )
    parser.add_argument(
        "--session-path",
        required=True,
        help="Path to the source session directory",
    )
    parser.add_argument(
        "--initial-size",
        type=int,
        default=10,
        help="Starting size for the first stage (default: 10)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of observations for training (default: 0.7)",
    )

    args = parser.parse_args()

    if args.train_ratio <= 0 or args.train_ratio >= 1:
        print("Error: train-ratio must be between 0 and 1 (exclusive)")
        return 1

    if args.initial_size < 2:
        print("Error: initial-size must be at least 2")
        return 1

    try:
        splits = create_staged_splits(
            args.session_path,
            args.initial_size,
            args.train_ratio,
        )
        print(f"Successfully created {len(splits)} staged splits:")
        for i, (train_path, validate_path) in enumerate(splits, 1):
            print(f"  Stage {i}:")
            print(f"    Train: {train_path}")
            print(f"    Validate: {validate_path}")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
