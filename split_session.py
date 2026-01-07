"""
Split Session Script

Splits a recorded session into two parts based on a specified number of observations.
The original session is preserved (untouched).

Usage:
    python split_session.py --session-path saved/sessions/so101/session_name --size 100

This creates two new sessions:
    - session_name_part1_100 (first 100 observations)
    - session_name_part2_XXX (remaining observations)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from session_explorer_lib import load_session_metadata, load_session_events


def split_session(session_path: str, size: int) -> tuple[str, str]:
    """
    Split a session into two parts.

    Args:
        session_path: Path to the source session directory
        size: Number of observations for the first split

    Returns:
        Tuple of (part1_path, part2_path)
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

    # Calculate total observations
    total_steps = metadata.get("total_steps", len(events))
    total_observations = total_steps // 2  # observations and actions alternate

    if size < 1:
        raise ValueError(f"Size must be at least 1, got {size}")
    if size >= total_observations:
        raise ValueError(
            f"Size ({size}) must be less than total observations ({total_observations})"
        )

    # Calculate split point in steps (obs at step 0, action at step 1, obs at step 2, etc.)
    # To get first N observations, we need steps 0 to (N*2 - 1)
    split_step = size * 2  # First step of part 2

    # Split events
    events_part1 = [e for e in events if e["step"] < split_step]
    events_part2 = [e for e in events if e["step"] >= split_step]

    remaining_observations = total_observations - size

    # Create output directory names
    session_name = session_path.name
    parent_dir = session_path.parent

    part1_name = f"{session_name}_part1_{size}"
    part2_name = f"{session_name}_part2_{remaining_observations}"

    part1_path = parent_dir / part1_name
    part2_path = parent_dir / part2_name

    # Check if output directories already exist
    if part1_path.exists():
        raise FileExistsError(f"Output directory already exists: {part1_path}")
    if part2_path.exists():
        raise FileExistsError(f"Output directory already exists: {part2_path}")

    # Create part 1
    print(f"Creating part 1: {part1_path}")
    _create_split_session(
        session_path,
        part1_path,
        events_part1,
        metadata,
        part1_name,
        step_offset=0,
    )

    # Create part 2
    print(f"Creating part 2: {part2_path}")
    _create_split_session(
        session_path,
        part2_path,
        events_part2,
        metadata,
        part2_name,
        step_offset=split_step,
    )

    return str(part1_path), str(part2_path)


def _create_split_session(
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
        source_path: Path to source session
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

    print(f"  Created {total_steps} events ({total_steps // 2} observations) in {total_shards} shard(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Split a session into two parts based on observation count"
    )
    parser.add_argument(
        "--session-path",
        required=True,
        help="Path to the source session directory",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Number of observations for the first split",
    )

    args = parser.parse_args()

    try:
        part1_path, part2_path = split_session(args.session_path, args.size)
        print(f"\nSuccessfully split session:")
        print(f"  Part 1: {part1_path}")
        print(f"  Part 2: {part2_path}")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
