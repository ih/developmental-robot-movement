"""
Convert LeRobot v3.0 dataset to concat_world_model_explorer session format.

This script reads a LeRobot v3.0 dataset (Parquet + MP4) and converts it to
the concat_world_model_explorer session format with:
- session_meta.json
- events_shard_*.jsonl
- frames/ directory with JPG images

Usage:
    python convert_lerobot_to_explorer.py \\
        --lerobot-path /path/to/lerobot/dataset \\
        --output-dir saved/sessions/so101 \\
        --cameras base_0_rgb left_wrist_0_rgb \\
        --stack-cameras vertical

    # Can also specify HuggingFace repo_id to download from Hub:
    python convert_lerobot_to_explorer.py \\
        --lerobot-path username/dataset-name \\
        --output-dir saved/sessions/so101
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Try to import av for video decoding
try:
    import av
except ImportError:
    print("Error: PyAV is required for video decoding.")
    print("Install it with: pip install av")
    sys.exit(1)

# Try to import pandas for Parquet reading
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required for reading Parquet files.")
    print("Install it with: pip install pandas pyarrow")
    sys.exit(1)


# SO-101 joint names
SO101_JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos"
]


def resolve_dataset_path(lerobot_path_or_repo: str) -> Path:
    """Resolve dataset path, downloading from Hub if needed.

    Args:
        lerobot_path_or_repo: Either a local path or a HuggingFace repo_id

    Returns:
        Local path to the dataset
    """
    # Check if it's a local path
    local_path = Path(lerobot_path_or_repo)
    if local_path.exists():
        return local_path

    # Assume it's a Hub repo_id, download to cache
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required to download from Hub.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading dataset from Hub: {lerobot_path_or_repo}")
    cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / lerobot_path_or_repo

    snapshot_download(
        repo_id=lerobot_path_or_repo,
        repo_type="dataset",
        local_dir=str(cache_dir)
    )
    return cache_dir


@dataclass
class DiscreteActionLog:
    """Parsed discrete action log with header and decisions."""
    header: dict
    decisions: list

    @property
    def action_duration(self) -> float:
        return self.header.get("action_duration", 0.5)

    @property
    def position_delta(self) -> float:
        return self.header.get("position_delta", 0.1)

    @property
    def joint_name(self) -> str:
        return self.header.get("joint_name", "shoulder_pan.pos")



def load_discrete_action_log(log_path: Path) -> Optional[DiscreteActionLog]:
    """Load discrete action log from JSONL file.

    First line is header with recording parameters.
    Subsequent lines are action decisions.

    Returns:
        DiscreteActionLog with header and decisions, or None if file not found
    """
    if not log_path.exists():
        return None

    header = None
    decisions = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "header":
                header = entry
            elif entry.get("type") == "action":
                decisions.append(entry)

    if header is None:
        return None

    return DiscreteActionLog(header=header, decisions=decisions)


def load_decision_bearing_logs(log_dir: Path) -> List[DiscreteActionLog]:
    """Load action logs from a directory, filtering out header-only (spurious) files.

    During multi-episode recording, the reset-phase patch calls policy.reset()
    between episodes, creating spurious header-only log files. This function
    returns only logs that contain actual action decisions, in file-sorted order.

    Args:
        log_dir: Path to discrete_action_logs directory

    Returns:
        List of DiscreteActionLog objects that have at least one decision entry,
        ordered by filename. Index 0 = first real episode, index 1 = second, etc.
    """
    log_files = sorted(log_dir.glob("episode_*.jsonl"))
    result = []
    for lf in log_files:
        log = load_discrete_action_log(lf)
        if log and log.decisions:
            result.append(log)
    return result


def get_decision_frame_indices(
    log: DiscreteActionLog,
    total_frames: int,
) -> List[Tuple[int, int]]:
    """Map each action decision to a frame index using logged frame indices.

    Each decision in the log must have a 'frame_index' field recorded by the
    policy during recording. This gives exact frame-to-decision correspondence
    since select_action() is called once per video frame.

    Args:
        log: The discrete action log with decisions (must have frame_index)
        total_frames: Total number of frames in the episode (for validation)

    Returns:
        List of (frame_index, discrete_action) tuples

    Raises:
        ValueError: If decisions are missing the frame_index field
    """
    if not log.decisions:
        return []

    # Require frame_index in all decisions
    missing = [i for i, d in enumerate(log.decisions) if "frame_index" not in d]
    if missing:
        raise ValueError(
            f"Discrete action log is missing 'frame_index' field in {len(missing)} "
            f"decision(s) (first at index {missing[0]}). "
            f"Re-record with the updated SimpleJointPolicy that logs frame indices."
        )

    decision_frames = []
    for decision in log.decisions:
        frame_idx = decision["frame_index"]
        if frame_idx < 0 or frame_idx >= total_frames:
            print(f"  Warning: frame_index {frame_idx} out of range [0, {total_frames}), clamping")
            frame_idx = max(0, min(total_frames - 1, frame_idx))
        decision_frames.append((frame_idx, decision["discrete_action"]))

    return decision_frames


@dataclass
class ConversionConfig:
    """Configuration for dataset conversion."""
    lerobot_path: str
    output_dir: str
    cameras: List[str]
    stack_cameras: str  # "vertical", "horizontal", or "single"
    joint_name: str
    joint_index: int
    action_duration: float  # Duration of move actions in seconds
    velocity_threshold: float  # For fallback discretization
    frame_size: Tuple[int, int]  # (H, W) for single camera
    shard_size: int
    session_prefix: str
    combine_episodes: bool = False  # Combine all episodes into a single session


class LeRobotV3Reader:
    """Reads LeRobot v3.0 dataset format."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        self.videos_path = self.dataset_path / "videos"

        # Load metadata
        self.info = self._load_info()
        self.episodes = self._load_episodes()
        self.tasks = self._load_tasks()

    def _load_info(self) -> dict:
        """Load dataset info.json."""
        info_path = self.meta_path / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        with open(info_path) as f:
            return json.load(f)

    def _load_episodes(self) -> pd.DataFrame:
        """Load episode metadata from parquet files."""
        episodes_path = self.meta_path / "episodes.parquet"
        if episodes_path.exists():
            return pd.read_parquet(episodes_path)

        # Try directory structure
        episodes_dir = self.meta_path / "episodes"
        if episodes_dir.exists():
            parquet_files = sorted(episodes_dir.glob("**/*.parquet"))
            if parquet_files:
                dfs = [pd.read_parquet(f) for f in parquet_files]
                return pd.concat(dfs, ignore_index=True)

        raise FileNotFoundError(f"Episodes metadata not found in {self.meta_path}")

    def _load_tasks(self) -> pd.DataFrame:
        """Load task metadata."""
        tasks_path = self.meta_path / "tasks.parquet"
        if tasks_path.exists():
            return pd.read_parquet(tasks_path)
        return pd.DataFrame()

    def get_data_chunk(self, chunk_idx: int) -> pd.DataFrame:
        """Load data from a specific chunk."""
        chunk_dir = self.data_path / f"chunk-{chunk_idx:03d}"
        if not chunk_dir.exists():
            return pd.DataFrame()

        parquet_files = sorted(chunk_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    def get_video_path(self, camera_key: str, chunk_idx: int, file_idx: int = 0) -> Optional[Path]:
        """Get path to video file for a camera and chunk."""
        # Try observation.images.{camera_key} format
        video_dir = self.videos_path / f"observation.images.{camera_key}" / f"chunk-{chunk_idx:03d}"
        if not video_dir.exists():
            # Try just {camera_key}
            video_dir = self.videos_path / camera_key / f"chunk-{chunk_idx:03d}"

        if not video_dir.exists():
            return None

        # Find video file
        mp4_files = sorted(video_dir.glob("*.mp4"))
        if file_idx < len(mp4_files):
            return mp4_files[file_idx]
        return None

    def iterate_episodes(self):
        """Iterate over all episodes with their metadata."""
        for idx, row in self.episodes.iterrows():
            yield {
                "episode_index": row.get("episode_index", idx),
                "length": row.get("length", 0),
                "task_index": row.get("task_index", 0),
            }

    @property
    def fps(self) -> float:
        """Get dataset FPS."""
        return self.info.get("fps", 30)

    @property
    def total_episodes(self) -> int:
        """Get total number of episodes."""
        return len(self.episodes)

    @property
    def chunks_size(self) -> int:
        """Get chunk size."""
        return self.info.get("chunks_size", 1000)


class VideoFrameExtractor:
    """Extracts frames from MP4 video files using PyAV."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.container = av.open(str(self.video_path))
        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate) if self.stream.average_rate else 30.0

        # Build frame index for seeking
        self._frames: List[np.ndarray] = []
        self._loaded = False

    def load_all_frames(self):
        """Load all frames into memory."""
        if self._loaded:
            return

        self.container.seek(0)
        for frame in self.container.decode(video=0):
            self._frames.append(frame.to_ndarray(format='rgb24'))
        self._loaded = True

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Get a specific frame by index."""
        if not self._loaded:
            self.load_all_frames()

        if 0 <= frame_index < len(self._frames):
            return self._frames[frame_index]
        return None

    def get_frames_range(self, start: int, end: int) -> List[np.ndarray]:
        """Get a range of frames."""
        if not self._loaded:
            self.load_all_frames()

        return self._frames[start:end]

    @property
    def total_frames(self) -> int:
        """Get total number of frames."""
        if not self._loaded:
            self.load_all_frames()
        return len(self._frames)

    def close(self):
        """Close the video container."""
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to target size.

    Args:
        frame: RGB numpy array (H, W, 3)
        target_size: (height, width) tuple

    Returns:
        Resized RGB numpy array
    """
    img = Image.fromarray(frame)
    # PIL resize takes (width, height)
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(img)


def stack_frames(frames: List[np.ndarray], mode: str = "vertical") -> np.ndarray:
    """Stack multiple camera frames.

    Args:
        frames: List of RGB numpy arrays
        mode: "vertical", "horizontal", or "single"

    Returns:
        Stacked RGB numpy array
    """
    if mode == "single" or len(frames) == 1:
        return frames[0]
    elif mode == "vertical":
        return np.vstack(frames)
    elif mode == "horizontal":
        return np.hstack(frames)
    else:
        raise ValueError(f"Unknown stack mode: {mode}")


def save_frame(frame: np.ndarray, path: str, quality: int = 85):
    """Save frame as JPEG."""
    img = Image.fromarray(frame)
    img.save(path, "JPEG", quality=quality)


def discretize_action(
    current_pos: float,
    target_pos: float,
    dt: float,
    velocity_threshold: float
) -> int:
    """Convert continuous action to discrete action based on velocity.

    Args:
        current_pos: Current joint position (radians)
        target_pos: Target joint position from action (radians)
        dt: Time delta between frames (seconds)
        velocity_threshold: Minimum velocity to register as movement

    Returns:
        0 = no movement (|velocity| < threshold)
        1 = positive movement (velocity >= threshold)
        2 = negative movement (velocity <= -threshold)
    """
    if dt <= 0:
        return 0

    velocity = (target_pos - current_pos) / dt

    if abs(velocity) < velocity_threshold:
        return 0  # Stay
    elif velocity > 0:
        return 1  # Move positive
    else:
        return 2  # Move negative


class SessionWriter:
    """Writes concat_world_model_explorer session format."""

    def __init__(
        self,
        output_dir: str,
        session_name: str,
        shard_size: int = 1000
    ):
        self.session_dir = Path(output_dir) / session_name
        self.frames_dir = self.session_dir / "frames"
        self.shard_size = shard_size

        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)

        # State
        self.events: List[dict] = []
        self.step_count = 0
        self.current_shard = 0
        self.action_space: List[dict] = []
        self.robot_type = "unknown"
        self.extra_meta: dict = {}

    def set_metadata(
        self,
        robot_type: str,
        action_space: List[dict],
        extra_meta: Optional[dict] = None
    ):
        """Set session metadata."""
        self.action_space = action_space
        self.robot_type = robot_type
        self.extra_meta = extra_meta or {}

    def add_observation(self, frame: np.ndarray, timestamp: float):
        """Add an observation event."""
        frame_filename = f"frame_{self.step_count:06d}.jpg"
        frame_path = self.frames_dir / frame_filename
        save_frame(frame, str(frame_path))

        self.events.append({
            "step": self.step_count,
            "type": "observation",
            "data": {"frame_path": f"frames/{frame_filename}"},
            "timestamp": timestamp
        })

        self._increment_step()

    def add_action(self, action: dict, timestamp: float):
        """Add an action event."""
        self.events.append({
            "step": self.step_count,
            "type": "action",
            "data": action,
            "timestamp": timestamp
        })

        self._increment_step()

    def _increment_step(self):
        """Increment step counter and rotate shard if needed."""
        self.step_count += 1

        if len(self.events) >= self.shard_size:
            self._write_shard()

    def _write_shard(self):
        """Write current events to shard file."""
        if not self.events:
            return

        shard_path = self.session_dir / f"events_shard_{self.current_shard:03d}.jsonl"
        with open(shard_path, 'w') as f:
            for event in self.events:
                f.write(json.dumps(event) + '\n')

        print(f"  Wrote shard {self.current_shard} ({len(self.events)} events)")
        self.events = []
        self.current_shard += 1

    def finalize(self):
        """Finalize the session and write metadata."""
        # Write remaining events
        if self.events:
            self._write_shard()

        # Write session metadata
        meta = {
            "session_name": self.session_dir.name,
            "start_time": datetime.now().isoformat(),
            "shard_size": self.shard_size,
            "action_space": self.action_space,
            "robot_type": self.robot_type,
            "total_shards": self.current_shard,
            "total_steps": self.step_count,
            **self.extra_meta
        }

        meta_path = self.session_dir / "session_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  Session saved: {self.step_count} steps, {self.current_shard} shards")


def convert_episode(
    reader: LeRobotV3Reader,
    episode_info: dict,
    config: ConversionConfig,
    output_dir: str
) -> bool:
    """Convert a single episode to explorer format.

    Returns:
        True if conversion successful, False otherwise
    """
    episode_idx = episode_info["episode_index"]
    length = episode_info["length"]

    if length == 0:
        print(f"  Skipping episode {episode_idx}: no frames")
        return False

    # Determine chunk and offset
    chunk_idx = episode_idx // reader.chunks_size
    chunk_offset = episode_idx % reader.chunks_size

    # Load data for the chunk
    all_data = reader.get_data_chunk(chunk_idx)
    if all_data.empty:
        print(f"  Skipping episode {episode_idx}: no data in chunk {chunk_idx}")
        return False

    # Filter to this episode
    episode_data = all_data[all_data["episode_index"] == episode_idx]
    if len(episode_data) == 0:
        print(f"  Skipping episode {episode_idx}: episode not found in data")
        return False

    # Load video extractors for each camera
    extractors = {}
    for camera in config.cameras:
        video_path = reader.get_video_path(camera, chunk_idx)
        if video_path is None:
            print(f"  Warning: No video for camera {camera}, episode {episode_idx}")
            continue
        extractors[camera] = VideoFrameExtractor(str(video_path))
        extractors[camera].load_all_frames()

    if not extractors:
        print(f"  Skipping episode {episode_idx}: no video data")
        return False

    # Try to load discrete action log from meta/discrete_action_logs/
    # Filter to decision-bearing logs only (skips spurious header-only files
    # created by double-reset during multi-episode recording)
    dataset_path = Path(config.lerobot_path)
    log_dir = dataset_path / "meta" / "discrete_action_logs"
    action_log = None

    if log_dir.exists():
        real_logs = load_decision_bearing_logs(log_dir)
        if episode_idx < len(real_logs):
            action_log = real_logs[episode_idx]
            if action_log:
                print(f"  Using discrete action log: {len(action_log.decisions)} decisions")
                print(f"    Recording params: duration={action_log.action_duration}s, "
                      f"delta={action_log.position_delta}, joint={action_log.joint_name}")

    # Determine action_duration to use (from log or config)
    action_duration = action_log.action_duration if action_log else config.action_duration

    # Generate session name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{config.session_prefix}_ep{episode_idx:04d}_{timestamp}"

    # Create session writer
    writer = SessionWriter(
        output_dir=output_dir,
        session_name=session_name,
        shard_size=config.shard_size
    )

    # Define action space
    action_space = [
        {"action": 0, "duration": 0.0},
        {"action": 1, "duration": action_duration},
        {"action": 2, "duration": action_duration},
    ]

    writer.set_metadata(
        robot_type="so101_follower",
        action_space=action_space,
        extra_meta={
            "source": "lerobot_v3",
            "source_path": str(config.lerobot_path),
            "episode_index": episode_idx,
            "joint_name": action_log.joint_name if action_log else config.joint_name,
            "joint_index": config.joint_index,
            "action_duration": action_duration,
            "position_delta": action_log.position_delta if action_log else None,
            "used_discrete_action_log": action_log is not None,
            "cameras": config.cameras,
            "stack_mode": config.stack_cameras,
        }
    )

    # Compute time delta between frames
    dt = 1.0 / reader.fps

    # Find video offset for this episode
    # LeRobot v3 frame_index is 0-based per episode; use the global index column
    video_offset = 0
    if "index" in episode_data.columns:
        video_offset = episode_data["index"].iloc[0]
    elif "frame_index" in episode_data.columns:
        video_offset = episode_data["frame_index"].iloc[0]

    # Get total frames from first camera extractor
    first_camera = config.cameras[0]
    total_frames = extractors[first_camera].total_frames if first_camera in extractors else len(episode_data)

    def get_combined_frame(frame_idx: int) -> Optional[np.ndarray]:
        """Extract and combine frames from all cameras for a given frame index."""
        camera_frames = []
        for camera in config.cameras:
            if camera not in extractors:
                continue
            frame = extractors[camera].get_frame(frame_idx)
            if frame is not None:
                resized = resize_frame(frame, config.frame_size)
                camera_frames.append(resized)
        if not camera_frames:
            return None
        return stack_frames(camera_frames, config.stack_cameras)

    # Decision-boundary format: record observations only at action decision boundaries
    if action_log and action_log.decisions:
        # Map action decisions to frame indices using logged frame indices
        decision_frames = get_decision_frame_indices(
            action_log, total_frames,
        )

        # Trim trailing no-op actions (action=0) from the end
        # Keep the frame indices for result observations before trimming
        all_frame_indices = [f[0] for f in decision_frames]
        original_count = len(decision_frames)
        while decision_frames and decision_frames[-1][1] == 0:
            decision_frames.pop()

        if len(decision_frames) < original_count:
            print(f"  Trimmed {original_count - len(decision_frames)} trailing no-op actions")

        if not decision_frames:
            print(f"  Skipping episode {episode_idx}: all actions are no-ops")
            for extractor in extractors.values():
                extractor.close()
            return False

        print(f"  Converting {len(decision_frames)} decisions to decision-boundary format")

        # Verification diagnostic: show sample frame-action mappings
        action_names = {0: "stay", 1: "move+", 2: "move-"}
        sample_first = decision_frames[:5]
        sample_last = decision_frames[-3:] if len(decision_frames) > 5 else []
        first_str = ", ".join(f"f{f}->{action_names.get(a, '?')}" for f, a in sample_first)
        print(f"  Mapping (first 5): {first_str}")
        if sample_last:
            last_str = ", ".join(f"f{f}->{action_names.get(a, '?')}" for f, a in sample_last)
            print(f"  Mapping (last 3):  {last_str}")

        # Record initial observation (at first decision)
        first_frame_idx = decision_frames[0][0] + video_offset
        combined_frame = get_combined_frame(first_frame_idx)
        last_valid_frame = combined_frame
        if combined_frame is not None:
            writer.add_observation(combined_frame, 0.0)

        # For each action decision, record action then resulting observation
        for i, (frame_idx, discrete_action) in enumerate(decision_frames):
            # Record the action
            action_dict = {
                "action": discrete_action,
                "duration": action_duration if discrete_action != 0 else 0.0
            }
            writer.add_action(action_dict, (i + 0.5) * action_duration)

            # Record the resulting observation
            # Use the original frame index list (before trimming) to get proper result frames
            if i + 1 < len(all_frame_indices):
                next_frame_idx = all_frame_indices[i + 1] + video_offset
            else:
                # After last action, use the last frame of this episode
                next_frame_idx = min(video_offset + length - 1, total_frames - 1)

            combined_frame = get_combined_frame(next_frame_idx)
            if combined_frame is None:
                combined_frame = last_valid_frame
            if combined_frame is not None:
                writer.add_observation(combined_frame, (i + 1) * action_duration)
                last_valid_frame = combined_frame

    else:
        # Fallback: velocity-based discretization (frame-level format)
        print(f"  No discrete action log, using velocity-based discretization")
        for i, (_, row) in enumerate(episode_data.iterrows()):
            frame_idx = i + video_offset

            # Get timestamp
            timestamp_val = row.get("timestamp", i * dt)
            if hasattr(timestamp_val, 'item'):
                timestamp_val = timestamp_val.item()

            combined_frame = get_combined_frame(frame_idx)
            if combined_frame is None:
                continue

            # Add observation
            writer.add_observation(combined_frame, timestamp_val)

            # Get state and action for discretization
            state = row.get("observation.state")
            action_continuous = row.get("action")

            if state is not None and action_continuous is not None:
                # Convert to numpy arrays if needed
                if hasattr(state, 'tolist'):
                    state = np.array(state)
                elif isinstance(state, list):
                    state = np.array(state)

                if hasattr(action_continuous, 'tolist'):
                    action_continuous = np.array(action_continuous)
                elif isinstance(action_continuous, list):
                    action_continuous = np.array(action_continuous)

                # Get joint positions
                current_pos = state[config.joint_index]
                target_pos = action_continuous[config.joint_index]

                # Discretize action based on velocity
                discrete_action = discretize_action(
                    current_pos,
                    target_pos,
                    dt,
                    config.velocity_threshold
                )

                # Add action event
                action_dict = {
                    "action": discrete_action,
                    "duration": action_duration if discrete_action != 0 else 0.0
                }
                writer.add_action(action_dict, timestamp_val + dt * 0.5)

    # Finalize session
    writer.finalize()

    # Close extractors
    for extractor in extractors.values():
        extractor.close()

    return True


def convert_combined(reader: LeRobotV3Reader, config: ConversionConfig):
    """Convert all episodes into a single combined session.

    Episodes are concatenated sequentially with a stay action (action 0) inserted
    between episodes as a transition marker. This is designed for multi-height
    recordings where each episode is at a different shoulder_lift height.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{config.session_prefix}_combined_{timestamp_str}"

    writer = SessionWriter(
        output_dir=config.output_dir,
        session_name=session_name,
        shard_size=config.shard_size
    )

    # Load discrete action logs, filtering out spurious header-only files
    # created by double-reset during multi-episode recording
    dataset_path = Path(config.lerobot_path)
    log_dir = dataset_path / "meta" / "discrete_action_logs"
    episode_logs = []
    height_targets = []

    if log_dir.exists():
        episode_logs = load_decision_bearing_logs(log_dir)
        for log in episode_logs:
            if log.header.get("height_target") is not None:
                height_targets.append(log.header["height_target"])

    # Use first episode's log for base parameters
    first_log = next((l for l in episode_logs if l is not None), None)
    action_duration = first_log.action_duration if first_log else config.action_duration

    # Define action space
    action_space = [
        {"action": 0, "duration": 0.0},
        {"action": 1, "duration": action_duration},
        {"action": 2, "duration": action_duration},
    ]

    writer.set_metadata(
        robot_type="so101_follower",
        action_space=action_space,
        extra_meta={
            "source": "lerobot_v3",
            "source_path": str(config.lerobot_path),
            "joint_name": first_log.joint_name if first_log else config.joint_name,
            "joint_index": config.joint_index,
            "action_duration": action_duration,
            "position_delta": first_log.position_delta if first_log else None,
            "used_discrete_action_log": first_log is not None,
            "cameras": config.cameras,
            "stack_mode": config.stack_cameras,
            "combined_episodes": True,
            "num_episodes": reader.total_episodes,
            "height_targets": height_targets if height_targets else None,
        }
    )

    running_timestamp = 0.0
    episodes_converted = 0

    for ep_idx, episode_info in enumerate(
        tqdm(list(reader.iterate_episodes()), desc="Combining episodes")
    ):
        episode_idx = episode_info["episode_index"]
        length = episode_info["length"]

        if length == 0:
            print(f"  Skipping episode {episode_idx}: no frames")
            continue

        # Determine chunk and offset
        chunk_idx = episode_idx // reader.chunks_size

        # Load data for the chunk
        all_data = reader.get_data_chunk(chunk_idx)
        if all_data.empty:
            continue

        episode_data = all_data[all_data["episode_index"] == episode_idx]
        if len(episode_data) == 0:
            continue

        # Load video extractors
        extractors = {}
        for camera in config.cameras:
            video_path = reader.get_video_path(camera, chunk_idx)
            if video_path is None:
                continue
            extractors[camera] = VideoFrameExtractor(str(video_path))
            extractors[camera].load_all_frames()

        if not extractors:
            continue

        # Get action log for this episode (ep_idx is the sequential episode counter)
        action_log = episode_logs[ep_idx] if ep_idx < len(episode_logs) else None
        ep_action_duration = action_log.action_duration if action_log else action_duration

        # Find video offset
        # LeRobot v3 frame_index is 0-based per episode; use the global index column
        video_offset = 0
        if "index" in episode_data.columns:
            video_offset = episode_data["index"].iloc[0]
        elif "frame_index" in episode_data.columns:
            video_offset = episode_data["frame_index"].iloc[0]

        first_camera = config.cameras[0]
        total_frames = (extractors[first_camera].total_frames
                        if first_camera in extractors else len(episode_data))

        def get_combined_frame(frame_idx: int) -> Optional[np.ndarray]:
            camera_frames = []
            for camera in config.cameras:
                if camera not in extractors:
                    continue
                frame = extractors[camera].get_frame(frame_idx)
                if frame is not None:
                    resized = resize_frame(frame, config.frame_size)
                    camera_frames.append(resized)
            if not camera_frames:
                return None
            return stack_frames(camera_frames, config.stack_cameras)

        # Insert transition stay action between episodes (not before first)
        if episodes_converted > 0:
            writer.add_action(
                {"action": 0, "duration": 0.0},
                running_timestamp
            )
            running_timestamp += ep_action_duration

        # Decision-boundary format (with discrete action log)
        if action_log and action_log.decisions:
            decision_frames = get_decision_frame_indices(action_log, total_frames)

            all_frame_indices = [f[0] for f in decision_frames]

            # Trim trailing no-op actions
            while decision_frames and decision_frames[-1][1] == 0:
                decision_frames.pop()

            if not decision_frames:
                for ext in extractors.values():
                    ext.close()
                continue

            # Record initial observation
            first_frame_idx = decision_frames[0][0] + video_offset
            combined_frame = get_combined_frame(first_frame_idx)
            last_valid_frame = combined_frame
            if combined_frame is not None:
                writer.add_observation(combined_frame, running_timestamp)

            # Record actions and resulting observations
            for i, (frame_idx, discrete_action) in enumerate(decision_frames):
                action_dict = {
                    "action": discrete_action,
                    "duration": ep_action_duration if discrete_action != 0 else 0.0
                }
                running_timestamp += ep_action_duration * 0.5
                writer.add_action(action_dict, running_timestamp)
                running_timestamp += ep_action_duration * 0.5

                if i + 1 < len(all_frame_indices):
                    next_frame_idx = all_frame_indices[i + 1] + video_offset
                else:
                    # After last action, use the last frame of this episode
                    next_frame_idx = min(video_offset + length - 1, total_frames - 1)

                combined_frame = get_combined_frame(next_frame_idx)
                if combined_frame is None:
                    combined_frame = last_valid_frame
                if combined_frame is not None:
                    writer.add_observation(combined_frame, running_timestamp)
                    last_valid_frame = combined_frame

        else:
            # Velocity-based fallback
            dt = 1.0 / reader.fps
            for i, (_, row) in enumerate(episode_data.iterrows()):
                frame_idx = i + video_offset
                timestamp_val = row.get("timestamp", i * dt)
                if hasattr(timestamp_val, 'item'):
                    timestamp_val = timestamp_val.item()

                combined_frame = get_combined_frame(frame_idx)
                if combined_frame is None:
                    continue

                writer.add_observation(combined_frame, running_timestamp)

                state = row.get("observation.state")
                action_continuous = row.get("action")

                if state is not None and action_continuous is not None:
                    if hasattr(state, 'tolist'):
                        state = np.array(state)
                    elif isinstance(state, list):
                        state = np.array(state)

                    if hasattr(action_continuous, 'tolist'):
                        action_continuous = np.array(action_continuous)
                    elif isinstance(action_continuous, list):
                        action_continuous = np.array(action_continuous)

                    current_pos = state[config.joint_index]
                    target_pos = action_continuous[config.joint_index]

                    discrete_action = discretize_action(
                        current_pos, target_pos, dt, config.velocity_threshold
                    )

                    action_dict = {
                        "action": discrete_action,
                        "duration": ep_action_duration if discrete_action != 0 else 0.0
                    }
                    running_timestamp += dt
                    writer.add_action(action_dict, running_timestamp)

        # Close extractors for this episode
        for ext in extractors.values():
            ext.close()

        episodes_converted += 1

    writer.finalize()
    print(f"\nCombined {episodes_converted} episodes into: {writer.session_dir}")


def convert_dataset(config: ConversionConfig):
    """Convert LeRobot dataset to explorer format."""
    print(f"Loading LeRobot dataset from {config.lerobot_path}")
    reader = LeRobotV3Reader(config.lerobot_path)

    print(f"Found {reader.total_episodes} episodes at {reader.fps} FPS")
    print(f"Converting with cameras: {config.cameras}")
    print(f"Stack mode: {config.stack_cameras}")
    print(f"Joint: {config.joint_name} (index {config.joint_index})")

    if config.combine_episodes:
        print(f"Mode: combining all episodes into single session")
        convert_combined(reader, config)
        return

    # Convert each episode separately
    successful = 0
    for episode_info in tqdm(list(reader.iterate_episodes()), desc="Converting episodes"):
        if convert_episode(reader, episode_info, config, config.output_dir):
            successful += 1

    print(f"\nConversion complete: {successful}/{reader.total_episodes} episodes converted")
    print(f"Output directory: {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3.0 dataset to concat_world_model_explorer format"
    )
    parser.add_argument(
        "--lerobot-path",
        type=str,
        required=True,
        help="Path to LeRobot v3.0 dataset OR HuggingFace repo_id (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved/sessions/so101",
        help="Output directory for sessions"
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["base_0_rgb", "left_wrist_0_rgb"],
        help="Camera keys to use (e.g., base_0_rgb left_wrist_0_rgb)"
    )
    parser.add_argument(
        "--stack-cameras",
        type=str,
        choices=["vertical", "horizontal", "single"],
        default="vertical",
        help="How to combine multiple cameras"
    )
    parser.add_argument(
        "--joint-name",
        type=str,
        default="shoulder_pan.pos",
        help="Joint name for action discretization"
    )
    parser.add_argument(
        "--action-duration",
        type=float,
        default=0.5,
        help="Duration of move actions in seconds (used if no discrete action log available)"
    )
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=0.05,
        help="Minimum velocity (rad/s) to register as movement"
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Output frame size per camera (height width)"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Events per shard file"
    )
    parser.add_argument(
        "--session-prefix",
        type=str,
        default="session_so101",
        help="Prefix for session names"
    )
    parser.add_argument(
        "--combine-episodes",
        action="store_true",
        default=False,
        help="Combine all episodes into a single session (for multi-height recordings)"
    )

    args = parser.parse_args()

    # Resolve dataset path (download from Hub if needed)
    dataset_path = resolve_dataset_path(args.lerobot_path)

    # Validate joint name
    if args.joint_name not in SO101_JOINTS:
        print(f"Error: Unknown joint '{args.joint_name}'")
        print(f"Valid joints: {SO101_JOINTS}")
        sys.exit(1)

    joint_index = SO101_JOINTS.index(args.joint_name)

    config = ConversionConfig(
        lerobot_path=str(dataset_path),
        output_dir=args.output_dir,
        cameras=args.cameras,
        stack_cameras=args.stack_cameras,
        joint_name=args.joint_name,
        joint_index=joint_index,
        action_duration=args.action_duration,
        velocity_threshold=args.velocity_threshold,
        frame_size=tuple(args.frame_size),
        shard_size=args.shard_size,
        session_prefix=args.session_prefix,
        combine_episodes=args.combine_episodes,
    )

    convert_dataset(config)


if __name__ == "__main__":
    main()
