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


def _estimate_decision_frame_range(
    log: DiscreteActionLog,
    episode_data: pd.DataFrame,
    joint_index: int,
    total_frames: int,
    fps: float,
) -> Tuple[int, int]:
    """Estimate the video frame range spanned by action log decisions.

    Uses the Parquet data's per-frame action targets and joint states to find
    the exact video frame where the first MOVE action starts, then back-
    calculates where the first decision (which may be a "stay") occurred.

    This anchors the decision-to-frame mapping to ground truth, correcting
    for any offset between the start of video recording and the first policy
    call.

    Args:
        log: The discrete action log
        episode_data: Parquet DataFrame with 'observation.state' and 'action'
        joint_index: Which joint to inspect (0 = shoulder_pan)
        total_frames: Total video frames
        fps: Video frames per second

    Returns:
        (first_decision_frame, last_decision_frame) tuple
    """
    try:
        # Extract action targets and current states for the controlled joint
        states = np.array([row[joint_index] for row in episode_data["observation.state"]])
        targets = np.array([row[joint_index] for row in episode_data["action"]])
        deltas = targets - states

        threshold = log.position_delta / 2.0

        # Find first non-zero action in the log
        first_nonzero_log_idx = None
        for i, d in enumerate(log.decisions):
            if d["discrete_action"] != 0:
                first_nonzero_log_idx = i
                break

        # Find first frame where motor target diverges from current position
        first_move_frame = None
        for i in range(len(deltas)):
            if abs(deltas[i]) > threshold:
                first_move_frame = i
                break

        if first_nonzero_log_idx is not None and first_move_frame is not None:
            # Time from first decision to first non-zero decision
            t_to_nonzero = (log.decisions[first_nonzero_log_idx]["timestamp"]
                            - log.decisions[0]["timestamp"])
            # Scale by time_scale (video_duration / log_duration)
            log_span = log.decisions[-1]["timestamp"] - log.decisions[0]["timestamp"]
            video_duration = total_frames / fps
            time_scale = video_duration / log_span if log_span > 0 else 1.0
            scaled_offset_frames = t_to_nonzero * time_scale * fps
            first_frame = max(0, round(first_move_frame - scaled_offset_frames))

            print(f"  Anchor: first_move_frame={first_move_frame}, "
                  f"log_nonzero_idx={first_nonzero_log_idx}, "
                  f"t_to_nonzero={t_to_nonzero:.2f}s, "
                  f"first_decision_frame={first_frame}")

            return first_frame, total_frames - 1

    except Exception as e:
        print(f"  Warning: anchor estimation failed ({e}), using frame 0")

    return 0, total_frames - 1


def get_decision_frame_indices(
    log: DiscreteActionLog,
    total_frames: int,
    episode_data: pd.DataFrame = None,
    joint_index: int = 0,
    fps: float = 10.0,
) -> List[Tuple[int, int]]:
    """Map each action decision to a frame index using Parquet-anchored mapping.

    When Parquet data is available, uses the per-frame action/state columns to
    find the exact video frame of the first MOVE action, then anchors the
    proportional mapping to that ground-truth point.  This corrects for any
    offset between video start and first policy call.

    Falls back to pure proportional mapping (first decision → frame 0) when
    Parquet data is unavailable, and to even distribution when timestamps are
    missing entirely.

    Args:
        log: The discrete action log with decisions
        total_frames: Total number of frames in the episode
        episode_data: Optional Parquet DataFrame for anchor estimation
        joint_index: Joint index for anchor detection (default 0 = shoulder_pan)
        fps: Video FPS for anchor offset calculation

    Returns:
        List of (frame_index, discrete_action) tuples
    """
    if not log.decisions:
        return []

    # Check if decisions have timestamps
    has_timestamps = all("timestamp" in d for d in log.decisions)

    if has_timestamps and len(log.decisions) >= 2:
        first_decision_time = log.decisions[0]["timestamp"]
        last_decision_time = log.decisions[-1]["timestamp"]
        log_span = last_decision_time - first_decision_time

        # Estimate frame range using Parquet anchor if available
        first_frame = 0
        last_frame = total_frames - 1

        if episode_data is not None and log_span > 0:
            first_frame, last_frame = _estimate_decision_frame_range(
                log, episode_data, joint_index, total_frames, fps
            )

        # Map all decisions proportionally within [first_frame, last_frame]
        frame_range = last_frame - first_frame
        decision_frames = []
        for decision in log.decisions:
            elapsed = decision["timestamp"] - first_decision_time
            frac = elapsed / log_span if log_span > 0 else 0.0
            frame_idx = first_frame + round(frac * frame_range)
            frame_idx = max(0, min(total_frames - 1, frame_idx))
            decision_frames.append((frame_idx, decision["discrete_action"]))

        return decision_frames

    # Fallback: even distribution (no timestamps available)
    num_decisions = len(log.decisions)
    if total_frames < num_decisions + 1:
        num_decisions = total_frames - 1

    decision_frames = []
    for i, decision in enumerate(log.decisions[:num_decisions]):
        frame_idx = int(i * (total_frames - 1) / num_decisions)
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
    dataset_path = Path(config.lerobot_path)
    log_dir = dataset_path / "meta" / "discrete_action_logs"
    action_log = None

    if log_dir.exists():
        log_files = sorted(log_dir.glob("episode_*.jsonl"))
        if episode_idx < len(log_files):
            action_log = load_discrete_action_log(log_files[episode_idx])
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
    # LeRobot stores video offset in the data
    video_offset = 0
    if "frame_index" in episode_data.columns:
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
        # Map action decisions to frame indices using Parquet-anchored mapping
        decision_frames = get_decision_frame_indices(
            action_log, total_frames,
            episode_data=episode_data,
            joint_index=config.joint_index,
            fps=reader.fps,
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
                # After last action, use the last available frame
                next_frame_idx = total_frames - 1 + video_offset

            combined_frame = get_combined_frame(next_frame_idx)
            if combined_frame is not None:
                writer.add_observation(combined_frame, (i + 1) * action_duration)

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


def convert_dataset(config: ConversionConfig):
    """Convert LeRobot dataset to explorer format."""
    print(f"Loading LeRobot dataset from {config.lerobot_path}")
    reader = LeRobotV3Reader(config.lerobot_path)

    print(f"Found {reader.total_episodes} episodes at {reader.fps} FPS")
    print(f"Converting with cameras: {config.cameras}")
    print(f"Stack mode: {config.stack_cameras}")
    print(f"Joint: {config.joint_name} (index {config.joint_index})")

    # Convert each episode
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
    )

    convert_dataset(config)


if __name__ == "__main__":
    main()
