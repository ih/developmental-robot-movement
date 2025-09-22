import json
import os
import time
import glob
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import config

class RecordingWriter:
    """Writes recording data with shard rotation to manage disk space."""

    def __init__(self, base_dir: str = None, session_name: str = None,
                 shard_size: int = None, max_disk_gb: float = None):
        """Initialize recording writer.

        Args:
            base_dir: Base directory for recordings (defaults to config.RECORDING_BASE_DIR)
            session_name: Session name (auto-generated if None)
            shard_size: Steps per shard (defaults to config.RECORDING_SHARD_SIZE)
            max_disk_gb: Maximum disk space in GB (defaults to config.RECORDING_MAX_DISK_GB)
        """
        self.base_dir = base_dir or config.RECORDING_BASE_DIR
        self.shard_size = shard_size or config.RECORDING_SHARD_SIZE
        self.max_disk_gb = max_disk_gb or config.RECORDING_MAX_DISK_GB

        # Generate session name if not provided
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{timestamp}"

        self.session_name = session_name
        self.session_dir = os.path.join(self.base_dir, session_name)

        # Create session directory
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "frames"), exist_ok=True)

        # Initialize counters and state
        self.step_count = 0
        self.current_shard = 0
        self.shard_step_count = 0

        # Buffer for current shard (single interleaved sequence)
        self.event_buffer = []

        # Session metadata
        self.session_meta = {
            'session_name': session_name,
            'start_time': datetime.now().isoformat(),
            'shard_size': self.shard_size,
            'max_disk_gb': self.max_disk_gb,
            'action_space': None,  # Will be set when first initialized
            'robot_type': None,    # Will be set when first initialized
            'total_shards': 0
        }

        print(f"RecordingWriter: Initialized session '{session_name}'")
        print(f"RecordingWriter: Shard size={self.shard_size}, Max disk space={self.max_disk_gb} GB")

    def set_session_metadata(self, action_space: List[Dict[str, Any]], robot_type: str = "unknown"):
        """Set session metadata (action space, robot type, etc.)."""
        self.session_meta['action_space'] = action_space
        self.session_meta['robot_type'] = robot_type
        self._write_session_metadata()

    def record_observation(self, frame: np.ndarray, timestamp: float = None):
        """Record an observation.

        Args:
            frame: Current observation frame (numpy array)
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        # Save frame to disk
        frame_filename = f"frame_{self.step_count:06d}.jpg"
        frame_path = os.path.join(self.session_dir, "frames", frame_filename)
        self._save_frame(frame, frame_path)

        # Add observation event to buffer
        self.event_buffer.append({
            'step': self.step_count,
            'type': 'observation',
            'data': {
                'frame_path': f"frames/{frame_filename}"
            },
            'timestamp': timestamp
        })

        self._increment_step()

    def record_action(self, action: Dict[str, Any], timestamp: float = None):
        """Record an action.

        Args:
            action: Action executed
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        # Add action event to buffer
        self.event_buffer.append({
            'step': self.step_count,
            'type': 'action',
            'data': action,
            'timestamp': timestamp
        })

        self._increment_step()

    def _increment_step(self):
        """Increment step counters and check for shard rotation."""
        self.step_count += 1
        self.shard_step_count += 1

        # Check if we need to rotate shard
        if self.shard_step_count >= self.shard_size:
            self._rotate_shard()

    def _save_frame(self, frame: np.ndarray, path: str):
        """Save frame to disk as JPG."""
        # Handle different frame formats
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            # Assume normalized [-1, 1] or [0, 1], convert to [0, 255]
            if frame.min() >= 0 and frame.max() <= 1:
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:  # Assume [-1, 1]
                frame_uint8 = ((frame + 1) * 127.5).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)

        # Handle channel order (CHW -> HWC if needed)
        if len(frame_uint8.shape) == 3 and frame_uint8.shape[0] == 3:
            frame_uint8 = np.transpose(frame_uint8, (1, 2, 0))

        # Convert to PIL and save
        img = Image.fromarray(frame_uint8)
        img.save(path, "JPEG", quality=85)

    def _rotate_shard(self):
        """Rotate to next shard, saving current buffers and cleaning up old shards."""
        print(f"RecordingWriter: Rotating to shard {self.current_shard + 1} after {self.shard_step_count} steps")

        # Write current shard
        self._write_shard_files()

        # Move to next shard
        self.current_shard += 1
        self.shard_step_count = 0

        # Clear buffer
        self.event_buffer = []

        # Clean up old sessions if we exceed disk space limit
        self._cleanup_old_sessions()

        # Update total_shards to reflect actual shards on disk
        self._update_total_shards_count()

        # Update session metadata
        self._write_session_metadata()

    def _write_shard_files(self):
        """Write current buffer to shard file."""
        shard_suffix = f"_shard_{self.current_shard:03d}"

        # Write single interleaved event sequence for this shard
        events_file = os.path.join(self.session_dir, f"events{shard_suffix}.jsonl")
        with open(events_file, 'w') as f:
            for event in self.event_buffer:
                f.write(json.dumps(event) + '\n')

        print(f"RecordingWriter: Wrote shard {self.current_shard} ({len(self.event_buffer)} events)")

    def _cleanup_old_sessions(self):
        """Remove old session directories if total recording size exceeds disk limit."""
        total_size_gb = self._get_total_recordings_size_gb()

        if total_size_gb <= self.max_disk_gb:
            return

        print(f"RecordingWriter: Total recordings size {total_size_gb:.2f} GB exceeds limit {self.max_disk_gb} GB")

        # Get all session directories sorted by modification time (oldest first)
        session_dirs = []
        if os.path.exists(self.base_dir):
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path) and item.startswith('session_'):
                    # Skip current session
                    if item_path != self.session_dir:
                        mtime = os.path.getmtime(item_path)
                        session_dirs.append((mtime, item_path, item))

        # Sort by modification time (oldest first)
        session_dirs.sort(key=lambda x: x[0])

        # Remove oldest sessions until we're under the limit
        removed_count = 0
        for mtime, session_path, session_name in session_dirs:
            if total_size_gb <= self.max_disk_gb:
                break

            session_size_gb = self._get_directory_size_gb(session_path)
            print(f"RecordingWriter: Removing old session '{session_name}' ({session_size_gb:.2f} GB)")

            try:
                self._remove_directory_recursive(session_path)
                total_size_gb -= session_size_gb
                removed_count += 1
            except Exception as e:
                print(f"RecordingWriter: Warning - failed to remove session {session_name}: {e}")

        if removed_count > 0:
            print(f"RecordingWriter: Cleaned up {removed_count} old sessions. New total: {total_size_gb:.2f} GB")

    def _get_total_recordings_size_gb(self) -> float:
        """Get total size of all recordings in GB."""
        if not os.path.exists(self.base_dir):
            return 0.0
        return self._get_directory_size_gb(self.base_dir)

    def _get_directory_size_gb(self, directory: str) -> float:
        """Get size of directory in GB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        except Exception as e:
            print(f"RecordingWriter: Warning - error calculating directory size: {e}")
        return total_size / (1024 ** 3)  # Convert to GB

    def _remove_directory_recursive(self, directory: str):
        """Recursively remove a directory and all its contents."""
        import shutil
        shutil.rmtree(directory)

    def _update_total_shards_count(self):
        """Update total_shards count to reflect actual shards on disk."""
        shard_pattern = os.path.join(self.session_dir, "events_shard_*.jsonl")
        actual_shard_files = glob.glob(shard_pattern)
        self.session_meta['total_shards'] = len(actual_shard_files)

    def _write_session_metadata(self):
        """Write session metadata to disk."""
        meta_path = os.path.join(self.session_dir, "session_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(self.session_meta, f, indent=2)

    def finalize(self):
        """Finalize recording - write remaining buffer and update metadata."""
        if self.event_buffer:
            print(f"RecordingWriter: Finalizing with {len(self.event_buffer)} remaining events")
            self._write_shard_files()

        # Update total_shards to reflect actual shards on disk
        self._update_total_shards_count()

        # Update final metadata
        self.session_meta['end_time'] = datetime.now().isoformat()
        self.session_meta['total_steps'] = self.step_count
        self._write_session_metadata()

        print(f"RecordingWriter: Finalized session '{self.session_name}' with {self.step_count} total steps")

    def get_session_path(self) -> str:
        """Get the full path to the session directory."""
        return self.session_dir