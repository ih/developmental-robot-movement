import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import config

class RecordingWriter:
    """Writes recording data with shard rotation to manage disk space."""

    def __init__(self, base_dir: str = None, session_name: str = None,
                 shard_size: int = None, max_shards: int = None):
        """Initialize recording writer.

        Args:
            base_dir: Base directory for recordings (defaults to config.RECORDING_BASE_DIR)
            session_name: Session name (auto-generated if None)
            shard_size: Steps per shard (defaults to config.RECORDING_SHARD_SIZE)
            max_shards: Maximum shards to keep (defaults to config.RECORDING_MAX_SHARDS)
        """
        self.base_dir = base_dir or config.RECORDING_BASE_DIR
        self.shard_size = shard_size or config.RECORDING_SHARD_SIZE
        self.max_shards = max_shards or config.RECORDING_MAX_SHARDS

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
            'max_shards': self.max_shards,
            'action_space': None,  # Will be set when first initialized
            'robot_type': None,    # Will be set when first initialized
            'total_shards': 0
        }

        print(f"RecordingWriter: Initialized session '{session_name}'")
        print(f"RecordingWriter: Shard size={self.shard_size}, Max shards={self.max_shards}")

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

        # Clean up old shards if we exceed max_shards
        self._cleanup_old_shards()

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

    def _cleanup_old_shards(self):
        """Remove old shard files if we exceed max_shards."""
        if self.current_shard + 1 <= self.max_shards:
            return

        # Calculate which shards to remove
        oldest_shard_to_keep = self.current_shard + 1 - self.max_shards
        shards_to_remove = list(range(0, oldest_shard_to_keep))

        for shard_idx in shards_to_remove:
            shard_suffix = f"_shard_{shard_idx:03d}"

            # Remove shard file
            events_file = os.path.join(self.session_dir, f"events{shard_suffix}.jsonl")
            if os.path.exists(events_file):
                os.remove(events_file)

            # Remove associated frame files
            frame_start = shard_idx * self.shard_size
            frame_end = min((shard_idx + 1) * self.shard_size, self.step_count)

            for frame_idx in range(frame_start, frame_end):
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(self.session_dir, "frames", frame_filename)
                if os.path.exists(frame_path):
                    os.remove(frame_path)

        print(f"RecordingWriter: Cleaned up {len(shards_to_remove)} old shards")

    def _update_total_shards_count(self):
        """Update total_shards count to reflect actual shards on disk."""
        import glob
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