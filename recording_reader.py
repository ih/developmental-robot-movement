import json
import os
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import config

class RecordingReader:
    """Reads recorded session data with separate frame and action sequences."""

    def __init__(self, session_dir: str):
        """Initialize reader for a recorded session.

        Expected structure:
        session_dir/
        ├── session_meta.json    # {action_space: [...], start_time: ..., robot_type: ...}
        ├── events_shard_000.jsonl # [{"step": 0, "type": "observation", "data": {...}}, {"step": 1, "type": "action", "data": {...}}, ...]
        ├── events_shard_001.jsonl # (if multiple shards)
        └── frames/              # frame_000000.jpg, frame_000002.jpg, ... (only observation steps)

        The sequencing is:
        - step_0: observation → step_1: action → step_2: observation → step_3: action → ...
        - get_observation() returns frame at current step (must be observation)
        - action_selector() gets action at current step + 1 (must be action)
        - execute_action() advances to next step pair
        """
        self.session_dir = session_dir
        self.session_meta_path = os.path.join(session_dir, "session_meta.json")

        # Load session metadata
        self.session_meta = {}
        if os.path.exists(self.session_meta_path):
            with open(self.session_meta_path, 'r') as f:
                self.session_meta = json.load(f)
        else:
            raise FileNotFoundError(f"Session metadata not found: {self.session_meta_path}")

        # Load all event shards
        self.events = []
        self._load_event_shards()

        # Sort by step to ensure proper sequencing
        self.events.sort(key=lambda x: x['step'])

        self.current_step = 0  # Current step in the sequence
        self.total_steps = len(self.events)

        print(f"RecordingReader: Loaded {len(self.events)} events from {session_dir}")
        print(f"RecordingReader: Will replay {self.total_steps} steps")

    def _load_event_shards(self):
        """Load all event shard files from the session directory."""
        # Find all event shard files
        import glob
        shard_pattern = os.path.join(self.session_dir, "events_shard_*.jsonl")
        shard_files = sorted(glob.glob(shard_pattern))

        if not shard_files:
            raise FileNotFoundError(f"No event shard files found in {self.session_dir}")

        # Load all events from all shards
        for shard_file in shard_files:
            with open(shard_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.events.append(json.loads(line))

    def _load_frame(self, frame_path: str) -> np.ndarray:
        """Load frame from disk in same format as live robot."""
        full_path = os.path.join(self.session_dir, frame_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Frame file not found: {full_path}")

        # Load image and convert to numpy array in (H, W, 3) format like live robot
        img = Image.open(full_path).convert('RGB')
        return np.array(img)

    def get_next_observation(self) -> np.ndarray:
        """Get current or next observation, advancing only if safe."""
        # Find the next observation from current position
        for i in range(self.current_step, self.total_steps):
            event = self.events[i]
            if event['type'] == 'observation':
                # Check if next event is an action - if so, don't advance yet
                # This prevents skipping observations when autoencoder training repeats
                if i + 1 < self.total_steps and self.events[i + 1]['type'] == 'action':
                    # Next event is action - return this observation but don't advance
                    frame_path = event['data']['frame_path']
                    return self._load_frame(frame_path)
                else:
                    # Safe to advance past this observation
                    self.current_step = i + 1
                    frame_path = event['data']['frame_path']
                    return self._load_frame(frame_path)

        raise StopIteration("No more observations found in recording")

    def get_next_action(self) -> Dict[str, Any]:
        """Get next action and advance past it."""
        # Search forward from current step to find next action
        for i in range(self.current_step, self.total_steps):
            event = self.events[i]
            if event['type'] == 'action':
                # Advance past this action
                self.current_step = i + 1
                return event['data']

        raise StopIteration("No more actions found in recording")

    def advance_step(self) -> None:
        """Advance to next step (called after execute_action)."""
        self.current_step += 1

    def reset(self) -> None:
        """Reset to beginning of recording."""
        self.current_step = 0

    def seek(self, step_idx: int) -> None:
        """Seek to specific step index."""
        if step_idx < 0 or step_idx >= self.total_steps:
            raise IndexError(f"Step index {step_idx} out of range [0, {self.total_steps})")
        self.current_step = step_idx

    def is_finished(self) -> bool:
        """Check if we've reached the end of the recording."""
        return self.current_step >= self.total_steps

    def progress(self) -> float:
        """Return progress as fraction [0.0, 1.0]."""
        if self.total_steps == 0:
            return 1.0
        return self.current_step / self.total_steps

    def get_current_metadata(self) -> Dict[str, Any]:
        """Get metadata for current step (timestamps, etc.)."""
        if self.current_step >= self.total_steps:
            return {}

        event = self.events[self.current_step]

        return {
            'step': self.current_step,
            'event_step': event['step'],
            'event_type': event['type'],
            'timestamp': event.get('timestamp')
        }

    def get_action_space(self) -> List[Dict[str, Any]]:
        """Get the action space from session metadata."""
        return self.session_meta.get('action_space', [])

    def get_session_info(self) -> Dict[str, Any]:
        """Get full session metadata."""
        return self.session_meta.copy()