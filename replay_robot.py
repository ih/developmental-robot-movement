from typing import Dict, List, Any, Optional
import numpy as np
from robot_interface import RobotInterface
from recording_reader import RecordingReader

class ReplayRobot(RobotInterface):
    """Drop-in replacement for live robot that replays recorded actions and observations."""

    def __init__(self, reader: RecordingReader, action_space: List[Dict[str, Any]]):
        """Initialize replay robot.

        Args:
            reader: RecordingReader instance with loaded session data
            action_space: List of possible actions (from session metadata)
        """
        self.reader = reader
        self._action_space = action_space

        print(f"ReplayRobot: Initialized for {self.reader.total_steps} steps")
        print(f"ReplayRobot: Action space has {len(self._action_space)} actions")

    def get_observation(self) -> Optional[np.ndarray]:
        """Return next observation from recording."""
        # Let StopIteration propagate so main loop can exit properly
        return self.reader.get_next_observation()

    def execute_action(self, action: Dict[str, Any]) -> bool:
        """No-op for replay - action already consumed by action selector."""
        # In replay mode, actions are already consumed by the recorded policy
        # No need to advance or validate since everything is pre-recorded
        return True


    @property
    def action_space(self) -> List[Dict[str, Any]]:
        """Return possible actions from session metadata."""
        return self._action_space

    def cleanup(self) -> None:
        """Clean up resources - no-op for replay robot."""
        print("ReplayRobot: Cleanup complete")

    def get_status(self) -> Dict[str, Any]:
        """Get current replay status."""
        return {
            'current_step': self.reader.current_step,
            'total_steps': self.reader.total_steps,
            'progress': self.reader.progress(),
            'is_finished': self.reader.is_finished(),
            'metadata': self.reader.get_current_metadata()
        }