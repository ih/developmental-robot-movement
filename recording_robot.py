from typing import Dict, List, Any, Optional
import numpy as np
import time
from robot_interface import RobotInterface
from recording_writer import RecordingWriter

class RecordingRobot(RobotInterface):
    """Wrapper robot that records all observations and actions while passing through to underlying robot."""

    def __init__(self, robot: RobotInterface, recording_writer: RecordingWriter):
        """Initialize recording robot wrapper.

        Args:
            robot: Underlying robot interface to wrap
            recording_writer: RecordingWriter instance to save data to
        """
        self.robot = robot
        self.writer = recording_writer

        # Initialize session metadata
        self.writer.set_session_metadata(
            action_space=robot.action_space,
            robot_type=type(robot).__name__
        )

        print(f"RecordingRobot: Wrapping {type(robot).__name__} with recording")

    def get_observation(self) -> Optional[np.ndarray]:
        """Get observation from underlying robot and record it."""
        observation = self.robot.get_observation()

        if observation is not None:
            # Record the observation
            self.writer.record_observation(observation, time.time())

        return observation

    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Record action and execute it on underlying robot."""
        # Record the action
        self.writer.record_action(action, time.time())

        # Execute the action on the underlying robot
        return self.robot.execute_action(action)

    @property
    def action_space(self) -> List[Dict[str, Any]]:
        """Return action space from underlying robot."""
        return self.robot.action_space

    def cleanup(self) -> None:
        """Clean up recording and underlying robot."""
        print("RecordingRobot: Finalizing recording...")
        self.writer.finalize()
        self.robot.cleanup()

    def get_recording_path(self) -> str:
        """Get path to the recording session directory."""
        return self.writer.get_session_path()

    def get_recording_stats(self) -> Dict[str, Any]:
        """Get current recording statistics."""
        return {
            'session_name': self.writer.session_name,
            'total_steps': self.writer.step_count,
            'current_shard': self.writer.current_shard,
            'shard_step_count': self.writer.shard_step_count,
            'session_path': self.writer.get_session_path()
        }