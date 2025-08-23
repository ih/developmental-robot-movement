from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

class RobotInterface(ABC):
    """An abstract interface for a robot, defining the methods 
    the AdaptiveWorldModel needs to interact with it."""

    @abstractmethod
    def get_observation(self) -> Optional[np.ndarray]:
        """Capture the current sensory input from the robot (e.g., a camera frame).
        
        Returns:
            np.ndarray: Current observation (e.g., camera frame) or None if capture fails
        """
        pass

    @abstractmethod
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Send a command to the robot's actuators.
        
        Args:
            action: Dictionary containing action parameters (e.g., {'motor_1': 1, 'motor_2': -1})
            
        Returns:
            bool: True if action executed successfully, False otherwise
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> List[Dict[str, Any]]:
        """Return a list of the possible actions the robot can take.
        
        Returns:
            List of action dictionaries that can be passed to execute_action()
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and safely stop the robot."""
        pass