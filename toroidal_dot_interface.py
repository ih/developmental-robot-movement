"""
Toroidal Dot Robot Interface

Implementation of RobotInterface for the toroidal white dot environment.
"""

import time
import numpy as np
from typing import Dict, List
from robot_interface import RobotInterface
from toroidal_dot_env import ToroidalDotEnvironment


class ToroidalDotRobot(RobotInterface):
    """
    Robot interface for the toroidal dot environment.

    This provides a RobotInterface implementation for the simulated
    toroidal environment, making it compatible with AdaptiveWorldModel,
    recording, and replay systems.
    """

    def __init__(self,
                 img_size: int = 224,
                 dot_radius: int = 2,
                 move_pixels: int = 7,
                 action_delay: float = 0.0,
                 seed: int = None):
        """
        Initialize the toroidal dot robot interface.

        Args:
            img_size: Size of the square image (default 224)
            dot_radius: Radius of the white dot in pixels (default 2)
            move_pixels: Number of pixels to move right when action=1 (default 7)
            action_delay: Delay in seconds after each action (default 0.0)
            seed: Random seed for reproducibility (optional)
        """
        self.env = ToroidalDotEnvironment(
            img_size=img_size,
            dot_radius=dot_radius,
            move_pixels=move_pixels,
            seed=seed
        )
        self.action_delay = action_delay

    def get_observation(self) -> np.ndarray:
        """
        Get current observation from the environment.

        Returns:
            224x224x3 RGB numpy array
        """
        return self.env.render()

    def execute_action(self, action: Dict) -> None:
        """
        Execute an action in the environment.

        Args:
            action: Dictionary with 'action' key (0 or 1)
                   0 = no movement
                   1 = move right
        """
        action_value = action['action']
        self.env.step(action_value)

        # Optional delay for consistency with physical robots
        if self.action_delay > 0:
            time.sleep(self.action_delay)

    @property
    def action_space(self) -> List[Dict]:
        """
        Get the action space for this robot.

        Returns:
            List of valid actions: [{'action': 0}, {'action': 1}]
        """
        return [
            {'action': 0},  # No movement
            {'action': 1}   # Move right
        ]

    def cleanup(self) -> None:
        """
        Cleanup resources (none needed for simulated environment).
        """
        pass

    def reset(self) -> np.ndarray:
        """
        Reset the environment to a new random vertical position.

        Returns:
            Initial observation after reset
        """
        return self.env.reset()

    def get_position(self):
        """
        Get current dot position (useful for debugging/testing).

        Returns:
            (x, y) tuple of current dot position
        """
        return self.env.get_position()
