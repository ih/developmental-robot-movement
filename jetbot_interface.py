import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional
from robot_interface import RobotInterface
from jetbot_remote_client import RemoteJetBot

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class JetBotInterface(RobotInterface):
    """Concrete implementation of RobotInterface for JetBot robots."""
    
    def __init__(self, ip_address: str, port: int = 18861):
        """Initialize JetBot interface.
        
        Args:
            ip_address: IP address of the JetBot
            port: Port number for RPyC connection (default: 18861)
        """
        self.jetbot = RemoteJetBot(ip_address, port)
        
        # Track current motor speeds for smooth ramping
        self.current_left_speed = 0.0
        self.current_right_speed = 0.0
        
        # Define action space (single motor only for simplified learning)
        # Forward-only movement for gentler gearbox operation
        motor_values = [0, 0.1]  # Stop and forward only
        duration = 0.2  # Increased duration for smoother motion
        self._action_space = []
        for right in motor_values:
            self._action_space.append({
                'motor_left': 0,  # Keep left motor at 0
                'motor_right': right,
                'duration': duration
            })
    
    def get_observation(self) -> Optional[np.ndarray]:
        """Capture current frame from JetBot camera.
        
        Returns:
            np.ndarray: Camera frame in RGB format (H, W, 3) or None if capture fails
        """
        try:
            frame = self.jetbot.get_frame()
            if frame is not None:
                # Convert BGR to RGB for consistency with world model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            return None
        except Exception as e:
            logger.error(f"Failed to get observation: {e}")
            return None
    
    def _ramp_motors(self, target_left: float, target_right: float, steps: int = 4, step_delay: float = 0.015) -> None:
        """Gradually ramp motors from current speeds to target speeds.
        
        Args:
            target_left: Target speed for left motor (-1.0 to 1.0)
            target_right: Target speed for right motor (-1.0 to 1.0)
            steps: Number of intermediate steps for ramping
            step_delay: Delay between each step in seconds
        """
        for i in range(1, steps + 1):
            # Calculate intermediate speeds using linear interpolation
            intermediate_left = self.current_left_speed + (target_left - self.current_left_speed) * (i / steps)
            intermediate_right = self.current_right_speed + (target_right - self.current_right_speed) * (i / steps)
            
            # Set motors to intermediate speeds
            self.jetbot.set_motors(intermediate_left, intermediate_right)
            time.sleep(step_delay)
        
        # Update current speeds
        self.current_left_speed = target_left
        self.current_right_speed = target_right
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute motor action on JetBot with smooth ramping for gearbox protection.
        
        Args:
            action: Dictionary with 'motor_left', 'motor_right', and 'duration' keys
            
        Returns:
            bool: True if action executed successfully, False otherwise
        """
        try:
            target_left = action.get('motor_left', 0)
            target_right = action.get('motor_right', 0)
            duration = action.get('duration', 0.2)
            
            logger.debug(f"Executing action {action}: left={target_left}, right={target_right} for {duration}s")
            
            # Gradually ramp to target speeds (gearbox-friendly)
            self._ramp_motors(target_left, target_right)
            
            # Hold at target speeds for remaining duration
            # (subtract ramping time: 4 steps * 0.015s = 0.06s)
            hold_duration = max(0, duration - 0.06)
            if hold_duration > 0:
                time.sleep(hold_duration)
            
            # Gradually ramp down to stop (gearbox-friendly)
            self._ramp_motors(0, 0)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")
            return False
    
    @property
    def action_space(self) -> List[Dict[str, Any]]:
        """Get available actions for the JetBot.
        
        Returns:
            List of action dictionaries with 'motor_left' and 'motor_right' keys
            (motor_left always 0, motor_right values: 0=stop, 0.1=forward)
            Total actions: 2 (stop, forward)
        """
        return self._action_space.copy()
    
    def cleanup(self) -> None:
        """Clean up JetBot connection and stop motors gently."""
        try:
            # Gently stop motors with ramping
            self._ramp_motors(0, 0)
            self.jetbot.cleanup()
            logger.info("JetBot interface cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")