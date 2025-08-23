import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional
from robot_interface import RobotInterface
from jetbot_remote_client import RemoteJetBot

logger = logging.getLogger(__name__)

class JetBotInterface(RobotInterface):
    """Concrete implementation of RobotInterface for JetBot robots."""
    
    def __init__(self, ip_address: str, port: int = 18861):
        """Initialize JetBot interface.
        
        Args:
            ip_address: IP address of the JetBot
            port: Port number for RPyC connection (default: 18861)
        """
        self.jetbot = RemoteJetBot(ip_address, port)
        
        # Define action space (cross product of motor values with fixed duration)
        motor_values = [-0.15, 0, 0.15]
        duration = 0.1  # Fixed duration in seconds
        self._action_space = []
        for left in motor_values:
            for right in motor_values:
                self._action_space.append({
                    'motor_left': left, 
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
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute motor action on JetBot with duration.
        
        Args:
            action: Dictionary with 'motor_left', 'motor_right', and 'duration' keys
            
        Returns:
            bool: True if action executed successfully, False otherwise
        """
        try:
            left_speed = action.get('motor_left', 0)
            right_speed = action.get('motor_right', 0)
            duration = action.get('duration', 0.1)
            
            # Set motors to specified speeds
            self.jetbot.set_motors(left_speed, right_speed)
            logger.debug(f"Executing action {action}: left={left_speed}, right={right_speed} for {duration}s")
            
            # Wait for specified duration
            time.sleep(duration)
            
            # Stop motors
            self.jetbot.set_motors(0, 0)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")
            return False
    
    @property
    def action_space(self) -> List[Dict[str, Any]]:
        """Get available actions for the JetBot.
        
        Returns:
            List of action dictionaries with 'motor_left' and 'motor_right' keys
        """
        return self._action_space.copy()
    
    def cleanup(self) -> None:
        """Clean up JetBot connection and stop motors."""
        try:
            self.jetbot.cleanup()
            logger.info("JetBot interface cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")