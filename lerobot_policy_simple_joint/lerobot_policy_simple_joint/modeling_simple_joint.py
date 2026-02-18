"""SimpleJoint policy implementation for LeRobot.

A simple non-learned policy that controls a single joint with 3 discrete actions:
- Action 0: Stay (no movement)
- Action 1: Move positive direction by position_delta
- Action 2: Move negative direction by position_delta
"""

import time
from typing import Dict

from torch import Tensor

from .base_joint_policy import BaseJointPolicy
from .configuration_simple_joint import SimpleJointConfig


class SimpleJointPolicy(BaseJointPolicy):
    """A simple policy that controls a single joint with discrete actions.

    This policy:
    - Observes the current joint states
    - Outputs absolute position targets for the controlled joint
    - Actions 1 and 2 move the joint by position_delta in +/- direction
    - Action 0 maintains current position (stay)
    - All other joints maintain their current positions
    - Each discrete action lasts for action_duration before selecting next action

    The policy can operate in three modes:
    1. Default: Always outputs action 0 (stay)
    2. Random: Randomly selects from actions 0, 1, 2 (infinite)
    3. Sequence: Executes action sequence once, then stays at action 0 (no wrapping)
    """

    config_class = SimpleJointConfig
    name = "simple_joint"

    def _compute_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute action based on current observations.

        The SO101 follower expects absolute position targets, so this method:
        1. Reads current joint positions from observation.state
        2. Applies velocity-based delta to the controlled joint
        3. Returns absolute position targets for all joints

        Args:
            batch: Dictionary with 'observation.state' tensor

        Returns:
            Action tensor of shape [B, action_dim] - absolute position targets
        """
        # Get current state (joint positions)
        state = batch.get("observation.state")
        if state is None:
            raise ValueError("observation.state not found in batch")

        self._last_state = state.clone()
        batch_size = state.shape[0]
        device = state.device

        # Check if current action has completed its duration.
        current_time = time.time()
        action_changed = False
        frame_index = self._frame_counter
        self._frame_counter += 1

        if self._action_start_time is not None:
            elapsed = current_time - self._action_start_time
            if elapsed >= self.config.action_duration:
                # Action + settle completed, select new action
                self._current_action = self._get_discrete_action(batch_size, device).item()
                self._action_start_time = current_time
                action_changed = True
        else:
            # First action selection
            self._current_action = self._get_discrete_action(batch_size, device).item()
            self._action_start_time = current_time
            action_changed = True

        # Compute target position when action changes
        if action_changed:
            self._log_discrete_action(current_time, self._current_action,
                                      frame_index)

            current_pos = state[0, self.config.joint_index].item()

            if self._current_action == 1:
                self._target_position = current_pos + self.config.position_delta
            elif self._current_action == 2:
                self._target_position = current_pos - self.config.position_delta
            else:
                self._target_position = current_pos

            self._target_position = max(-100, min(100, self._target_position))

        # Start with current positions as the target (hold all joints in place)
        action = state.clone()

        # Set controlled joint to computed target position
        if self._target_position is not None:
            action[:, self.config.joint_index] = self._target_position

        return action
