"""MultiSecondaryJoint policy implementation for LeRobot.

Controls a primary joint with discrete actions within each episode, while a
secondary joint changes position between episodes. Each episode randomly selects
a new position for the secondary joint.
"""

import random
import time
from typing import Dict, Optional

from torch import Tensor

from .base_joint_policy import BaseJointPolicy
from .configuration_multi_secondary_joint import MultiSecondaryJointConfig


class MultiSecondaryJointPolicy(BaseJointPolicy):
    """A policy that controls a primary joint at varying secondary joint positions.

    Each episode:
    1. Randomly selects a new secondary joint target (current + or - secondary_position_delta)
    2. Holds secondary joint at that target throughout the episode
    3. Executes primary joint action sequence (action 0=stay, 1=move+, 2=move-)

    The secondary joint change physically occurs during the reset phase between
    episodes (handled by run_lerobot_record.py's reset-phase patch). LeRobot's
    reset_time_s parameter should be set to allow the secondary servo to settle.

    On the first episode, no secondary joint command is issued — the servo stays
    at its physical position.
    """

    config_class = MultiSecondaryJointConfig
    name = "multi_secondary_joint"

    def __init__(self, config: MultiSecondaryJointConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._current_secondary_target: Optional[float] = None
        self._secondary_target_locked: bool = False
        self._current_episode: int = 1
        self._secondary_rng = random.Random(config.random_seed)

    def reset(self):
        """Reset for new episode. Selects a new random secondary joint target."""
        # Read current secondary position before resetting state
        current_pos = None
        if self._last_state is not None:
            current_pos = self._last_state[0, self.config.secondary_joint_index].item()
        elif self._current_secondary_target is not None:
            current_pos = self._current_secondary_target

        # If the reset-phase patch locked a target, preserve it across reset
        locked_target = None
        if self._secondary_target_locked:
            locked_target = self._current_secondary_target
            self._secondary_target_locked = False

        super().reset()

        if locked_target is not None:
            # Preserve target already set during reset phase (don't pick a new one)
            self._current_secondary_target = locked_target
        elif current_pos is not None:
            # Pick new random position delta from known current position
            delta = self.config.secondary_position_delta
            change = self._secondary_rng.choice([-delta, delta])
            self._current_secondary_target = max(-100, min(100, current_pos + change))
        else:
            # First episode: no position data yet — leave servo at physical position
            self._current_secondary_target = None

        self._current_episode += 1

    def get_reset_motor_targets(self) -> dict:
        """Return secondary motor target to command between episodes.

        Called by run_lerobot_record.py's reset-phase patch to physically move
        the secondary joint before the next episode begins recording.
        Returns empty dict on first episode (no target yet).
        """
        if self._current_secondary_target is None:
            return {}
        motor_name = self.config.secondary_joint_name.replace(".pos", "")
        return {motor_name: self._current_secondary_target}

    def _get_log_header_extra(self) -> dict:
        """Add secondary joint configuration to log header."""
        return {
            "policy_type": "multi_secondary_joint",
            "secondary_joint_name": self.config.secondary_joint_name,
            "secondary_position_delta": self.config.secondary_position_delta,
            "secondary_target": self._current_secondary_target,
            "episode_index": self._current_episode - 1,
        }

    def _compute_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute action controlling both primary and secondary joints.

        The primary joint follows discrete action logic (stay, move+, move-).
        The secondary joint is held at the current episode's target position.

        Args:
            batch: Dictionary with 'observation.state' tensor

        Returns:
            Action tensor of shape [B, action_dim] - absolute position targets
        """
        state = batch.get("observation.state")
        if state is None:
            raise ValueError("observation.state not found in batch")

        self._last_state = state.clone()
        batch_size = state.shape[0]
        device = state.device

        # Primary joint action timing
        current_time = time.time()
        action_changed = False
        frame_index = self._frame_counter
        self._frame_counter += 1

        if self._action_start_time is not None:
            elapsed = current_time - self._action_start_time
            if elapsed >= self.config.action_duration:
                self._current_action = self._get_discrete_action(batch_size, device).item()
                self._action_start_time = current_time
                action_changed = True
        else:
            self._current_action = self._get_discrete_action(batch_size, device).item()
            self._action_start_time = current_time
            action_changed = True

        # Compute primary target position when action changes
        if action_changed:
            self._log_discrete_action(current_time, self._current_action, frame_index)

            current_primary_pos = state[0, self.config.joint_index].item()

            if self._current_action == 1:
                self._target_position = current_primary_pos + self.config.position_delta
            elif self._current_action == 2:
                self._target_position = current_primary_pos - self.config.position_delta
            else:
                self._target_position = current_primary_pos

            self._target_position = max(-100, min(100, self._target_position))

        # Start with current positions as target (hold all joints)
        action = state.clone()

        # Set primary joint target
        if self._target_position is not None:
            action[:, self.config.joint_index] = self._target_position

        # Hold secondary joint at episode target (None = leave at current position)
        if self._current_secondary_target is not None:
            action[:, self.config.secondary_joint_index] = self._current_secondary_target

        return action
