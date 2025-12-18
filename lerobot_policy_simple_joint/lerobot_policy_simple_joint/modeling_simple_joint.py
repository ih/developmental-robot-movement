"""SimpleJoint policy implementation for LeRobot.

A simple non-learned policy that controls a single joint with 3 discrete actions:
- Action 0: Stay (no movement)
- Action 1: Move positive direction for configured duration
- Action 2: Move negative direction for configured duration
"""

import time
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from lerobot.common.policies.pretrained import PreTrainedPolicy

from .configuration_simple_joint import SimpleJointConfig


class SimpleJointPolicy(PreTrainedPolicy):
    """A simple policy that controls a single joint with discrete actions.

    This policy:
    - Observes the current joint states
    - Outputs velocity commands for the controlled joint
    - Actions 1 and 2 move the joint for a configured duration
    - Action 0 commands zero velocity (stay)
    - All other joints maintain zero velocity

    The policy can operate in three modes:
    1. Default: Always outputs action 0 (stay)
    2. Random: Randomly selects from actions 0, 1, 2
    3. Sequence: Cycles through a predefined action sequence
    """

    config_class = SimpleJointConfig
    name = "simple_joint"

    def __init__(
        self,
        config: SimpleJointConfig,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        """Initialize the SimpleJoint policy.

        Args:
            config: Policy configuration
            dataset_stats: Statistics for normalization (not used by this policy)
        """
        super().__init__(config)
        self.config = config

        # Internal state for action selection
        self._sequence_index = 0
        self._rng = None
        if config.random_seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(config.random_seed)

        # Movement state tracking
        self._current_action = 0
        self._action_start_time: Optional[float] = None
        self._last_state: Optional[Tensor] = None

        # Create a dummy parameter so PyTorch recognizes this as a module
        # This is needed for device placement
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def reset(self):
        """Reset policy state between episodes.

        Should be called on env.reset().
        """
        self._sequence_index = 0
        self._current_action = 0
        self._action_start_time = None
        self._last_state = None

        # Reset random generator if seed was provided
        if self.config.random_seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self.config.random_seed)

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Select an action based on current observations.

        Args:
            batch: Dictionary with 'observation.state' tensor of shape [B, state_dim]
                   where state_dim is 6 for SO-101 (6 joint positions)

        Returns:
            Action tensor of shape [B, action_dim] representing joint velocities
            For SO-101: 6-dimensional velocity commands
        """
        # Get current state
        state = batch.get("observation.state")
        if state is None:
            raise ValueError("observation.state not found in batch")

        self._last_state = state.clone()
        batch_size = state.shape[0]
        device = state.device
        num_joints = state.shape[1]  # Should be 6 for SO-101

        # Check if current action has completed its duration
        current_time = time.time()
        if self._action_start_time is not None:
            elapsed = current_time - self._action_start_time
            if elapsed >= self.config.move_duration:
                # Action duration completed, select new action
                self._current_action = self._get_discrete_action(batch_size, device).item()
                self._action_start_time = current_time
        else:
            # First action selection
            self._current_action = self._get_discrete_action(batch_size, device).item()
            self._action_start_time = current_time

        # Convert discrete action to velocity command
        # Start with zero velocities for all joints
        action = torch.zeros(batch_size, num_joints, device=device, dtype=state.dtype)

        # Apply velocity to controlled joint based on discrete action
        if self._current_action == 1:
            # Move positive direction
            action[:, self.config.joint_index] = self.config.move_speed
        elif self._current_action == 2:
            # Move negative direction
            action[:, self.config.joint_index] = -self.config.move_speed
        # Action 0: stay (velocity already 0)

        return action

    def _get_discrete_action(self, batch_size: int, device: torch.device) -> Tensor:
        """Get discrete action (0, 1, or 2).

        Args:
            batch_size: Number of actions to generate
            device: Target device

        Returns:
            Tensor of discrete actions, shape [B]
        """
        if self.config.action_sequence is not None:
            # Use predefined sequence
            action = self.config.action_sequence[self._sequence_index]
            self._sequence_index = (self._sequence_index + 1) % len(self.config.action_sequence)
            return torch.full((batch_size,), action, dtype=torch.long, device=device)

        elif self.config.use_random_policy:
            # Random action
            if self._rng is not None:
                return torch.randint(0, 3, (batch_size,), generator=self._rng).to(device)
            else:
                return torch.randint(0, 3, (batch_size,), device=device)

        else:
            # Default: no movement (action 0)
            return torch.zeros(batch_size, dtype=torch.long, device=device)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass for training.

        Since this is a non-learned policy, forward just returns zero loss.
        This method exists for compatibility with the training pipeline.

        Args:
            batch: Training batch

        Returns:
            Dictionary with loss tensor (always 0)
        """
        device = next(iter(batch.values())).device
        return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

    def get_optim_params(self):
        """Return parameters for optimization.

        Since this policy has no trainable parameters, return empty list.
        """
        return []

    @property
    def device(self) -> torch.device:
        """Get the device this policy is on."""
        return self._dummy.device

    def to(self, device):
        """Move policy to device."""
        super().to(device)
        return self
