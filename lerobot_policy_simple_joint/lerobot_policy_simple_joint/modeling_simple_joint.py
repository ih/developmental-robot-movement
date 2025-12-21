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

from lerobot.policies.pretrained import PreTrainedPolicy

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

    def _compute_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute action based on current observations.

        Internal method used by both select_action and predict_action_chunk.

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

        # Start with current positions as the target (hold all joints in place)
        action = state.clone()

        # Apply position delta to controlled joint based on discrete action
        # Note: SO101 uses normalized range -100 to 100, so move_speed is in normalized units/second
        # Delta = speed × time_step (approximate with small fixed step)
        dt = 0.033  # ~30 FPS
        delta = self.config.move_speed * dt

        if self._current_action == 1:
            # Move positive direction
            action[:, self.config.joint_index] += delta
        elif self._current_action == 2:
            # Move negative direction
            action[:, self.config.joint_index] -= delta
        # Action 0: stay (position unchanged)

        # Clamp to valid range to avoid motor errors
        action[:, self.config.joint_index] = action[:, self.config.joint_index].clamp(-100, 100)

        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor], **kwargs) -> Tensor:
        """Predict an action chunk for the given observation.

        For this simple policy without action chunking, returns a single action
        as a chunk of size 1.

        Args:
            batch: Dictionary with observation tensors
            **kwargs: Additional arguments (ignored)

        Returns:
            Action tensor of shape [B, 1, action_dim]
        """
        action = self._compute_action(batch)
        # Add chunk dimension (size 1 since no action chunking)
        return action.unsqueeze(1)

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
        return self._compute_action(batch)

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
