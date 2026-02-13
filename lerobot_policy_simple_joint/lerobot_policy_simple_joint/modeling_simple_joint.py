"""SimpleJoint policy implementation for LeRobot.

A simple non-learned policy that controls a single joint with 3 discrete actions:
- Action 0: Stay (no movement)
- Action 1: Move positive direction by position_delta
- Action 2: Move negative direction by position_delta
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_simple_joint import SimpleJointConfig


class SimpleJointPolicy(PreTrainedPolicy):
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
        self._target_position: Optional[float] = None
        self._sequence_completed = False  # Track if action sequence has completed

        # Create a dummy parameter so PyTorch recognizes this as a module
        # This is needed for device placement
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def reset(self):
        """Reset policy state between episodes.

        Should be called on env.reset().
        If discrete_action_log_dir is set, creates a per-episode log file
        and writes the header automatically.
        """
        self._sequence_index = 0
        self._current_action = 0
        self._action_start_time = None
        self._last_state = None
        self._target_position = None
        self._sequence_completed = False  # Reset sequence completion for new episode

        # Reset random generator if seed was provided
        if self.config.random_seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self.config.random_seed)

        # Set up per-episode log file if log directory is configured
        if self.config.discrete_action_log_dir:
            log_dir = Path(self.config.discrete_action_log_dir)
            # Auto-detect episode number from existing log files
            existing = sorted(log_dir.glob("episode_*.jsonl"))
            episode_num = len(existing)
            self.config.discrete_action_log_path = str(
                log_dir / f"episode_{episode_num:06d}.jsonl"
            )
            self._write_log_header()

    def _write_log_header(self):
        """Write metadata header to log file with recording parameters.

        Called when a new episode starts to write the header with all
        recording configuration parameters.
        """
        if not self.config.discrete_action_log_path:
            return

        # Create log directory lazily (first time writing)
        # This avoids creating the cache directory before LeRobot does
        log_path = Path(self.config.discrete_action_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config.discrete_action_log_path, "w") as f:
            f.write(json.dumps({
                "type": "header",
                "joint_name": self.config.joint_name,
                "action_duration": self.config.action_duration,
                "position_delta": self.config.position_delta,
                "calibrated": self.config.calibrated_action_duration,
                "use_random_policy": self.config.use_random_policy,
                "action_sequence": self.config.action_sequence,
                "random_seed": self.config.random_seed
            }) + "\n")

    def _log_discrete_action(self, timestamp: float, discrete_action: int):
        """Log a discrete action decision to the log file.

        Args:
            timestamp: When the action decision was made
            discrete_action: The discrete action chosen (0, 1, or 2)
        """
        if not self.config.discrete_action_log_path:
            return

        with open(self.config.discrete_action_log_path, "a") as f:
            f.write(json.dumps({
                "type": "action",
                "timestamp": timestamp,
                "discrete_action": discrete_action
            }) + "\n")

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

        # Check if current action has completed its duration.
        # action_duration should be calibrated to include servo settling time.
        current_time = time.time()
        action_changed = False

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
        # This allows smooth motion instead of choppy per-frame increments
        if action_changed:
            # Log the discrete action decision
            self._log_discrete_action(current_time, self._current_action)

            current_pos = state[0, self.config.joint_index].item()

            if self._current_action == 1:
                # Move positive direction by position_delta
                self._target_position = current_pos + self.config.position_delta
            elif self._current_action == 2:
                # Move negative direction by position_delta
                self._target_position = current_pos - self.config.position_delta
            else:
                # Action 0: stay at current position
                self._target_position = current_pos

            # Clamp to valid range to avoid motor errors
            self._target_position = max(-100, min(100, self._target_position))

        # Start with current positions as the target (hold all joints in place)
        action = state.clone()

        # Set controlled joint to computed target position
        # The servo will smoothly interpolate to this target
        if self._target_position is not None:
            action[:, self.config.joint_index] = self._target_position

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
            # Use predefined sequence (no wrapping - stay at action 0 after completion)
            if self._sequence_completed:
                # Sequence has completed, return action 0 (stay)
                return torch.zeros(batch_size, dtype=torch.long, device=device)

            # Get current action from sequence
            action = self.config.action_sequence[self._sequence_index]
            self._sequence_index += 1

            # Check if we've reached the end of the sequence
            if self._sequence_index >= len(self.config.action_sequence):
                self._sequence_completed = True

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
