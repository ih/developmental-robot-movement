"""Base class for discrete joint control policies.

Provides shared functionality for policies that control SO-101 joints with
discrete actions (stay, move positive, move negative). Subclasses implement
_compute_action() for joint-specific control logic.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy


class BaseJointPolicy(PreTrainedPolicy):
    """Base class for policies that control joints with discrete actions.

    Provides:
    - Wall-clock action timing (_action_start_time, _current_action)
    - Discrete action logging (_write_log_header, _log_discrete_action)
    - Frame counting
    - _get_discrete_action (sequence/random/default modes)
    - forward, get_optim_params, device, to, predict_action_chunk, select_action

    Subclasses must implement:
    - _compute_action(batch) -> Tensor
    - config_class (class attribute)
    - name (class attribute)
    """

    # Placeholders required by PreTrainedPolicy.__init_subclass__
    # Subclasses MUST override these with their actual config class and name
    config_class = PreTrainedConfig
    name = "base_joint"

    def __init__(self, config, **kwargs):
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
        self._sequence_completed = False
        self._frame_counter = 0

        # Dummy parameter for device placement
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def reset(self):
        """Reset policy state between episodes.

        Subclasses should call super().reset() first, then do their own setup.
        """
        self._sequence_index = 0
        self._current_action = 0
        self._action_start_time = None
        self._last_state = None
        self._target_position = None
        self._sequence_completed = False
        self._frame_counter = 0

        # Reset random generator if seed was provided
        if self.config.random_seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self.config.random_seed)

        # Set up per-episode log file if log directory is configured
        if self.config.discrete_action_log_dir:
            log_dir = Path(self.config.discrete_action_log_dir)
            existing = sorted(log_dir.glob("episode_*.jsonl"))
            episode_num = len(existing)
            self.config.discrete_action_log_path = str(
                log_dir / f"episode_{episode_num:06d}.jsonl"
            )
            self._write_log_header()

    def _get_log_header_extra(self) -> dict:
        """Return extra fields for the log header. Override in subclasses."""
        return {}

    def get_reset_motor_targets(self) -> dict:
        """Return {motor_name: position} to command during reset between episodes.

        Motor names are bus names (e.g. "shoulder_lift", not "shoulder_lift.pos").
        Override in subclasses to command specific motors between episodes.
        """
        return {}

    def _write_log_header(self):
        """Write metadata header to log file with recording parameters."""
        if not self.config.discrete_action_log_path:
            return

        log_path = Path(self.config.discrete_action_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        header = {
            "type": "header",
            "joint_name": self.config.joint_name,
            "action_duration": self.config.action_duration,
            "position_delta": self.config.position_delta,
            "calibrated": self.config.calibrated_action_duration,
            "use_random_policy": self.config.use_random_policy,
            "action_sequence": self.config.action_sequence,
            "random_seed": self.config.random_seed,
        }
        header.update(self._get_log_header_extra())

        with open(self.config.discrete_action_log_path, "w") as f:
            f.write(json.dumps(header) + "\n")

    def _log_discrete_action(self, timestamp: float, discrete_action: int,
                             frame_index: int):
        """Log a discrete action decision to the log file."""
        if not self.config.discrete_action_log_path:
            return

        with open(self.config.discrete_action_log_path, "a") as f:
            f.write(json.dumps({
                "type": "action",
                "timestamp": timestamp,
                "discrete_action": discrete_action,
                "frame_index": frame_index
            }) + "\n")

    def _get_discrete_action(self, batch_size: int, device: torch.device) -> Tensor:
        """Get discrete action (0, 1, or 2).

        Returns:
            Tensor of discrete actions, shape [B]
        """
        if self.config.action_sequence is not None:
            # Use predefined sequence (no wrapping - stay at action 0 after completion)
            if self._sequence_completed:
                return torch.zeros(batch_size, dtype=torch.long, device=device)

            action = self.config.action_sequence[self._sequence_index]
            self._sequence_index += 1

            if self._sequence_index >= len(self.config.action_sequence):
                self._sequence_completed = True

            return torch.full((batch_size,), action, dtype=torch.long, device=device)

        elif self.config.use_random_policy:
            if self._rng is not None:
                return torch.randint(0, 3, (batch_size,), generator=self._rng).to(device)
            else:
                return torch.randint(0, 3, (batch_size,), device=device)

        else:
            # Default: no movement (action 0)
            return torch.zeros(batch_size, dtype=torch.long, device=device)

    def _compute_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute action based on current observations.

        Subclasses must implement this method.

        Args:
            batch: Dictionary with 'observation.state' tensor

        Returns:
            Action tensor of shape [B, action_dim]
        """
        raise NotImplementedError("Subclasses must implement _compute_action")

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor], **kwargs) -> Tensor:
        """Predict an action chunk (size 1 since no action chunking)."""
        action = self._compute_action(batch)
        return action.unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Select an action based on current observations."""
        return self._compute_action(batch)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass for training. Returns zero loss (non-learned policy)."""
        device = next(iter(batch.values())).device
        return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

    def get_optim_params(self):
        """No trainable parameters."""
        return []

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def to(self, device):
        super().to(device)
        return self
