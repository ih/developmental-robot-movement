"""Configuration for MultiSecondaryJoint policy.

A policy that controls a primary joint with discrete actions within each episode,
while a secondary joint changes position between episodes. Each episode randomly
selects a new position for the secondary joint.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("multi_secondary_joint")
@dataclass
class MultiSecondaryJointConfig(PreTrainedConfig):
    """Configuration for the MultiSecondaryJoint policy.

    This policy controls two joints:
    - Primary joint (joint_name): Moves within each episode using discrete actions
      (action 0=stay, 1=move+, 2=move-)
    - Secondary joint (secondary_joint_name): Moves to a random position between
      episodes. The target is randomly chosen as current_pos +/- secondary_position_delta.

    Args:
        joint_name: Primary joint to control within episodes (default: shoulder_pan.pos)
        secondary_joint_name: Secondary joint that changes between episodes (default: shoulder_lift.pos)
        secondary_position_delta: Max magnitude for random secondary joint changes per episode
        action_duration: How long each discrete primary action lasts (seconds)
        position_delta: How far the primary joint moves per action
        use_random_policy: If True, select primary actions randomly
        action_sequence: Optional fixed sequence of primary actions to execute once per episode
        random_seed: Seed for reproducible random behavior
    """

    # Primary joint configuration (moves within each episode)
    joint_name: str = "shoulder_pan.pos"

    # Secondary joint configuration (changes between episodes)
    secondary_joint_name: str = "shoulder_lift.pos"
    secondary_position_delta: float = 10.0  # Max magnitude for random secondary change

    # Movement parameters (for primary joint)
    action_duration: float = 0.5
    position_delta: float = 0.1

    # Policy mode (for primary movements)
    use_random_policy: bool = False
    action_sequence: Optional[List[int]] = None
    random_seed: Optional[int] = None

    # Calibration metadata
    calibrated_action_duration: bool = False

    # Discrete action logging
    discrete_action_log_dir: Optional[str] = None
    discrete_action_log_path: Optional[str] = None

    # Required PreTrainedConfig fields
    n_obs_steps: int = 1
    n_action_steps: int = 1

    # Required abstract property implementations
    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    # SO-101 joint names
    SO101_JOINTS: List[str] = field(default_factory=lambda: [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    ])

    # Computed fields
    joint_index: int = field(init=False, default=0)
    secondary_joint_index: int = field(init=False, default=1)

    def __post_init__(self):
        super().__post_init__()

        # Validate primary joint
        if self.joint_name not in self.SO101_JOINTS:
            raise ValueError(
                f"Unknown primary joint '{self.joint_name}'. "
                f"Valid joints: {self.SO101_JOINTS}"
            )

        # Validate secondary joint
        if self.secondary_joint_name not in self.SO101_JOINTS:
            raise ValueError(
                f"Unknown secondary joint '{self.secondary_joint_name}'. "
                f"Valid joints: {self.SO101_JOINTS}"
            )

        if self.joint_name == self.secondary_joint_name:
            raise ValueError("Primary and secondary joints must be different")

        self.joint_index = self.SO101_JOINTS.index(self.joint_name)
        self.secondary_joint_index = self.SO101_JOINTS.index(self.secondary_joint_name)

        # Validate movement parameters
        if self.action_duration <= 0:
            raise ValueError("action_duration must be positive")
        if self.position_delta <= 0:
            raise ValueError("position_delta must be positive")
        if self.secondary_position_delta <= 0:
            raise ValueError("secondary_position_delta must be positive")

        # Validate action sequence if provided
        if self.action_sequence is not None:
            for action in self.action_sequence:
                if action not in [0, 1, 2]:
                    raise ValueError(
                        f"Invalid action {action} in sequence. Must be 0, 1, or 2"
                    )

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        if not hasattr(self, 'input_features') or self.input_features is None:
            return
        has_state = any(
            'state' in key.lower()
            for key in self.input_features.keys()
        )
        if not has_state:
            raise ValueError(
                "MultiSecondaryJointPolicy requires 'observation.state' in input_features"
            )
