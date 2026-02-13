"""Configuration for SimpleJoint policy.

A simple non-learned policy for controlling a single joint of the SO-101 robot arm
with 3 discrete actions: stay, move positive, move negative.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("simple_joint")
@dataclass
class SimpleJointConfig(PreTrainedConfig):
    """Configuration for the SimpleJoint policy.

    This policy controls a single joint with 3 discrete actions:
    - Action 0: No movement (stay)
    - Action 1: Move in positive direction by position_delta
    - Action 2: Move in negative direction by position_delta

    All other joints maintain their current positions from observation.state.

    Policy modes:
    - Default (no flags set): Always outputs action 0
    - Random (use_random_policy=True): Randomly selects actions indefinitely
    - Sequence (action_sequence provided): Executes sequence once, then outputs action 0

    Note: Sequence mode does NOT wrap - after completing the sequence, the policy
    will continue outputting action 0 (stay) for the remainder of the episode.

    Args:
        joint_name: Name of the joint to control (default: "shoulder_pan.pos")
        action_duration: How long each discrete action lasts before selecting next action (default: 0.5s)
        position_delta: How far to move the joint when action 1 or 2 is selected (default: 0.1 radians)
        use_random_policy: If True, select actions randomly indefinitely (default: False)
        action_sequence: Optional fixed sequence of actions to execute once (default: None)
        random_seed: Seed for random action selection
        n_obs_steps: Number of observation steps (always 1 for this policy)
        n_action_steps: Number of action steps (always 1 for this policy)
    """

    # Joint configuration
    joint_name: str = "shoulder_pan.pos"

    # Movement parameters
    action_duration: float = 0.5  # How long each discrete action lasts (seconds)
    position_delta: float = 0.1   # How far to move when action 1 or 2 is selected (radians)

    # Policy mode
    use_random_policy: bool = False
    action_sequence: Optional[List[int]] = None
    random_seed: Optional[int] = None

    # Calibration metadata
    calibrated_action_duration: bool = False  # True if action_duration was auto-calibrated

    # Discrete action logging (for recording sessions)
    discrete_action_log_dir: Optional[str] = None   # Directory for discrete action logs
    discrete_action_log_path: Optional[str] = None  # Current episode's log path (set by reset())

    # Required PreTrainedConfig fields (fixed for this simple policy)
    n_obs_steps: int = 1
    n_action_steps: int = 1

    # Required abstract property implementations
    @property
    def observation_delta_indices(self) -> None:
        """This policy does not use delta observations."""
        return None

    @property
    def action_delta_indices(self) -> None:
        """This policy does not use delta actions."""
        return None

    @property
    def reward_delta_indices(self) -> None:
        """This policy does not use reward deltas."""
        return None

    # SO-101 joint names for reference and validation
    SO101_JOINTS: List[str] = field(default_factory=lambda: [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    ])

    # Computed field (set in __post_init__)
    joint_index: int = field(init=False, default=0)

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()

        # Validate joint name
        if self.joint_name not in self.SO101_JOINTS:
            raise ValueError(
                f"Unknown joint '{self.joint_name}'. "
                f"Valid joints: {self.SO101_JOINTS}"
            )

        # Set joint index based on joint name
        self.joint_index = self.SO101_JOINTS.index(self.joint_name)

        # Validate movement parameters
        if self.action_duration <= 0:
            raise ValueError("action_duration must be positive")
        if self.position_delta <= 0:
            raise ValueError("position_delta must be positive")

        # Validate action sequence if provided
        if self.action_sequence is not None:
            for action in self.action_sequence:
                if action not in [0, 1, 2]:
                    raise ValueError(
                        f"Invalid action {action} in sequence. Must be 0, 1, or 2"
                    )

    def get_optimizer_preset(self):
        """Return None since this policy has no trainable parameters."""
        return None

    def get_scheduler_preset(self):
        """Return None since this policy has no scheduler."""
        return None

    def validate_features(self):
        """Validate input/output feature compatibility.

        This simple policy only requires observation.state to be present.
        """
        if not hasattr(self, 'input_features') or self.input_features is None:
            return

        # Check that observation.state is present
        has_state = any(
            'state' in key.lower()
            for key in self.input_features.keys()
        )
        if not has_state:
            raise ValueError(
                "SimpleJointPolicy requires 'observation.state' in input_features"
            )
