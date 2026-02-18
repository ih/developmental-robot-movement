"""LeRobot Joint Policy Package.

Discrete joint control policies for the SO-101 robot arm:
- SimpleJoint: Controls a single joint with 3 discrete actions
- MultiHeightJoint: Controls shoulder_pan at varying shoulder_lift heights across episodes

Usage:
    # Install the package
    pip install -e .

    # Use with lerobot-record
    lerobot-record \\
        --robot.type=so101_follower \\
        --policy.type=simple_joint \\
        --policy.joint_name=shoulder_pan.pos \\
        --policy.action_duration=0.5 \\
        --policy.use_random_policy=true \\
        ...
"""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package:\n"
        "  pip install lerobot\n"
        "Or follow the LeRobot installation instructions."
    )

from .configuration_simple_joint import SimpleJointConfig
from .modeling_simple_joint import SimpleJointPolicy
from .configuration_multi_secondary_joint import MultiSecondaryJointConfig
from .modeling_multi_secondary_joint import MultiSecondaryJointPolicy
from .processor_simple_joint import make_simple_joint_pre_post_processors

__all__ = [
    "SimpleJointConfig",
    "SimpleJointPolicy",
    "MultiSecondaryJointConfig",
    "MultiSecondaryJointPolicy",
    "make_simple_joint_pre_post_processors",
]

__version__ = "0.1.0"
