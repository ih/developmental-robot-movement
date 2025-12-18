"""LeRobot SimpleJoint Policy Package.

A simple non-learned policy for controlling a single joint of the SO-101 robot arm
with 3 discrete actions (stay, move positive, move negative).

Usage:
    # Install the package
    pip install -e .

    # Use with lerobot-record
    lerobot-record \\
        --robot.type=so101_follower \\
        --policy.type=simple_joint \\
        --policy.joint_name=shoulder_pan.pos \\
        --policy.move_duration=0.5 \\
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

__all__ = [
    "SimpleJointConfig",
    "SimpleJointPolicy",
]

__version__ = "0.1.0"
