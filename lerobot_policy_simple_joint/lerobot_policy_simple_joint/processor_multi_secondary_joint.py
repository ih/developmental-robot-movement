"""Pre/post processors for MultiSecondaryJoint policy.

Uses the same identity processors as SimpleJoint since both policies work
directly with absolute joint positions without normalization.
"""

from .processor_simple_joint import make_simple_joint_pre_post_processors


def make_multi_secondary_joint_pre_post_processors(config, pretrained_path=None, **kwargs):
    """Create identity pre/post processors for MultiSecondaryJoint policy."""
    return make_simple_joint_pre_post_processors(config, pretrained_path, **kwargs)
