"""Processor functions for SimpleJoint policy.

Creates identity (passthrough) pre/post processors since this is a non-learned
policy that works directly with absolute joint positions.
"""

from lerobot.policies.factory import (
    PolicyProcessorPipeline,
    batch_to_transition,
    transition_to_batch,
    policy_action_to_transition,
    transition_to_policy_action,
)


def make_simple_joint_pre_post_processors(config, pretrained_path=None, **kwargs):
    """Create identity pre/post processors for SimpleJoint policy.

    No transformations are needed since the policy reads observation.state
    directly and outputs absolute joint position targets.
    """
    preprocessor = PolicyProcessorPipeline(
        steps=[],
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
