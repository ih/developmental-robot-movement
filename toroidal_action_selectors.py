"""
Toroidal Action Selectors

Action selector functions for the toroidal dot environment.
These can be used as drop-in replacements for the default uncertainty-based
action selection in AdaptiveWorldModel.

Usage:
    from toroidal_action_selectors import create_constant_action_selector, create_sequence_action_selector

    # Constant action selector
    selector = create_constant_action_selector({'action': 1})
    model = AdaptiveWorldModel(robot, action_selector=selector)

    # Sequence action selector
    sequence = [{'action': 0}, {'action': 1}, {'action': 1}]
    selector = create_sequence_action_selector(sequence)
    model = AdaptiveWorldModel(robot, action_selector=selector)
"""

from typing import Dict, Any, List, Tuple, Optional


def create_constant_action_selector(action: Dict[str, Any]):
    """
    Create an action selector that always returns the same action.

    This is useful for testing the world model's predictions when the robot
    performs a consistent behavior (e.g., always moving right, or always staying still).

    Args:
        action: The action to always select, e.g., {'action': 1}

    Returns:
        Function that returns (action, all_action_predictions) tuples

    Example:
        # Always move right
        selector = create_constant_action_selector({'action': 1})
        model = AdaptiveWorldModel(robot, action_selector=selector)

        # Always stay still
        selector = create_constant_action_selector({'action': 0})
        model = AdaptiveWorldModel(robot, action_selector=selector)
    """
    def constant_action_selector(current_features: Optional[Any] = None) -> Tuple[Dict[str, Any], List[Any]]:
        """
        Return the same action every time.

        Args:
            current_features: Current visual features (ignored)

        Returns:
            Tuple of (action, all_action_predictions) where:
            - action: The constant action specified at creation
            - all_action_predictions: Empty list (no predictions computed)
        """
        return action, []

    return constant_action_selector


def create_sequence_action_selector(action_sequence: List[Dict[str, Any]]):
    """
    Create an action selector that cycles through a sequence of actions.

    This is useful for testing predictable movement patterns, such as:
    - Alternating between stay and move
    - Multi-step sequences like "stay, stay, move, move"
    - Complex patterns for testing prediction accuracy

    The selector will iterate through the sequence repeatedly, wrapping back
    to the beginning when it reaches the end.

    Args:
        action_sequence: List of actions to cycle through, e.g.,
                        [{'action': 0}, {'action': 1}]

    Returns:
        Function that returns (action, all_action_predictions) tuples

    Example:
        # Alternate between stay and move
        sequence = [{'action': 0}, {'action': 1}]
        selector = create_sequence_action_selector(sequence)
        model = AdaptiveWorldModel(robot, action_selector=selector)

        # Stay twice, move once, repeat
        sequence = [{'action': 0}, {'action': 0}, {'action': 1}]
        selector = create_sequence_action_selector(sequence)
        model = AdaptiveWorldModel(robot, action_selector=selector)
    """
    if not action_sequence:
        raise ValueError("action_sequence cannot be empty")

    # Use a mutable container to store state across calls
    state = {'index': 0}

    def sequence_action_selector(current_features: Optional[Any] = None) -> Tuple[Dict[str, Any], List[Any]]:
        """
        Return the next action in the sequence, wrapping around when reaching the end.

        Args:
            current_features: Current visual features (ignored)

        Returns:
            Tuple of (action, all_action_predictions) where:
            - action: The current action from the sequence
            - all_action_predictions: Empty list (no predictions computed)
        """
        # Get current action
        current_action = action_sequence[state['index']]

        # Advance to next action, wrapping around
        state['index'] = (state['index'] + 1) % len(action_sequence)

        return current_action, []

    return sequence_action_selector


# Convenience action sequences for toroidal dot environment
TOROIDAL_ACTION_STAY = {'action': 0}
TOROIDAL_ACTION_MOVE = {'action': 1}

# Pre-defined common sequences
SEQUENCE_ALWAYS_MOVE = [TOROIDAL_ACTION_MOVE]
SEQUENCE_ALWAYS_STAY = [TOROIDAL_ACTION_STAY]
SEQUENCE_ALTERNATE = [TOROIDAL_ACTION_STAY, TOROIDAL_ACTION_MOVE]
SEQUENCE_DOUBLE_MOVE = [TOROIDAL_ACTION_MOVE, TOROIDAL_ACTION_MOVE, TOROIDAL_ACTION_STAY]
SEQUENCE_TRIPLE_MOVE = [TOROIDAL_ACTION_MOVE, TOROIDAL_ACTION_MOVE, TOROIDAL_ACTION_MOVE, TOROIDAL_ACTION_STAY]
