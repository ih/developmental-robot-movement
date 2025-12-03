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


def create_random_duration_action_selector(
    min_duration: int = 5,
    max_duration: int = 15,
    seed: int = None
):
    """
    Create an action selector that randomly chooses actions and maintains them
    for a random number of consecutive steps.

    This selector is useful for testing if the world model can learn to condition
    on actions by providing sequences of consistent actions with varying durations.

    The selector randomly chooses between action 0 (stay) and action 1 (move right),
    then repeats that action for a randomly chosen duration between min_duration
    and max_duration steps (inclusive).

    Args:
        min_duration: Minimum number of steps to maintain an action (default 5)
        max_duration: Maximum number of steps to maintain an action (default 15)
        seed: Random seed for reproducibility (optional)

    Returns:
        Function that returns (action, all_action_predictions) tuples

    Example:
        # Create selector with 5-15 step durations
        selector = create_random_duration_action_selector(
            min_duration=5,
            max_duration=15,
            seed=42
        )

        # Use with world model or recording
        robot = ToroidalDotRobot(initial_y=112)  # Fixed y-position
        for _ in range(100):
            action, _ = selector()
            robot.execute_action(action)
    """
    if min_duration < 1:
        raise ValueError("min_duration must be at least 1")
    if max_duration < min_duration:
        raise ValueError("max_duration must be >= min_duration")

    # Use numpy for random number generation
    import numpy as np
    rng = np.random.RandomState(seed)

    # Use a mutable container to store state across calls
    state = {
        'current_action': None,
        'remaining_steps': 0
    }

    def random_duration_action_selector(
        current_features: Optional[Any] = None
    ) -> Tuple[Dict[str, Any], List[Any]]:
        """
        Return the current action, selecting a new one with random duration
        when the current duration expires.

        Args:
            current_features: Current visual features (ignored)

        Returns:
            Tuple of (action, all_action_predictions) where:
            - action: The current action (maintained for duration steps)
            - all_action_predictions: Empty list (no predictions computed)
        """
        # Check if we need to select a new action
        if state['remaining_steps'] <= 0:
            # Randomly choose action 0 or 1
            action_value = rng.randint(0, 2)  # 0 or 1
            state['current_action'] = {'action': action_value}

            # Randomly choose duration
            state['remaining_steps'] = rng.randint(min_duration, max_duration + 1)

        # Decrement remaining steps
        state['remaining_steps'] -= 1

        return state['current_action'], []

    return random_duration_action_selector
