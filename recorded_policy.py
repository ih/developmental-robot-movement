from typing import Dict, Any, Tuple, Optional, List, Callable
from recording_reader import RecordingReader

def create_recorded_action_selector(reader: RecordingReader, action_filter: Optional[Callable[[Dict[str, Any]], bool]] = None):
    """Create a recorded action selector function.

    This can be used as a drop-in replacement for select_action_by_uncertainty
    when in replay mode.

    Args:
        reader: RecordingReader instance
        action_filter: Optional filter function that returns True if action should be replayed.
                      If provided, actions not matching the filter are skipped.

    Returns:
        Function that returns (action, all_action_predictions) tuples matching select_action_by_uncertainty signature
    """
    def recorded_action_selector(current_features: Optional[Any] = None) -> Tuple[Dict[str, Any], List[Any]]:
        """Return recorded action for current step.

        Args:
            current_features: Current visual features (ignored in replay mode)

        Returns:
            Tuple of (action, all_action_predictions) where:
            - action: The recorded action for current step (matching filter if specified)
            - all_action_predictions: Empty list (no predictions in replay mode)
        """
        while True:
            try:
                recorded_action = reader.get_next_action()

                # Apply filter if specified
                if action_filter is not None:
                    if action_filter(recorded_action):
                        # Action matches filter - return it
                        return recorded_action, []
                    else:
                        # Action doesn't match filter - skip to next action
                        continue
                else:
                    # No filter - return all actions
                    return recorded_action, []

            except StopIteration:
                # End of recording - return a no-op action or raise
                raise StopIteration("End of recorded actions reached")

    return recorded_action_selector