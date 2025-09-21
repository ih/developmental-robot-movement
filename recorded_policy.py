from typing import Dict, Any, Tuple, Optional, List
from recording_reader import RecordingReader

def create_recorded_action_selector(reader: RecordingReader):
    """Create a recorded action selector function.

    This can be used as a drop-in replacement for select_action_by_uncertainty
    when in replay mode.

    Args:
        reader: RecordingReader instance

    Returns:
        Function that returns (action, all_action_predictions) tuples matching select_action_by_uncertainty signature
    """
    def recorded_action_selector(current_features: Optional[Any] = None) -> Tuple[Dict[str, Any], List[Any]]:
        """Return recorded action for current step.

        Args:
            current_features: Current visual features (ignored in replay mode)

        Returns:
            Tuple of (action, all_action_predictions) where:
            - action: The recorded action for current step
            - all_action_predictions: Empty list (no predictions in replay mode)
        """
        try:
            recorded_action = reader.get_next_action()
            # Return empty predictions list since we don't generate predictions in replay mode
            return recorded_action, []

        except StopIteration:
            # End of recording - return a no-op action or raise
            raise StopIteration("End of recorded actions reached")

    return recorded_action_selector