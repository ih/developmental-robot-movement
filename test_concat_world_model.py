"""
Test script for AutoencoderConcatPredictorWorldModel
"""

import torch
import config
from autoencoder_concat_predictor_world_model import AutoencoderConcatPredictorWorldModel
from recording_reader import RecordingReader
from replay_robot import ReplayRobot
from toroidal_action_selectors import SEQUENCE_ALWAYS_MOVE, create_sequence_action_selector
from session_explorer_lib import load_session_metadata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load session
session_dir = "saved/sessions/toroidal_dot/session_move_right"
print(f"\nLoading session: {session_dir}")

metadata = load_session_metadata(session_dir)
print(f"Metadata: {metadata}")

action_space = metadata.get("action_space", [])
print(f"Action space: {action_space}")

# Create recording reader
reader = RecordingReader(session_dir)
print(f"Reader created with {reader.total_steps} steps")

# Create replay robot
replay_robot = ReplayRobot(reader, action_space)
print(f"Replay robot created")

# Create world model
toroidal_selector = create_sequence_action_selector(SEQUENCE_ALWAYS_MOVE)

def action_selector_adapter(observation, action_space):
    action, _ = toroidal_selector()
    return action

world_model = AutoencoderConcatPredictorWorldModel(
    replay_robot,
    action_selector=action_selector_adapter,
    device=device
)
print(f"World model created")

# Run for a few iterations
print("\nRunning world model for 5 iterations...")
try:
    world_model.run(max_iterations=5)
    print("\nCompleted successfully!")

    # Check state
    print(f"\nFinal state:")
    print(f"- History length: {len(world_model.interleaved_history)}")
    print(f"- Has last prediction: {world_model.last_prediction is not None}")
    print(f"- Has training canvas: {world_model.last_training_canvas is not None}")
    print(f"- Training loss: {world_model.last_training_loss}")
    print(f"- Training iterations: {world_model.last_training_iterations}")
    print(f"- Has prediction canvas: {world_model.last_prediction_canvas is not None}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
