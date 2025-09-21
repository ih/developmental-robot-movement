#!/usr/bin/env python3
"""
Example script showing how to replay recorded robot sessions
using the AdaptiveWorldModel with RecordingReader and ReplayRobot.
"""

from adaptive_world_model import AdaptiveWorldModel
from recording_reader import RecordingReader
from replay_robot import ReplayRobot
from recorded_policy import create_recorded_action_selector
import config
import logging
import wandb
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress debug messages from 3rd party packages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def main():
    # Configuration
    SESSION_DIR = config.REPLAY_SESSION_DIR
    CHECKPOINT_DIR = config.DEFAULT_CHECKPOINT_DIR  # Shared checkpoint directory

    logger.info("Starting robot session replay...")
    logger.info(f"Loading session from: {SESSION_DIR}")

    try:
        # Create recording reader and replay components
        reader = RecordingReader(SESSION_DIR)
        action_space = reader.get_action_space()
        robot = ReplayRobot(reader, action_space)
        action_selector = create_recorded_action_selector(reader)

        session_info = reader.get_session_info()
        robot_type = session_info.get('robot_type', 'unknown')

        logger.info(f"Replay setup complete: {reader.total_steps} steps loaded")
        logger.info(f"Robot type: {robot_type}")
        logger.info(f"Action space has {len(action_space)} actions")

    except Exception as e:
        logger.error(f"Failed to load replay session: {e}")
        sys.exit(1)

    # Create world model for replay (no interactive mode, no wandb by default)
    logger.info("Initializing AdaptiveWorldModel for replay...")

    world_model = AdaptiveWorldModel(
        robot,
        interactive=False,  # No interactive mode in replay
        wandb_project="jetbot_developmental-movement-replay",  # Disable wandb for replay by default
        checkpoint_dir=CHECKPOINT_DIR,
        action_selector=action_selector,
        autoencoder_lr=None,
        predictor_lr=None
    )

    # Run the replay
    logger.info("Starting replay...")
    try:
        world_model.main_loop()
        logger.info("Replay completed successfully!")
    except KeyboardInterrupt:
        logger.info("Replay interrupted by user")
    except StopIteration:
        logger.info("Replay finished - end of recording reached")
    except Exception as e:
        logger.error(f"Replay failed: {e}")
        raise
    finally:
        # Save final checkpoint after replay
        world_model.save_checkpoint()

        # Cleanup
        robot.cleanup()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()