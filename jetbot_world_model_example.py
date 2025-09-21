#!/usr/bin/env python3
"""
Example script showing how to integrate JetBot with AdaptiveWorldModel
using the RobotInterface abstraction.
"""

from jetbot_interface import JetBotInterface
from adaptive_world_model import AdaptiveWorldModel
from recording_writer import RecordingWriter
from recording_robot import RecordingRobot
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
logging.getLogger('rpyc').setLevel(logging.WARNING)

def main():
    # Configuration
    JETBOT_IP = '192.168.68.51'  # Replace with your JetBot's IP address

    # Optional: Override learning rates (leave as None to use saved optimizer rates or config defaults)
    AUTOENCODER_LR = None    # Use saved optimizer rate or config default
    PREDICTOR_LR = None      # Use saved optimizer rate or config default

    # Check mode - this file only handles online and record modes
    if config.MODE == "replay":
        logger.error("For replay mode, use replay_session_example.py instead")
        sys.exit(1)

    # Create JetBot interface
    logger.info("Connecting to JetBot...")
    try:
        jetbot = JetBotInterface(JETBOT_IP)
        # Test initial connection
        if jetbot.get_observation() is None:
            raise ConnectionError("Failed to get initial observation")
    except Exception as e:
        logger.error(f"Problem connecting to JetBot: {e}")
        sys.exit(1)

    # Setup based on mode
    if config.MODE == "record":
        logger.info("Starting in RECORD mode...")
        logger.info(f"Recording to: {config.RECORDING_BASE_DIR}")

        # Create recording writer and recording robot wrapper
        try:
            writer = RecordingWriter(
                base_dir=config.RECORDING_BASE_DIR,
                session_name=config.RECORDING_SESSION_NAME,
                shard_size=config.RECORDING_SHARD_SIZE,
                max_shards=config.RECORDING_MAX_SHARDS
            )
            robot = RecordingRobot(jetbot, writer)
            logger.info(f"Recording session: {writer.session_name}")
            logger.info(f"Session path: {writer.get_session_path()}")
        except Exception as e:
            logger.error(f"Failed to initialize recording: {e}")
            jetbot.cleanup()
            sys.exit(1)

    else:  # config.MODE == "online" or default
        logger.info("Starting in ONLINE mode...")
        robot = jetbot

    # Create world model with appropriate robot interface
    logger.info("Initializing AdaptiveWorldModel...")

    world_model = AdaptiveWorldModel(
        robot,
        interactive=config.INTERACTIVE_MODE,
        wandb_project="jetbot-developmental-movement-testing",
        checkpoint_dir=config.DEFAULT_CHECKPOINT_DIR,
        autoencoder_lr=AUTOENCODER_LR,
        predictor_lr=PREDICTOR_LR
    )
    
    try:
        logger.info("Starting world model main loop...")
        logger.info("Press Ctrl+C to stop")
        world_model.main_loop()
        
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    
    finally:
        logger.info("Saving final checkpoint...")
        world_model.save_checkpoint()

        # Show recording stats if in record mode
        if config.MODE == "record":
            stats = robot.get_recording_stats()
            logger.info(f"Recording complete!")
            logger.info(f"Session: {stats['session_name']}")
            logger.info(f"Total steps: {stats['total_steps']}")
            logger.info(f"Final shard: {stats['current_shard']}")
            logger.info(f"Session saved to: {stats['session_path']}")

        logger.info("Cleaning up...")
        robot.cleanup()
        # Clean up wandb run
        if world_model.wandb_enabled:
            wandb.finish()

if __name__ == "__main__":
    main()