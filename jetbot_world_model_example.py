#!/usr/bin/env python3
"""
Example script showing how to integrate JetBot with AutoencoderLatentPredictorWorldModel
using the RobotInterface abstraction.
"""

from jetbot_interface import JetBotInterface
from autoencoder_latent_predictor_world_model import AutoencoderLatentPredictorWorldModel
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
    PREDICTOR_LR = 2e-4      # Use saved optimizer rate or config default

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

    # Setup based on recording mode
    if config.RECORDING_MODE:
        logger.info("Starting in RECORD mode...")
        logger.info(f"Recording to: {config.JETBOT_RECORDING_DIR}")

        # Create recording writer and recording robot wrapper
        try:
            writer = RecordingWriter(
                base_dir=config.JETBOT_RECORDING_DIR,
                session_name=config.RECORDING_SESSION_NAME,
                shard_size=config.RECORDING_SHARD_SIZE,
                max_disk_gb=config.RECORDING_MAX_DISK_GB
            )
            robot = RecordingRobot(jetbot, writer)
            logger.info(f"Recording session: {writer.session_name}")
            logger.info(f"Session path: {writer.get_session_path()}")
        except Exception as e:
            logger.error(f"Failed to initialize recording: {e}")
            jetbot.cleanup()
            sys.exit(1)

    else:
        logger.info("Starting in ONLINE mode...")
        robot = jetbot

    # Create world model with appropriate robot interface
    logger.info("Initializing AutoencoderLatentPredictorWorldModel...")

    world_model = AutoencoderLatentPredictorWorldModel(
        robot,
        interactive=config.INTERACTIVE_MODE,
        wandb_project="jetbot-developmental-movement-testing",
        checkpoint_dir=config.JETBOT_CHECKPOINT_DIR,
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
        if config.RECORDING_MODE:
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