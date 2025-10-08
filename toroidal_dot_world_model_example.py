#!/usr/bin/env python3
"""
Example script showing how to integrate ToroidalDotRobot with AdaptiveWorldModel
using the RobotInterface abstraction.

This demonstrates the world model learning in a simple simulated environment.
"""

from toroidal_dot_interface import ToroidalDotRobot
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

def main():
    # Configuration - use toroidal dot parameters
    logger.info("Using Toroidal Dot Environment Configuration:")
    logger.info(f"  Image size: {config.ToroidalDotConfig.IMG_SIZE}x{config.ToroidalDotConfig.IMG_SIZE}")
    logger.info(f"  Dot radius: {config.ToroidalDotConfig.DOT_RADIUS} pixels")
    logger.info(f"  Movement per action: {config.ToroidalDotConfig.DOT_MOVE_PIXELS} pixels")

    # Optional: Override learning rates (leave as None to use saved optimizer rates or config defaults)
    AUTOENCODER_LR = None    # Use saved optimizer rate or config default
    PREDICTOR_LR = None      # Use saved optimizer rate or config default

    # Create ToroidalDotRobot interface
    logger.info("Creating ToroidalDotRobot...")
    try:
        toroidal_robot = ToroidalDotRobot(
            img_size=config.ToroidalDotConfig.IMG_SIZE,
            dot_radius=config.ToroidalDotConfig.DOT_RADIUS,
            move_pixels=config.ToroidalDotConfig.DOT_MOVE_PIXELS,
            action_delay=config.ToroidalDotConfig.DOT_ACTION_DELAY,
            seed=None  # Set to an integer for reproducibility
        )
        # Test initial observation
        if toroidal_robot.get_observation() is None:
            raise RuntimeError("Failed to get initial observation")
        logger.info("✅ ToroidalDotRobot created successfully")
    except Exception as e:
        logger.error(f"Problem creating ToroidalDotRobot: {e}")
        sys.exit(1)

    # Setup based on recording mode
    if config.RECORDING_MODE:
        logger.info("Starting in RECORD mode...")
        logger.info(f"Recording to: {config.TOROIDAL_DOT_RECORDING_DIR}")

        # Create recording writer and recording robot wrapper
        try:
            writer = RecordingWriter(
                base_dir=config.TOROIDAL_DOT_RECORDING_DIR,
                session_name=config.RECORDING_SESSION_NAME,
                shard_size=config.RECORDING_SHARD_SIZE,
                max_disk_gb=config.RECORDING_MAX_DISK_GB
            )
            robot = RecordingRobot(toroidal_robot, writer)
            logger.info(f"Recording session: {writer.session_name}")
            logger.info(f"Session path: {writer.get_session_path()}")
        except Exception as e:
            logger.error(f"Failed to initialize recording: {e}")
            toroidal_robot.cleanup()
            sys.exit(1)

    else:
        logger.info("Starting in ONLINE mode...")
        robot = toroidal_robot

    # Temporarily override ACTION_CHANNELS and ACTION_RANGES for toroidal dot
    original_action_channels = config.ACTION_CHANNELS
    original_action_ranges = config.ACTION_RANGES
    config.ACTION_CHANNELS = config.ToroidalDotConfig.ACTION_CHANNELS_DOT
    config.ACTION_RANGES = config.ToroidalDotConfig.ACTION_RANGES_DOT

    # Create world model with appropriate robot interface
    logger.info("Initializing AdaptiveWorldModel...")
    logger.info(f"Action space: {robot.action_space}")
    logger.info(f"Checkpoint directory: {config.TOROIDAL_DOT_CHECKPOINT_DIR}")

    world_model = AdaptiveWorldModel(
        robot,
        interactive=config.INTERACTIVE_MODE,
        wandb_project="toroidal-dot-developmental-learning",
        checkpoint_dir=config.TOROIDAL_DOT_CHECKPOINT_DIR,
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
        # Restore original action config
        config.ACTION_CHANNELS = original_action_channels
        config.ACTION_RANGES = original_action_ranges

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
