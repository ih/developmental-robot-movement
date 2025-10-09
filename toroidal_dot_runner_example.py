#!/usr/bin/env python3
"""
Example script showing how to run ToroidalDotRobot with RobotRunner
using deterministic action selectors.

This demonstrates running a robot with pre-defined policies without
any learning or world model components.
"""

from toroidal_dot_interface import ToroidalDotRobot
from robot_runner import RobotRunner
from toroidal_action_selectors import (
    create_constant_action_selector,
    create_sequence_action_selector,
    SEQUENCE_ALWAYS_MOVE,
    SEQUENCE_ALWAYS_STAY,
    SEQUENCE_ALTERNATE,
    SEQUENCE_DOUBLE_MOVE,
    SEQUENCE_TRIPLE_MOVE,
    TOROIDAL_ACTION_STAY,
    TOROIDAL_ACTION_MOVE
)
from recording_writer import RecordingWriter
from recording_robot import RecordingRobot
import config
import logging
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

    # Choose action selector
    # Options: SEQUENCE_ALWAYS_MOVE, SEQUENCE_ALWAYS_STAY, SEQUENCE_ALTERNATE,
    #          SEQUENCE_DOUBLE_MOVE, SEQUENCE_TRIPLE_MOVE
    # Or create custom:
    #   - create_constant_action_selector(TOROIDAL_ACTION_MOVE)
    #   - create_sequence_action_selector([{'action': 0}, {'action': 1}, {'action': 1}])

    logger.info("Creating action selector: SEQUENCE_ALWAYS_MOVE")
    action_selector = create_sequence_action_selector(SEQUENCE_ALWAYS_MOVE)

    # Alternative examples (uncomment to try):
    # action_selector = create_sequence_action_selector(SEQUENCE_ALTERNATE)
    # action_selector = create_sequence_action_selector(SEQUENCE_TRIPLE_MOVE)
    # action_selector = create_constant_action_selector(TOROIDAL_ACTION_STAY)
    # custom_sequence = [{'action': 0}, {'action': 0}, {'action': 1}, {'action': 1}, {'action': 1}]
    # action_selector = create_sequence_action_selector(custom_sequence)

    # Create robot runner
    logger.info("Initializing RobotRunner...")
    logger.info(f"Action space: {robot.action_space}")

    runner = RobotRunner(
        robot_interface=robot,
        action_selector=action_selector,
        interactive=config.INTERACTIVE_MODE,
        display_interval=1,  # Update display every 10 steps
        action_delay=0.1  # 0.1 second delay between actions
    )

    try:
        logger.info("Starting robot runner main loop...")
        logger.info("Press Ctrl+C to stop")
        runner.main_loop()

    except KeyboardInterrupt:
        logger.info("Stopped by user")

    finally:
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


if __name__ == "__main__":
    main()
