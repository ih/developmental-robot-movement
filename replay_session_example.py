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
import os
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress debug messages from 3rd party packages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def replay_single_session(session_dir, checkpoint_dir):
    """Replay a single session directory"""
    logger.info(f"Loading session from: {session_dir}")

    try:
        # Create recording reader and replay components
        reader = RecordingReader(session_dir)
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
        return False

    # Create world model for replay (no interactive mode, no wandb by default)
    logger.info("Initializing AdaptiveWorldModel for replay...")

    world_model = AdaptiveWorldModel(
        robot,
        interactive=False,  # No interactive mode in replay
        wandb_project="jetbot_developmental-movement-replay",  # Disable wandb for replay by default
        checkpoint_dir=checkpoint_dir,
        action_selector=action_selector,
        autoencoder_lr=None,
        predictor_lr=None
    )

    # Run the replay
    logger.info("Starting replay...")
    try:
        world_model.main_loop()
        logger.info("Replay completed successfully!")
        return True
    except KeyboardInterrupt:
        logger.info("Replay interrupted by user")
        return False
    except StopIteration:
        logger.info("Replay finished - end of recording reached")
        return True
    except Exception as e:
        logger.error(f"Replay failed: {e}")
        return False
    finally:
        # Save final checkpoint after replay
        world_model.save_checkpoint()

        # Cleanup
        robot.cleanup()


def get_all_session_dirs(base_dir):
    """Find all session directories in the base directory"""
    if not os.path.exists(base_dir):
        logger.error(f"Session base directory does not exist: {base_dir}")
        return []

    # Look for directories that match session naming pattern
    session_pattern = os.path.join(base_dir, "session_*")
    session_dirs = glob.glob(session_pattern)

    # Filter to only include directories that actually exist and have required files
    valid_sessions = []
    for session_dir in session_dirs:
        if os.path.isdir(session_dir):
            # Check if session has required files
            meta_file = os.path.join(session_dir, "session_meta.json")
            frames_dir = os.path.join(session_dir, "frames")
            if os.path.exists(meta_file) and os.path.exists(frames_dir):
                valid_sessions.append(session_dir)
            else:
                logger.warning(f"Skipping incomplete session: {session_dir}")

    # Sort by session name for consistent ordering
    valid_sessions.sort()
    return valid_sessions


def main():
    # Configuration
    SESSIONS_BASE_DIR = config.RECORDING_BASE_DIR
    CHECKPOINT_DIR = config.DEFAULT_CHECKPOINT_DIR  # Shared checkpoint directory

    logger.info("Starting robot session replay for all sessions...")
    logger.info(f"Scanning for sessions in: {SESSIONS_BASE_DIR}")

    # Get all session directories
    session_dirs = get_all_session_dirs(SESSIONS_BASE_DIR)

    if not session_dirs:
        logger.error("No valid session directories found")
        sys.exit(1)

    logger.info(f"Found {len(session_dirs)} valid sessions to replay")
    for i, session_dir in enumerate(session_dirs, 1):
        session_name = os.path.basename(session_dir)
        logger.info(f"  {i}. {session_name}")

    # Replay each session
    successful_replays = 0
    for i, session_dir in enumerate(session_dirs, 1):
        session_name = os.path.basename(session_dir)
        logger.info(f"\n{'='*60}")
        logger.info(f"REPLAYING SESSION {i}/{len(session_dirs)}: {session_name}")
        logger.info(f"{'='*60}")

        try:
            success = replay_single_session(session_dir, CHECKPOINT_DIR)
            if success:
                successful_replays += 1
                logger.info(f"✓ Session {session_name} completed successfully")
            else:
                logger.error(f"✗ Session {session_name} failed")
        except KeyboardInterrupt:
            logger.info(f"\nReplay interrupted by user during session {session_name}")
            break
        except Exception as e:
            logger.error(f"✗ Session {session_name} failed with error: {e}")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"REPLAY SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total sessions: {len(session_dirs)}")
    logger.info(f"Successful replays: {successful_replays}")
    logger.info(f"Failed replays: {len(session_dirs) - successful_replays}")
    logger.info("All session replays complete!")

if __name__ == "__main__":
    main()