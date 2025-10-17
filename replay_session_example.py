#!/usr/bin/env python3
"""
Example script showing how to replay recorded robot sessions
using the AutoencoderLatentPredictorWorldModel with RecordingReader and ReplayRobot.
"""

from autoencoder_latent_predictor_world_model import AutoencoderLatentPredictorWorldModel
from recording_reader import RecordingReader
from replay_robot import ReplayRobot
from recorded_policy import create_recorded_action_selector
import config
import logging
import wandb
import sys
import os
import glob
import argparse
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress debug messages from 3rd party packages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def create_action_filter(filter_spec: Optional[str]) -> Optional[callable]:
    """Create action filter function from specification string.

    Args:
        filter_spec: Filter specification in format "key=value" (e.g., "motor_right=0.12")
                     or None to replay all actions

    Returns:
        Filter function that returns True if action should be replayed, or None for no filtering
    """
    if filter_spec is None:
        return None

    try:
        key, value_str = filter_spec.split('=', 1)
        key = key.strip()

        # Try to parse value as float, then int, otherwise keep as string
        try:
            value = float(value_str)
            if value == int(value):
                value = int(value)
        except ValueError:
            value = value_str.strip()

        def filter_fn(action: Dict[str, Any]) -> bool:
            return action.get(key) == value

        logger.info(f"Action filter created: {key}={value}")
        return filter_fn

    except ValueError:
        logger.error(f"Invalid filter specification: '{filter_spec}'. Expected format: key=value")
        sys.exit(1)


def replay_single_session(session_dir, action_filter=None):
    """Replay a single session directory with automatic robot type detection"""
    logger.info(f"Loading session from: {session_dir}")

    try:
        # Create recording reader and replay components
        reader = RecordingReader(session_dir)
        action_space = reader.get_action_space()
        robot = ReplayRobot(reader, action_space)
        action_selector = create_recorded_action_selector(reader, action_filter=action_filter)

        session_info = reader.get_session_info()
        robot_type = session_info.get('robot_type', 'unknown')

        logger.info(f"Replay setup complete: {reader.total_steps} steps loaded")
        logger.info(f"Robot type: {robot_type}")
        logger.info(f"Action space has {len(action_space)} actions")

        if action_filter is not None:
            logger.info("Action filtering enabled - only matching actions will be replayed")

        # Select checkpoint directory and wandb project based on robot type
        # Normalize robot type for comparison (lowercase, remove spaces/underscores)
        robot_type_normalized = robot_type.lower().replace('_', '').replace(' ', '')

        if 'jetbot' in robot_type_normalized:
            checkpoint_dir = config.JETBOT_CHECKPOINT_DIR
            wandb_project = "jetbot-developmental-movement-replay"
            logger.info(f"Using JetBot checkpoint directory: {checkpoint_dir}")
        elif 'toroidaldot' in robot_type_normalized:
            checkpoint_dir = config.TOROIDAL_DOT_CHECKPOINT_DIR
            wandb_project = "ToroidalDotRobot-developmental-movement-replay"
            logger.info(f"Using Toroidal Dot checkpoint directory: {checkpoint_dir}")
        else:
            # Fallback to default checkpoint directory for unknown robot types
            checkpoint_dir = config.DEFAULT_CHECKPOINT_DIR
            wandb_project = f"{robot_type}-developmental-movement-replay"
            logger.warning(f"Unknown robot type '{robot_type}', using default checkpoint directory: {checkpoint_dir}")

    except Exception as e:
        logger.error(f"Failed to load replay session: {e}")
        return False

    # Create world model for replay (no interactive mode, wandb enabled by default)
    logger.info("Initializing AutoencoderLatentPredictorWorldModel for replay...")

    world_model = AutoencoderLatentPredictorWorldModel(
        robot,
        interactive=False,  # No interactive mode in replay
        wandb_project=wandb_project,
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
        import traceback
        logger.error(f"Replay failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Replay recorded robot sessions with optional action filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay all actions from all robot types
  python replay_session_example.py

  # Only replay actions with motor_right=0.12 (JetBot)
  python replay_session_example.py --filter-action motor_right=0.12

  # Only replay move actions (Toroidal Dot)
  python replay_session_example.py --filter-action action=1
        """
    )
    parser.add_argument(
        '--filter-action',
        type=str,
        default=None,
        metavar='KEY=VALUE',
        help='Filter actions by key=value (e.g., motor_right=0.12 or action=1). If not specified, all actions are replayed.'
    )
    args = parser.parse_args()

    # Create action filter from command-line argument
    action_filter = create_action_filter(args.filter_action)

    logger.info("Starting robot session replay for all sessions...")

    # Scan both robot-specific directories for sessions
    all_session_dirs = []

    # Scan JetBot sessions
    # jetbot_sessions = get_all_session_dirs(config.JETBOT_RECORDING_DIR)
    # if jetbot_sessions:
    #     logger.info(f"Found {len(jetbot_sessions)} JetBot sessions in: {config.JETBOT_RECORDING_DIR}")
    #     all_session_dirs.extend(jetbot_sessions)

    # Scan Toroidal Dot sessions
    toroidal_dot_sessions = get_all_session_dirs(config.TOROIDAL_DOT_RECORDING_DIR)
    if toroidal_dot_sessions:
        logger.info(f"Found {len(toroidal_dot_sessions)} Toroidal Dot sessions in: {config.TOROIDAL_DOT_RECORDING_DIR}")
        all_session_dirs.extend(toroidal_dot_sessions)

    # Also scan legacy base directory for any sessions
    legacy_sessions = get_all_session_dirs(config.RECORDING_BASE_DIR)
    # Filter out sessions already found in robot-specific directories
    legacy_sessions = [s for s in legacy_sessions if s not in all_session_dirs]
    if legacy_sessions:
        logger.info(f"Found {len(legacy_sessions)} legacy sessions in: {config.RECORDING_BASE_DIR}")
        all_session_dirs.extend(legacy_sessions)

    if not all_session_dirs:
        logger.error("No valid session directories found in any location")
        sys.exit(1)

    logger.info(f"\nTotal: {len(all_session_dirs)} valid sessions to replay")
    for i, session_dir in enumerate(all_session_dirs, 1):
        session_name = os.path.basename(session_dir)
        parent_dir = os.path.basename(os.path.dirname(session_dir))
        logger.info(f"  {i}. [{parent_dir}] {session_name}")

    # Replay each session (checkpoint directory determined automatically per robot type)
    successful_replays = 0
    for i, session_dir in enumerate(all_session_dirs, 1):
        session_name = os.path.basename(session_dir)
        logger.info(f"\n{'='*60}")
        logger.info(f"REPLAYING SESSION {i}/{len(all_session_dirs)}: {session_name}")
        logger.info(f"{'='*60}")

        try:
            success = replay_single_session(session_dir, action_filter)
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
    logger.info(f"Total sessions: {len(all_session_dirs)}")
    logger.info(f"Successful replays: {successful_replays}")
    logger.info(f"Failed replays: {len(all_session_dirs) - successful_replays}")
    logger.info("All session replays complete!")

if __name__ == "__main__":
    main()