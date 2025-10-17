#!/usr/bin/env python3
"""
Continuous replay training with plateau detection.
Runs replay sessions repeatedly until predictor loss plateaus or max epochs reached.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import glob

# Import necessary modules for direct execution
from autoencoder_latent_predictor_world_model import AutoencoderLatentPredictorWorldModel
from recording_reader import RecordingReader
from replay_robot import ReplayRobot
from recorded_policy import create_recorded_action_selector
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlateauDetector:
    """Detects when training has plateaued based on loss history."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        min_epochs: int = 3
    ):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            min_epochs: Minimum epochs before checking for plateau
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.loss_history: List[float] = []
        self.best_loss = float('inf')
        self.epochs_since_improvement = 0

    def update(self, loss: float) -> bool:
        """
        Update with new loss value.

        Returns:
            True if training has plateaued, False otherwise
        """
        self.loss_history.append(loss)

        # Don't check for plateau until minimum epochs
        if len(self.loss_history) < self.min_epochs:
            return False

        # Check if this is an improvement
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.epochs_since_improvement = 0
            logger.info(f"✓ New best loss: {loss:.6f}")
        else:
            self.epochs_since_improvement += 1
            logger.info(f"No improvement for {self.epochs_since_improvement}/{self.patience} epochs")

        # Check if plateaued
        if self.epochs_since_improvement >= self.patience:
            return True

        return False

    def get_stats(self) -> dict:
        """Get statistics about loss history."""
        if not self.loss_history:
            return {}

        recent_losses = self.loss_history[-min(5, len(self.loss_history)):]

        return {
            'current_loss': self.loss_history[-1],
            'best_loss': self.best_loss,
            'mean_recent_loss': np.mean(recent_losses),
            'std_recent_loss': np.std(recent_losses),
            'total_epochs': len(self.loss_history),
            'epochs_since_improvement': self.epochs_since_improvement
        }


def get_all_session_dirs(base_dir):
    """Find all session directories in the base directory"""
    if not os.path.exists(base_dir):
        return []

    session_pattern = os.path.join(base_dir, "session_*")
    session_dirs = glob.glob(session_pattern)

    # Filter to only include valid sessions
    valid_sessions = []
    for session_dir in session_dirs:
        if os.path.isdir(session_dir):
            meta_file = os.path.join(session_dir, "session_meta.json")
            frames_dir = os.path.join(session_dir, "frames")
            if os.path.exists(meta_file) and os.path.exists(frames_dir):
                valid_sessions.append(session_dir)

    valid_sessions.sort()
    return valid_sessions


def create_action_filter(filter_spec: Optional[str]) -> Optional[callable]:
    """Create action filter function from specification string."""
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
        return None


def extract_predictor_loss_from_checkpoint(checkpoint_dir: str) -> Optional[float]:
    """
    Extract the latest predictor training loss from checkpoint state.

    This reads the checkpoint to get the last logged loss value.
    You may need to modify this based on how you're tracking loss.
    """
    state_path = Path(checkpoint_dir) / 'state.pkl'

    if not state_path.exists():
        return None

    try:
        import pickle
        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        # Try to get loss from state if it's stored
        # If not available, we'll need to get it from wandb or logs
        return state.get('last_predictor_loss', None)
    except Exception as e:
        logger.warning(f"Could not extract loss from checkpoint: {e}")
        return None


def get_checkpoint_directory() -> str:
    """Get the checkpoint directory being used."""
    import config

    # Determine which checkpoint dir based on robot type
    # For now, assume toroidal dot
    return config.TOROIDAL_DOT_CHECKPOINT_DIR


def continuous_replay_with_plateau_detection(
    max_epochs: int = 100,
    patience: int = 5,
    min_delta: float = 0.001,
    min_epochs: int = 3,
    filter_action: Optional[str] = None
):
    """
    Run replay sessions continuously until loss plateaus.

    Args:
        max_epochs: Maximum number of epochs to run
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum loss improvement to count as progress
        min_epochs: Minimum epochs before checking for plateau
        filter_action: Optional action filter (e.g., "action=1")
    """
    detector = PlateauDetector(patience=patience, min_delta=min_delta, min_epochs=min_epochs)
    checkpoint_dir = get_checkpoint_directory()
    action_filter = create_action_filter(filter_action)

    # Find all sessions
    toroidal_dot_sessions = get_all_session_dirs(config.TOROIDAL_DOT_RECORDING_DIR)

    if not toroidal_dot_sessions:
        logger.error("No valid session directories found")
        return

    session_dir = toroidal_dot_sessions[0]  # Use first session
    session_name = os.path.basename(session_dir)

    logger.info("="*60)
    logger.info("CONTINUOUS REPLAY TRAINING WITH PLATEAU DETECTION")
    logger.info("="*60)
    logger.info(f"Session: {session_name}")
    logger.info(f"Max epochs: {max_epochs}")
    logger.info(f"Patience: {patience} epochs")
    logger.info(f"Min delta: {min_delta}")
    logger.info(f"Min epochs before plateau check: {min_epochs}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    if filter_action:
        logger.info(f"Action filter: {filter_action}")
    logger.info("="*60)

    # Create world model ONCE (will be reused across epochs)
    world_model = None
    epoch = 0
    last_predictor_loss = None

    try:
        while epoch < max_epochs:
            epoch += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch}/{max_epochs}")
            logger.info(f"{'='*60}\n")

            # Create new replay robot for this epoch
            reader = RecordingReader(session_dir)
            action_space = reader.get_action_space()
            robot = ReplayRobot(reader, action_space)
            action_selector = create_recorded_action_selector(reader, action_filter=action_filter)

            session_info = reader.get_session_info()
            robot_type = session_info.get('robot_type', 'unknown')

            logger.info(f"Replay setup: {reader.total_steps} steps")
            logger.info(f"Robot type: {robot_type}")

            # Set ACTION_CHANNELS and ACTION_RANGES based on robot type (before creating model)
            if "toroidal" in robot_type.lower():
                config.ACTION_CHANNELS = config.ToroidalDotConfig.ACTION_CHANNELS_DOT
                config.ACTION_RANGES = config.ToroidalDotConfig.ACTION_RANGES_DOT
                logger.info(f"Set ACTION_CHANNELS for ToroidalDotRobot: {config.ACTION_CHANNELS}")
            else:
                # Default to JetBot config (already set as defaults)
                logger.info(f"Using default JetBot ACTION_CHANNELS: {config.ACTION_CHANNELS}")

            # Create world model on first epoch, reuse thereafter
            if world_model is None:
                logger.info("Initializing AutoencoderLatentPredictorWorldModel (first epoch)...")
                wandb_project = "ToroidalDotRobot-developmental-movement-replay"

                world_model = AutoencoderLatentPredictorWorldModel(
                    robot,
                    interactive=False,
                    wandb_project=wandb_project,
                    checkpoint_dir=checkpoint_dir,
                    action_selector=action_selector,
                    autoencoder_lr=None,
                    predictor_lr=None
                )
            else:
                # Reuse existing world model but update robot
                logger.info(f"Reusing existing world model (epoch {epoch})...")
                world_model.robot = robot
                world_model.action_selector = action_selector

            # Run replay for this epoch
            try:
                world_model.main_loop()
                logger.info(f"Epoch {epoch} completed successfully")
            except StopIteration:
                logger.info(f"Epoch {epoch} finished - end of recording reached")
            except Exception as e:
                logger.error(f"Epoch {epoch} failed: {e}")
                robot.cleanup()
                break

            # Cleanup robot for this epoch
            robot.cleanup()

            # Save checkpoint after each epoch
            world_model.save_checkpoint()

            # Extract loss
            logger.info("\nExtracting predictor loss...")
            loss = extract_predictor_loss_from_checkpoint(checkpoint_dir)

            if loss is None:
                logger.warning("Could not extract predictor loss - using last known value")
                loss = last_predictor_loss if last_predictor_loss is not None else 1.0
            else:
                last_predictor_loss = loss

            # Check for plateau
            logger.info(f"Epoch {epoch} - Predictor Loss: {loss:.6f}")

            plateaued = detector.update(loss)

            # Print statistics
            stats = detector.get_stats()
            logger.info(f"Statistics: {json.dumps(stats, indent=2)}")

            if plateaued:
                logger.info("\n" + "="*60)
                logger.info("TRAINING PLATEAUED - STOPPING")
                logger.info("="*60)
                logger.info(f"Best loss: {detector.best_loss:.6f}")
                logger.info(f"Final loss: {loss:.6f}")
                logger.info(f"Total epochs: {epoch}")
                break

        else:
            logger.info("\n" + "="*60)
            logger.info("REACHED MAX EPOCHS - STOPPING")
            logger.info("="*60)
            logger.info(f"Total epochs: {epoch}")

    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("INTERRUPTED BY USER")
        logger.info("="*60)
        logger.info(f"Completed epochs: {epoch}")

    finally:
        # Print final summary
        stats = detector.get_stats()
        logger.info("\n" + "="*60)
        logger.info("FINAL TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total epochs completed: {len(detector.loss_history)}")
        if stats:
            logger.info(f"Best loss achieved: {stats['best_loss']:.6f}")
            logger.info(f"Final loss: {stats['current_loss']:.6f}")
            logger.info(f"Loss history: {[f'{l:.6f}' for l in detector.loss_history]}")
        logger.info("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Continuous replay training with plateau detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run until plateau with default settings
  python continuous_replay.py

  # Run with custom plateau detection parameters
  python continuous_replay.py --patience 10 --min-delta 0.0001

  # Run with action filtering
  python continuous_replay.py --filter-action action=1

  # Run with max epochs limit
  python continuous_replay.py --max-epochs 50
        """
    )

    parser.add_argument(
        '--max-epochs',
        type=int,
        default=100,
        help='Maximum number of epochs to run (default: 100)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of epochs with no improvement before stopping (default: 5)'
    )
    parser.add_argument(
        '--min-delta',
        type=float,
        default=0.001,
        help='Minimum loss improvement to count as progress (default: 0.001)'
    )
    parser.add_argument(
        '--min-epochs',
        type=int,
        default=3,
        help='Minimum epochs before checking for plateau (default: 3)'
    )
    parser.add_argument(
        '--filter-action',
        type=str,
        default=None,
        metavar='KEY=VALUE',
        help='Filter actions by key=value (e.g., action=1)'
    )

    args = parser.parse_args()

    continuous_replay_with_plateau_detection(
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        min_epochs=args.min_epochs,
        filter_action=args.filter_action
    )
