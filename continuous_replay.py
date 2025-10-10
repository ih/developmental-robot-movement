#!/usr/bin/env python3
"""
Continuous replay training with plateau detection.
Runs replay sessions repeatedly until predictor loss plateaus or max epochs reached.
"""

import subprocess
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

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


def extract_loss_from_wandb(checkpoint_dir: str) -> Optional[float]:
    """
    Extract average predictor loss from recent wandb logs.

    This is more reliable if you're using wandb.
    """
    try:
        import wandb

        # Get the most recent run
        api = wandb.Api()

        # You'll need to adjust this to your wandb project
        project = "ToroidalDotRobot-developmental-movement-replay"
        runs = api.runs(f"irvin-hwang-simulacra-systems/{project}")

        if not runs:
            return None

        latest_run = runs[0]
        history = latest_run.history(keys=["predictor_training_loss"], samples=100)

        if history.empty or 'predictor_training_loss' not in history.columns:
            return None

        # Get mean of recent losses (last 20 or all if less)
        recent_losses = history['predictor_training_loss'].dropna().tail(20)
        if len(recent_losses) > 0:
            return float(recent_losses.mean())

        return None
    except Exception as e:
        logger.warning(f"Could not extract loss from wandb: {e}")
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

    logger.info("="*60)
    logger.info("CONTINUOUS REPLAY TRAINING WITH PLATEAU DETECTION")
    logger.info("="*60)
    logger.info(f"Max epochs: {max_epochs}")
    logger.info(f"Patience: {patience} epochs")
    logger.info(f"Min delta: {min_delta}")
    logger.info(f"Min epochs before plateau check: {min_epochs}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    if filter_action:
        logger.info(f"Action filter: {filter_action}")
    logger.info("="*60)

    epoch = 0

    try:
        while epoch < max_epochs:
            epoch += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch}/{max_epochs}")
            logger.info(f"{'='*60}\n")

            # Build command
            cmd = [sys.executable, "replay_session_example.py"]
            if filter_action:
                cmd.extend(["--filter-action", filter_action])

            # Run replay
            result = subprocess.run(cmd)

            if result.returncode != 0:
                logger.error(f"Replay failed in epoch {epoch}")
                break

            # Extract loss from checkpoint or wandb
            logger.info("\nExtracting predictor loss...")

            # Try wandb first (more reliable)
            loss = extract_loss_from_wandb(checkpoint_dir)

            # Fallback to checkpoint if wandb not available
            if loss is None:
                loss = extract_predictor_loss_from_checkpoint(checkpoint_dir)

            if loss is None:
                logger.warning("Could not extract predictor loss - continuing anyway")
                continue

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
