"""
Record Toroidal Dot Sessions with Random Duration Action Selector

This script records toroidal dot sessions with a random duration action selector
that randomly chooses actions and maintains them for random consecutive steps.

Creates both training sessions (longer) and validation sessions (shorter) for
use with the concat world model explorer gradio app.
"""

import argparse
import os
import time
from datetime import datetime
from typing import Dict, List
import config
from toroidal_dot_interface import ToroidalDotRobot
from recording_writer import RecordingWriter
from recording_robot import RecordingRobot
from toroidal_action_selectors import create_random_duration_action_selector
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Record toroidal dot sessions with random duration action selector'
    )

    # Y-position control (mutually exclusive)
    y_group = parser.add_mutually_exclusive_group(required=True)
    y_group.add_argument(
        '--fixed-y',
        type=int,
        metavar='Y',
        help='Fixed y-position for dot (0-223)'
    )
    y_group.add_argument(
        '--random-y',
        action='store_true',
        help='Use random y-position for each session'
    )

    # X-position control (separate for train/val)
    parser.add_argument(
        '--train-initial-x',
        type=int,
        metavar='X',
        default=None,
        help='Fixed x-position for training sessions (0-223, default: random)'
    )
    parser.add_argument(
        '--val-initial-x',
        type=int,
        metavar='X',
        default=None,
        help='Fixed x-position for validation sessions (0-223, default: random)'
    )

    # Action selector parameters
    parser.add_argument(
        '--min-duration',
        type=int,
        default=1,
        help='Minimum action duration in steps (default: 1)'
    )
    parser.add_argument(
        '--max-duration',
        type=int,
        default=1,
        help='Maximum action duration in steps (default: 3)'
    )

    # Session parameters
    parser.add_argument(
        '--train-steps',
        type=int,
        default=1000,
        help='Steps per training session (default: 1000)'
    )
    parser.add_argument(
        '--val-steps',
        type=int,
        default=300,
        help='Steps per validation session (default: 100)'
    )
    parser.add_argument(
        '--num-train-sessions',
        type=int,
        default=1,
        help='Number of training sessions (default: 1)'
    )
    parser.add_argument(
        '--num-val-sessions',
        type=int,
        default=1,
        help='Number of validation sessions (default: 1)'
    )

    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )

    # Periodic reinitialization
    parser.add_argument(
        '--reinit-interval',
        type=int,
        default=None,
        metavar='N',
        help='Reinitialize dot position every N steps (disabled by default)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.fixed_y is not None and (args.fixed_y < 0 or args.fixed_y > 223):
        parser.error('--fixed-y must be in range [0, 223]')

    if args.train_initial_x is not None and (args.train_initial_x < 0 or args.train_initial_x > 223):
        parser.error('--train-initial-x must be in range [0, 223]')

    if args.val_initial_x is not None and (args.val_initial_x < 0 or args.val_initial_x > 223):
        parser.error('--val-initial-x must be in range [0, 223]')

    if args.min_duration < 1:
        parser.error('--min-duration must be at least 1')

    if args.max_duration < args.min_duration:
        parser.error('--max-duration must be >= --min-duration')

    if args.train_steps < 1:
        parser.error('--train-steps must be at least 1')

    if args.val_steps < 1:
        parser.error('--val-steps must be at least 1')

    if args.num_train_sessions < 0:
        parser.error('--num-train-sessions must be non-negative')

    if args.num_val_sessions < 0:
        parser.error('--num-val-sessions must be non-negative')

    if args.reinit_interval is not None and args.reinit_interval < 1:
        parser.error('--reinit-interval must be at least 1')

    return args


def record_session(
    session_name: str,
    session_type: str,
    num_steps: int,
    fixed_x: int,
    fixed_y: int,
    min_duration: int,
    max_duration: int,
    session_seed: int,
    action_seed: int,
    desc: str,
    reinit_interval: int = None
) -> Dict:
    """
    Record a single session.

    Args:
        session_name: Name for the session
        session_type: 'train' or 'val'
        num_steps: Number of steps to record
        fixed_x: Fixed x position (or None for random)
        fixed_y: Fixed y position (or None for random)
        min_duration: Minimum action duration
        max_duration: Maximum action duration
        session_seed: Seed for robot/environment
        action_seed: Seed for action selector
        desc: Description for progress bar
        reinit_interval: Reinitialize dot position every N steps (or None to disable)

    Returns:
        Dictionary with session statistics
    """
    # Create robot
    robot = ToroidalDotRobot(initial_x=fixed_x, initial_y=fixed_y, seed=session_seed)

    # Create recording writer
    writer = RecordingWriter(
        base_dir=config.TOROIDAL_DOT_RECORDING_DIR,
        session_name=session_name
    )

    # Wrap with recording robot
    recording_robot = RecordingRobot(robot, writer)

    # Create action selector
    selector = create_random_duration_action_selector(
        min_duration=min_duration,
        max_duration=max_duration,
        seed=action_seed
    )

    # Track statistics
    action_counts = {0: 0, 1: 0}
    action_switches = 0
    last_action = None

    # Execute steps
    start_time = time.time()
    for step in tqdm(range(num_steps), desc=desc):
        # Periodic reinitialization (before getting observation)
        if reinit_interval is not None and step > 0 and step % reinit_interval == 0:
            recording_robot.robot.reset()

        # Get observation
        obs = recording_robot.get_observation()

        # Select action
        action, _ = selector()
        action_value = action['action']

        # Track statistics
        action_counts[action_value] += 1
        if last_action is not None and last_action != action_value:
            action_switches += 1
        last_action = action_value

        # Execute action
        recording_robot.execute_action(action)

    # Cleanup
    recording_robot.cleanup()

    elapsed_time = time.time() - start_time

    return {
        'session_name': session_name,
        'session_type': session_type,
        'num_steps': num_steps,
        'action_counts': action_counts,
        'action_switches': action_switches,
        'elapsed_time': elapsed_time
    }


def print_session_stats(stats: Dict):
    """Print statistics for a single session."""
    print(f"\n  Session: {stats['session_name']}")
    print(f"  Steps: {stats['num_steps']}")
    print(f"  Actions: {stats['action_counts'][0]} stays, {stats['action_counts'][1]} moves")
    print(f"  Action switches: {stats['action_switches']}")
    print(f"  Time: {stats['elapsed_time']:.1f}s ({stats['num_steps']/stats['elapsed_time']:.1f} steps/s)")


def print_summary_stats(train_stats: List[Dict], val_stats: List[Dict]):
    """Print summary statistics for all sessions."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Training sessions
    if train_stats:
        total_train_steps = sum(s['num_steps'] for s in train_stats)
        total_train_time = sum(s['elapsed_time'] for s in train_stats)
        total_train_stays = sum(s['action_counts'][0] for s in train_stats)
        total_train_moves = sum(s['action_counts'][1] for s in train_stats)
        total_train_switches = sum(s['action_switches'] for s in train_stats)

        print(f"\nTraining Sessions: {len(train_stats)}")
        print(f"  Total steps: {total_train_steps}")
        print(f"  Total actions: {total_train_stays} stays, {total_train_moves} moves")
        print(f"  Total switches: {total_train_switches}")
        print(f"  Avg switches/session: {total_train_switches/len(train_stats):.1f}")
        print(f"  Total time: {total_train_time:.1f}s ({total_train_steps/total_train_time:.1f} steps/s)")

    # Validation sessions
    if val_stats:
        total_val_steps = sum(s['num_steps'] for s in val_stats)
        total_val_time = sum(s['elapsed_time'] for s in val_stats)
        total_val_stays = sum(s['action_counts'][0] for s in val_stats)
        total_val_moves = sum(s['action_counts'][1] for s in val_stats)
        total_val_switches = sum(s['action_switches'] for s in val_stats)

        print(f"\nValidation Sessions: {len(val_stats)}")
        print(f"  Total steps: {total_val_steps}")
        print(f"  Total actions: {total_val_stays} stays, {total_val_moves} moves")
        print(f"  Total switches: {total_val_switches}")
        print(f"  Avg switches/session: {total_val_switches/len(val_stats):.1f}")
        print(f"  Total time: {total_val_time:.1f}s ({total_val_steps/total_val_time:.1f} steps/s)")

    # Overall
    if train_stats or val_stats:
        grand_total_steps = sum(s['num_steps'] for s in train_stats + val_stats)
        grand_total_time = sum(s['elapsed_time'] for s in train_stats + val_stats)

        print(f"\nOverall:")
        print(f"  Total sessions: {len(train_stats) + len(val_stats)}")
        print(f"  Total steps: {grand_total_steps}")
        print(f"  Total time: {grand_total_time:.1f}s ({grand_total_steps/grand_total_time:.1f} steps/s)")

    print("\n" + "="*70)


def main():
    """Main recording function."""
    args = parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine y-mode string for naming
    if args.fixed_y is not None:
        y_mode = f"fixed_y{args.fixed_y}"
        fixed_y = args.fixed_y
    else:
        y_mode = "random_y"
        fixed_y = None

    # Determine x-mode strings for naming
    if args.train_initial_x is not None:
        train_x_mode = f"trainx{args.train_initial_x}"
        train_x = args.train_initial_x
    else:
        train_x_mode = "trainx_random"
        train_x = None

    if args.val_initial_x is not None:
        val_x_mode = f"valx{args.val_initial_x}"
        val_x = args.val_initial_x
    else:
        val_x_mode = "valx_random"
        val_x = None

    print("="*70)
    print("TOROIDAL DOT RECORDING WITH RANDOM DURATION ACTION SELECTOR")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Y-position: {y_mode}")
    print(f"  X-position: Training={train_x_mode}, Validation={val_x_mode}")
    print(f"  Action duration: {args.min_duration}-{args.max_duration} steps")
    print(f"  Reinit interval: {f'every {args.reinit_interval} steps' if args.reinit_interval is not None else 'disabled'}")
    print(f"  Training: {args.num_train_sessions} sessions × {args.train_steps} steps = {args.num_train_sessions * args.train_steps} total steps")
    print(f"  Validation: {args.num_val_sessions} sessions × {args.val_steps} steps = {args.num_val_sessions * args.val_steps} total steps")
    print(f"  Random seed: {args.seed if args.seed is not None else 'None (random)'}")
    print(f"  Output directory: {config.TOROIDAL_DOT_RECORDING_DIR}")
    print()

    # Track statistics
    train_stats = []
    val_stats = []

    # Record training sessions
    if args.num_train_sessions > 0:
        print("\n" + "="*70)
        print("RECORDING TRAINING SESSIONS")
        print("="*70)

        for session_idx in range(args.num_train_sessions):
            session_name = f"session_random_duration_{y_mode}_{train_x_mode}_train_{args.train_steps}steps_{session_idx+1:02d}_{timestamp}"

            # Generate seeds for this session
            if args.seed is not None:
                session_seed = args.seed + session_idx * 1000
                action_seed = args.seed + session_idx * 1000 + 1
            else:
                session_seed = None
                action_seed = None

            desc = f"Train {session_idx+1}/{args.num_train_sessions}"

            stats = record_session(
                session_name=session_name,
                session_type='train',
                num_steps=args.train_steps,
                fixed_x=train_x,
                fixed_y=fixed_y,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                session_seed=session_seed,
                action_seed=action_seed,
                desc=desc,
                reinit_interval=args.reinit_interval
            )

            train_stats.append(stats)
            print_session_stats(stats)

    # Record validation sessions
    if args.num_val_sessions > 0:
        print("\n" + "="*70)
        print("RECORDING VALIDATION SESSIONS")
        print("="*70)

        for session_idx in range(args.num_val_sessions):
            session_name = f"session_random_duration_{y_mode}_{val_x_mode}_val_{args.val_steps}steps_{session_idx+1:02d}_{timestamp}"

            # Generate seeds for this session
            if args.seed is not None:
                # Offset validation seeds to avoid overlap with training
                session_seed = args.seed + 100000 + session_idx * 1000
                action_seed = args.seed + 100000 + session_idx * 1000 + 1
            else:
                session_seed = None
                action_seed = None

            desc = f"Val {session_idx+1}/{args.num_val_sessions}"

            stats = record_session(
                session_name=session_name,
                session_type='val',
                num_steps=args.val_steps,
                fixed_x=val_x,
                fixed_y=fixed_y,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                session_seed=session_seed,
                action_seed=action_seed,
                desc=desc,
                reinit_interval=args.reinit_interval
            )

            val_stats.append(stats)
            print_session_stats(stats)

    # Print summary
    print_summary_stats(train_stats, val_stats)

    print("\n✓ Recording complete!")
    print(f"  Sessions saved to: {config.TOROIDAL_DOT_RECORDING_DIR}")
    print(f"  Training sessions: {len(train_stats)}")
    print(f"  Validation sessions: {len(val_stats)}")
    print()


if __name__ == '__main__':
    main()
