"""
Robot Runner - Execute Actions Without Learning

A simple runner that executes actions on a robot using an action selector,
without any learning or world model components. Useful for:
- Testing action selectors
- Collecting data with deterministic policies
- Running robots with pre-defined behavior patterns
- Debugging robot interfaces
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import signal
from typing import Optional, Callable, Tuple, List, Dict, Any


class RobotRunner:
    """
    Execute actions on a robot using an action selector without learning.

    This is a lightweight alternative to AdaptiveWorldModel that focuses
    on action execution without any neural network training or prediction.
    """

    def __init__(self,
                 robot_interface,
                 action_selector: Optional[Callable] = None,
                 interactive: bool = False,
                 display_interval: int = 10,
                 action_delay: float = 0.0):
        """
        Initialize the robot runner.

        Args:
            robot_interface: RobotInterface implementation
            action_selector: Function that takes current_observation and returns (action, metadata).
                           If None, selects actions randomly from action space.
            interactive: If True, prompt user before each action
            display_interval: Update display every N steps (0 to disable)
            action_delay: Delay in seconds after each action
        """
        self.robot = robot_interface
        self.interactive = interactive
        self.display_interval = display_interval
        self.action_delay = action_delay

        # Set default action selector if none provided
        if action_selector is None:
            self.action_selector = self._random_action_selector
        else:
            self.action_selector = action_selector

        # Tracking
        self.step_count = 0
        self.action_count = 0
        self.last_action_time = None
        self.action_time_intervals = []

        # Action statistics
        self.action_counts = {}
        for action in self.robot.action_space:
            action_key = self._action_to_key(action)
            self.action_counts[action_key] = 0

        # Display
        self.fig = None
        self.ax = None

        print(f"RobotRunner initialized")
        print(f"Action space: {self.robot.action_space}")
        print(f"Interactive mode: {interactive}")

    def _random_action_selector(self, current_observation: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], List[Any]]:
        """Default action selector that picks random actions."""
        action = np.random.choice(self.robot.action_space)
        return action, []

    def _action_to_key(self, action: Dict[str, Any]) -> str:
        """Convert action dict to a hashable key for counting."""
        return str(sorted(action.items()))

    def main_loop(self):
        """Main execution loop: observe -> select -> execute -> repeat."""
        print("\nStarting main loop...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Get current observation
                current_observation = self.robot.get_observation()
                if current_observation is None:
                    print("Warning: Failed to get observation, skipping step")
                    time.sleep(0.1)
                    continue

                # Select action
                selected_action, metadata = self.action_selector(current_observation)

                # Display current observation (periodically or always in interactive mode)
                should_display = (self.display_interval > 0 and
                                self.step_count % self.display_interval == 0) or self.interactive
                if should_display:
                    self.display_observation(current_observation, selected_action)

                # Interactive mode: get user confirmation
                if self.interactive:
                    action_to_execute = self.interactive_prompt(selected_action)
                else:
                    action_to_execute = selected_action

                # Track action timing
                current_time = time.time()
                if self.last_action_time is not None:
                    time_interval = current_time - self.last_action_time
                    self.action_time_intervals.append(time_interval)

                # Execute action
                self.robot.execute_action(action_to_execute)
                self.last_action_time = current_time

                # Update statistics
                action_key = self._action_to_key(action_to_execute)
                self.action_counts[action_key] = self.action_counts.get(action_key, 0) + 1
                self.action_count += 1
                self.step_count += 1

                # Print progress periodically
                if self.step_count % 100 == 0:
                    print(f"Step {self.step_count}, Actions: {self.action_count}")

                # Delay between actions
                if self.action_delay > 0:
                    time.sleep(self.action_delay)

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.cleanup()

    def display_observation(self, observation: np.ndarray, selected_action: Dict[str, Any]):
        """Display current observation and selected action."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
            plt.ion()
            plt.show(block=False)

        self.ax.clear()
        self.ax.imshow(observation)
        self.ax.set_title(f"Step {self.step_count}\nSelected Action: {selected_action}", fontsize=10)
        self.ax.axis('off')

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def interactive_prompt(self, selected_action: Dict[str, Any]) -> Dict[str, Any]:
        """Interactive mode: get user confirmation or override."""
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print(f"Step: {self.step_count}")
        print(f"Actions executed: {self.action_count}")
        print(f"Selected Action: {selected_action}")

        # Show action distribution
        print("\nAction Distribution:")
        for action_key, count in sorted(self.action_counts.items()):
            percentage = (count / self.action_count * 100) if self.action_count > 0 else 0
            print(f"  {action_key}: {count} ({percentage:.1f}%)")

        # Get user input
        while True:
            user_input = input("\nOptions:\n"
                             "1. Continue with selected action (press Enter)\n"
                             "2. Replace action (type new action as dict)\n"
                             "3. Stop (type 'stop')\n"
                             "Choice: ").strip()

            if user_input == "":
                return selected_action
            elif user_input.lower() == "stop":
                raise KeyboardInterrupt("Stopped by user")
            else:
                try:
                    new_action = eval(user_input)
                    if isinstance(new_action, dict):
                        print(f"Using custom action: {new_action}")
                        return new_action
                    else:
                        print("Invalid action format. Please use dict format.")
                except:
                    print("Invalid input. Please try again.")

    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        print(f"Total steps: {self.step_count}")
        print(f"Total actions: {self.action_count}")

        print("\nAction Distribution:")
        for action_key, count in sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.action_count * 100) if self.action_count > 0 else 0
            print(f"  {action_key}: {count} ({percentage:.1f}%)")

        if self.action_time_intervals:
            print(f"\nAction Timing:")
            print(f"  Mean interval: {np.mean(self.action_time_intervals):.3f}s")
            print(f"  Median interval: {np.median(self.action_time_intervals):.3f}s")

    def cleanup(self):
        """Cleanup resources."""
        print("\nCleaning up...")

        # Print summary
        self.print_summary()

        # Close display
        if self.fig is not None:
            plt.close(self.fig)

        # Cleanup robot
        self.robot.cleanup()

        print("Cleanup complete")


# Example usage
if __name__ == "__main__":
    from toroidal_dot_interface import ToroidalDotRobot
    from toroidal_action_selectors import create_sequence_action_selector, SEQUENCE_ALTERNATE

    print("Robot Runner Example")
    print("=" * 60)

    # Create robot
    robot = ToroidalDotRobot(
        img_size=224,
        dot_radius=5,
        move_pixels=27,
        action_delay=0.0
    )

    # Create action selector (alternating between stay and move)
    action_selector = create_sequence_action_selector(SEQUENCE_ALTERNATE)

    # Create runner
    runner = RobotRunner(
        robot_interface=robot,
        action_selector=action_selector,
        interactive=False,
        display_interval=10,
        action_delay=0.1
    )

    # Run
    try:
        runner.main_loop()
    except KeyboardInterrupt:
        print("\nStopped by user")
