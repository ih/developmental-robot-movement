"""
Toroidal White Dot Environment

A simple 224x224 simulated environment containing a single white dot on a black background.
The environment is toroidal (wraps around horizontally).
"""

import numpy as np
from typing import Tuple


class ToroidalDotEnvironment:
    """
    Simple toroidal environment with a white dot that can move horizontally.

    The environment is a 224x224 black image with a single white dot.
    Action 1 moves the dot right by a configurable number of pixels.
    Action 0 keeps the dot stationary.
    The horizontal axis wraps around (toroidal topology).
    """

    def __init__(self,
                 img_size: int = 224,
                 dot_radius: int = 2,
                 move_pixels: int = 7,
                 seed: int = None):
        """
        Initialize the toroidal dot environment.

        Args:
            img_size: Size of the square image (default 224)
            dot_radius: Radius of the white dot in pixels (default 2)
            move_pixels: Number of pixels to move right when action=1 (default 7)
            seed: Random seed for reproducibility (optional)
        """
        self.img_size = img_size
        self.dot_radius = dot_radius
        self.move_pixels = move_pixels

        # Set random seed if provided
        self.rng = np.random.RandomState(seed)

        # Dot position (x, y)
        self.dot_x = 0
        self.dot_y = 0

        # Initialize
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the environment with a new random position.

        Returns:
            Initial observation (224x224x3 RGB image)
        """
        # Randomize both x and y positions
        self.dot_x = self.rng.randint(0, self.img_size)
        self.dot_y = self.rng.randint(0, self.img_size)

        return self.render()

    def step(self, action: int) -> np.ndarray:
        """
        Execute one step in the environment.

        Args:
            action: 0 (no movement) or 1 (move right)

        Returns:
            Observation after action (224x224x3 RGB image)
        """
        if action == 1:
            # Move right with toroidal wrapping
            self.dot_x = (self.dot_x + self.move_pixels) % self.img_size
        # action == 0: no movement

        return self.render()

    def render(self) -> np.ndarray:
        """
        Render the current state as an RGB image.

        Returns:
            224x224x3 RGB numpy array with white dot on black background
        """
        # Create black image
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Draw white dot (circle)
        for dy in range(-self.dot_radius, self.dot_radius + 1):
            for dx in range(-self.dot_radius, self.dot_radius + 1):
                # Check if within circle radius
                if dx*dx + dy*dy <= self.dot_radius * self.dot_radius:
                    # Apply with toroidal wrapping
                    x = (self.dot_x + dx) % self.img_size
                    y = (self.dot_y + dy) % self.img_size
                    img[y, x] = [255, 255, 255]  # White

        return img

    def get_position(self) -> Tuple[int, int]:
        """
        Get current dot position.

        Returns:
            (x, y) tuple of current dot position
        """
        return (self.dot_x, self.dot_y)
