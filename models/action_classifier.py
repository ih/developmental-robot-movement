"""
Action Classifier module for classifying actions from predicted image differences.

This module provides a standalone action classification capability that can be
used with any predictor architecture. It takes pixel-level differences between
predicted and previous frames and classifies which action was taken.
"""

import torch
import torch.nn as nn


class ActionClassifier(nn.Module):
    """
    Classifies which action was taken based on pixel differences.

    Takes the difference between a predicted image and the previous image,
    and classifies which discrete action was taken to produce this change.
    This helps the predictor learn action-aware visual representations.
    """

    def __init__(self, num_actions):
        """
        Initialize action classifier.

        Args:
            num_actions: int, number of discrete actions to classify
        """
        super().__init__()

        self.num_actions = num_actions

        # Convolutional classifier that processes image differences
        # Input: (batch_size, 3, H, W) - RGB pixel differences
        # Output: (batch_size, num_actions) - action logits
        self.classifier = nn.Sequential(
            # Downsample and extract features from pixel differences
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Global pooling to get fixed-size representation
            nn.AdaptiveAvgPool2d(1),  # (B, 32, 1, 1)
            nn.Flatten(),             # (B, 32)

            # Classification head
            nn.Linear(32, num_actions)
        )

    def forward(self, pixel_diff):
        """
        Classify action from pixel differences.

        Args:
            pixel_diff: (batch_size, 3, H, W) tensor of pixel differences
                       between predicted and previous frames

        Returns:
            action_logits: (batch_size, num_actions) tensor of unnormalized scores
        """
        return self.classifier(pixel_diff)

    def compute_loss(self, pixel_diff, action_target):
        """
        Compute cross-entropy loss for action classification.

        Args:
            pixel_diff: (batch_size, 3, H, W) tensor of pixel differences
            action_target: (batch_size,) tensor of ground truth action indices

        Returns:
            loss: scalar tensor
        """
        logits = self.forward(pixel_diff)
        return torch.nn.functional.cross_entropy(logits, action_target)
