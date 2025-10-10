"""
Abstract base class for action-conditioned predictors.

Defines the interface that all predictor implementations must follow.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BasePredictor(ABC, nn.Module):
    """
    Abstract base class for action-conditioned predictors.

    All predictor implementations must inherit from this class and implement
    the required methods. This allows the adaptive world model to work with
    different predictor architectures (Transformer, LSTM, GRU, etc.).
    """

    def __init__(self, level=0, num_actions=2):
        """
        Initialize base predictor.

        Args:
            level: int, hierarchical level of this predictor
            num_actions: int, number of discrete actions in the action space
        """
        super().__init__()
        self.level = level
        self.max_lookahead = 10  # Default max lookahead
        self.num_actions = num_actions

    @abstractmethod
    def forward(self, encoder_features_history, actions, **kwargs):
        """
        Predict next latent features given history and actions.

        Args:
            encoder_features_history: list of (batch_size, num_tokens, embed_dim) tensors
                                     Each tensor represents encoded features from a frame
            actions: list of action dicts (length = len(encoder_features_history) - 1)
                    Each dict contains action parameters (e.g., motor_left, motor_right, duration)
            **kwargs: Additional architecture-specific arguments

        Returns:
            predicted_features: (batch_size, num_tokens, embed_dim) tensor
                              Predicted latent features for next frame
            Additional returns may vary by implementation (use kwargs for flexibility)
        """
        pass

    @abstractmethod
    def predict_uncertainty(self, encoder_features_history, actions, num_samples=10):
        """
        Predict uncertainty using dropout or ensemble methods.

        Args:
            encoder_features_history: list of latent feature tensors
            actions: list of action dicts
            num_samples: int, number of samples for uncertainty estimation

        Returns:
            mean_prediction: (batch_size, num_tokens, embed_dim) mean predicted features
            uncertainty: (batch_size,) tensor of uncertainty scores
        """
        pass

    def _normalize_action(self, action_dict, device):
        """
        Normalize action dict to [-1, 1] range.

        This is a helper method that can be overridden or used by subclasses.

        Args:
            action_dict: dict with action parameters
            device: torch.device

        Returns:
            normalized_action: (1, num_action_channels) tensor in [-1, 1]
        """
        import config
        values = []
        for key in config.ACTION_CHANNELS:
            value = float(action_dict.get(key, 0.0))
            min_val, max_val = config.ACTION_RANGES[key]
            value = max(min(value, max_val), min_val)
            if max_val == min_val:
                scaled = 0.0
            else:
                scaled = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
            values.append(scaled)
        return torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0)

    def get_embed_dim(self):
        """
        Get the embedding dimension expected by this predictor.

        Returns:
            embed_dim: int
        """
        return getattr(self, 'embed_dim', 256)
