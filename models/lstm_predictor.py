"""
LSTM-based action-conditioned predictor.

A recurrent alternative to the Transformer predictor.
"""

import torch
import torch.nn as nn
from models.base_predictor import BasePredictor
import config


class LSTMActionConditionedPredictor(BasePredictor):
    """
    LSTM-based predictor that processes sequences of latent features and actions
    to predict future latent features.

    The action is concatenated with the latent features at each timestep,
    providing action conditioning throughout the recurrent processing.
    """

    def __init__(
        self,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        level=0,
        num_actions=2,
    ):
        """
        Initialize LSTM predictor.

        Args:
            embed_dim: int, dimension of latent features from autoencoder
            hidden_dim: int, dimension of LSTM hidden state
            num_layers: int, number of LSTM layers
            dropout: float, dropout probability
            level: int, hierarchical level
            num_actions: int, number of discrete actions
        """
        super().__init__(level=level, num_actions=num_actions)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Action embedding: convert normalized action vector to learned representation
        num_action_channels = len(config.ACTION_CHANNELS)
        self.action_embedding = nn.Sequential(
            nn.Linear(num_action_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # LSTM processes concatenated [features, action_embedding] at each timestep
        # Input: (embed_dim + 64) from flattened features + action embedding
        # Note: We'll flatten spatial features for LSTM processing
        self.lstm = nn.LSTM(
            input_size=embed_dim + 64,  # Features + action embedding
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection: LSTM hidden state -> predicted features
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Dropout for uncertainty estimation
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features_history, actions, **kwargs):
        """
        Predict next latent features given history and actions.

        Args:
            encoder_features_history: list of (batch_size, num_tokens, embed_dim) tensors
            actions: list of action dicts
            **kwargs: unused (for interface compatibility)

        Returns:
            predicted_features: (batch_size, num_tokens, embed_dim) tensor
        """
        # Handle empty history case
        if not encoder_features_history:
            batch_size = 1
            num_tokens = 49  # Default for 7x7 CNN latent
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.randn(batch_size, num_tokens, self.embed_dim, device=device)

        device = encoder_features_history[0].device
        batch_size = encoder_features_history[0].shape[0]
        num_tokens = encoder_features_history[0].shape[1]

        # Build sequence: for each frame except the last, pair it with the following action
        # This creates input-output pairs: (frame_t, action_t) -> frame_{t+1}
        sequence_inputs = []

        for i in range(len(encoder_features_history) - 1):
            # Get current frame features (batch_size, num_tokens, embed_dim)
            frame_features = encoder_features_history[i]

            # Pool features across tokens (average pooling)
            # (batch_size, num_tokens, embed_dim) -> (batch_size, embed_dim)
            pooled_features = frame_features.mean(dim=1)

            # Normalize and embed action
            action_normalized = self._normalize_action(actions[i], device)
            action_embedded = self.action_embedding(action_normalized)  # (batch_size, 64)

            # Concatenate features and action
            combined = torch.cat([pooled_features, action_embedded], dim=1)  # (batch_size, embed_dim + 64)
            sequence_inputs.append(combined)

        # Stack into sequence: (batch_size, seq_len, embed_dim + 64)
        if sequence_inputs:
            sequence_tensor = torch.stack(sequence_inputs, dim=1)
        else:
            # If no history with actions, use just the last frame
            last_frame = encoder_features_history[-1]
            pooled_last = last_frame.mean(dim=1)  # (batch_size, embed_dim)
            zero_action = torch.zeros(batch_size, 64, device=device)
            combined_last = torch.cat([pooled_last, zero_action], dim=1)
            sequence_tensor = combined_last.unsqueeze(1)  # (batch_size, 1, embed_dim + 64)

        # Forward through LSTM
        lstm_output, (hidden, cell) = self.lstm(sequence_tensor)

        # Use the last hidden state to predict next frame
        last_hidden = lstm_output[:, -1, :]  # (batch_size, hidden_dim)

        # Project to predicted features
        predicted_flat = self.output_projection(last_hidden)  # (batch_size, embed_dim)

        # Expand to spatial tokens
        # Broadcast the same prediction to all tokens (simple approach)
        predicted_features = predicted_flat.unsqueeze(1).expand(batch_size, num_tokens, self.embed_dim)

        return predicted_features

    def predict_uncertainty(self, encoder_features_history, actions, num_samples=10):
        """
        Predict uncertainty using Monte Carlo dropout.

        Args:
            encoder_features_history: list of latent feature tensors
            actions: list of action dicts
            num_samples: int, number of samples for uncertainty estimation

        Returns:
            mean_prediction: (batch_size, num_tokens, embed_dim) mean predicted features
            uncertainty: (batch_size,) tensor of uncertainty scores
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(encoder_features_history, actions)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch_size, num_tokens, embed_dim)

        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1).mean(dim=-1)  # Average variance

        self.eval()  # Disable dropout

        return mean_prediction, uncertainty
