"""
Models package for the developmental robot movement system.
"""

# Base classes
from .base_autoencoder import BaseAutoencoder
from .base_predictor import BasePredictor

# Autoencoder implementations
from .vit_autoencoder import MaskedAutoencoderViT
from .cnn_autoencoder import CNNAutoencoder

# Predictor implementations
from .transformer_predictor import TransformerActionConditionedPredictor
from .lstm_predictor import LSTMActionConditionedPredictor

# Action classification
from .action_classifier import ActionClassifier

__all__ = [
    # Base classes
    'BaseAutoencoder',
    'BasePredictor',
    # Autoencoder implementations
    'MaskedAutoencoderViT',
    'CNNAutoencoder',
    # Predictor implementations
    'TransformerActionConditionedPredictor',
    'LSTMActionConditionedPredictor',
    # Action classification
    'ActionClassifier'
]