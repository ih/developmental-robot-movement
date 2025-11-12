"""
Models package for the developmental robot movement system.
"""

# Base classes
from .base_autoencoder import BaseAutoencoder

# Autoencoder implementations
from .vit_autoencoder import MaskedAutoencoderViT

# Canvas-based concat predictor utilities
from .autoencoder_concat_predictor import (
    build_canvas,
    TargetedMAEWrapper,
    canvas_to_tensor
)

__all__ = [
    # Base classes
    'BaseAutoencoder',
    # Autoencoder implementations
    'MaskedAutoencoderViT',
    # Canvas utilities
    'build_canvas',
    'TargetedMAEWrapper',
    'canvas_to_tensor'
]
