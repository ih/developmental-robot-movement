"""
Models package for the developmental robot movement system.
"""

# Base classes
from .base_autoencoder import BaseAutoencoder

# Autoencoder implementations
from .vit_autoencoder import MaskedAutoencoderViT
from .vit_decoder_only import DecoderOnlyViT

# Canvas-based concat predictor utilities
from .autoencoder_concat_predictor import (
    build_canvas,
    TargetedMAEWrapper,
    TargetedDecoderOnlyWrapper,
    canvas_to_tensor
)

__all__ = [
    # Base classes
    'BaseAutoencoder',
    # Autoencoder implementations
    'MaskedAutoencoderViT',
    'DecoderOnlyViT',
    # Canvas utilities
    'build_canvas',
    'TargetedMAEWrapper',
    'TargetedDecoderOnlyWrapper',
    'canvas_to_tensor'
]
