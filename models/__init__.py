"""
Models package for the developmental robot movement system.
"""

# Base classes
from .base_autoencoder import BaseAutoencoder

# Autoencoder implementations
from .vit_autoencoder import MaskedAutoencoderViT
from .vit_decoder_only import DecoderOnlyViT

# Diffusion Transformer
from .vit_dit import DiffusionViT, DiTBlock, SinusoidalTimestepEmbedding
from .noise_scheduler import NoiseScheduler

# VAE/Encoder backends
from .vae import CanvasVAE, PretrainedSDVAE, PretrainedFluxVAE, DINOv2Encoder, create_vae

# Latent Diffusion wrapper
from .latent_diffusion import LatentDiffusionWrapper

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
    # Diffusion Transformer
    'DiffusionViT',
    'DiTBlock',
    'SinusoidalTimestepEmbedding',
    'NoiseScheduler',
    # VAE/Encoder backends
    'CanvasVAE',
    'PretrainedSDVAE',
    'PretrainedFluxVAE',
    'DINOv2Encoder',
    'create_vae',
    # Latent Diffusion wrapper
    'LatentDiffusionWrapper',
    # Canvas utilities
    'build_canvas',
    'TargetedMAEWrapper',
    'TargetedDecoderOnlyWrapper',
    'canvas_to_tensor',
]
