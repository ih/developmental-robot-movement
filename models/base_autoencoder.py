"""
Abstract base class for autoencoders.

Defines the interface that all autoencoder implementations must follow.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseAutoencoder(ABC, nn.Module):
    """
    Abstract base class for autoencoder architectures.

    All autoencoder implementations must inherit from this class and implement
    the required methods. This allows the adaptive world model to work with
    different autoencoder architectures (ViT, CNN, etc.).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, imgs):
        """
        Encode images to latent features.

        Args:
            imgs: (batch_size, 3, H, W) tensor of images

        Returns:
            latent_features: (batch_size, num_tokens, embed_dim) tensor of latent features
                           For ViT: num_tokens = num_patches + 1 (CLS token)
                           For CNN: num_tokens can be flattened spatial features
        """
        pass

    @abstractmethod
    def decode_from_latent(self, latent_features):
        """
        Decode latent features back to images.

        Args:
            latent_features: (batch_size, num_tokens, embed_dim) tensor

        Returns:
            decoded_imgs: (batch_size, 3, H, W) tensor of reconstructed images
        """
        pass

    @abstractmethod
    def reconstruct(self, imgs):
        """
        Full reconstruction: encode then decode without masking.

        Args:
            imgs: (batch_size, 3, H, W) tensor of images

        Returns:
            reconstructed_imgs: (batch_size, 3, H, W) tensor of reconstructed images
        """
        pass

    @abstractmethod
    def forward(self, imgs, **kwargs):
        """
        Forward pass for training.

        Args:
            imgs: (batch_size, 3, H, W) tensor of images
            **kwargs: Architecture-specific arguments (e.g., mask_ratio for MAE)

        Returns:
            Architecture-specific outputs (e.g., reconstructed images, latent features)
        """
        pass

    @abstractmethod
    def compute_reconstruction_loss(self, imgs, **kwargs):
        """
        Compute reconstruction loss for training.

        Args:
            imgs: (batch_size, 3, H, W) tensor of images
            **kwargs: Architecture-specific arguments (e.g., mask_ratio for MAE)

        Returns:
            loss: scalar tensor
        """
        pass

    @abstractmethod
    def train_step(self, imgs, optimizer, **kwargs):
        """
        Perform one training step on a batch of images.

        Args:
            imgs: (batch_size, 3, H, W) tensor of images
            optimizer: torch optimizer instance
            **kwargs: Architecture-specific training arguments

        Returns:
            loss_value: float, the loss value for this step
        """
        pass

    def get_latent_dim(self):
        """
        Get the dimension of latent features.

        Returns:
            embed_dim: int, dimension of each latent token
        """
        # Default implementation - subclasses should override if needed
        return getattr(self, 'embed_dim', 256)

    def get_num_tokens(self, img_size=224):
        """
        Get the number of tokens in the latent representation.

        Args:
            img_size: int, size of input image (assumes square)

        Returns:
            num_tokens: int, number of latent tokens
        """
        # Default implementation - subclasses should override if needed
        return getattr(self, 'num_tokens', 197)  # 196 patches + 1 CLS for ViT
