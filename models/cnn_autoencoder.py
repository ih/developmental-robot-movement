"""
CNN-based autoencoder for visual encoding/decoding.

A convolutional alternative to the Vision Transformer autoencoder.
"""

import torch
import torch.nn as nn
from models.base_autoencoder import BaseAutoencoder


class CNNAutoencoder(BaseAutoencoder):
    """
    Convolutional Neural Network based Autoencoder.

    Uses standard conv layers for encoding and transposed conv layers for decoding.
    The spatial features from the encoder are flattened into a sequence of tokens
    for compatibility with sequence-based predictors.
    """

    def __init__(self, image_size=224, embed_dim=256, latent_spatial_size=7):
        """
        Initialize CNN autoencoder.

        Args:
            image_size: int, input image size (assumes square images)
            embed_dim: int, number of channels in the latent representation
            latent_spatial_size: int, spatial size of latent features (e.g., 7x7)
        """
        super().__init__()

        # Store dimensions for base class interface
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.latent_spatial_size = latent_spatial_size
        self.num_tokens = latent_spatial_size * latent_spatial_size  # Flattened spatial features

        # --------------------------------------------------------------------------
        # ENCODER (Convolutional downsampling)
        # --------------------------------------------------------------------------
        self.encoder = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 112x112x64 -> 56x56x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 56x56x128 -> 28x28x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 28x28x256 -> 14x14x256
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 14x14x256 -> 7x7x{embed_dim}
            nn.Conv2d(256, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------------------------------
        # DECODER (Transposed convolutional upsampling)
        # --------------------------------------------------------------------------
        self.decoder = nn.Sequential(
            # 7x7x{embed_dim} -> 14x14x256
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 14x14x256 -> 28x28x256
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 28x28x256 -> 56x56x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 56x56x128 -> 112x112x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 112x112x64 -> 224x224x3
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def encode(self, imgs):
        """
        Encode images to latent features.

        Args:
            imgs: (batch_size, 3, H, W) tensor

        Returns:
            latent_features: (batch_size, num_tokens, embed_dim) tensor
                           where num_tokens = latent_spatial_size^2
        """
        # Forward through encoder
        features = self.encoder(imgs)  # (B, embed_dim, H_lat, W_lat)

        # Reshape to sequence of tokens: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = features.shape
        latent_features = features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        return latent_features

    def decode_from_latent(self, latent_features):
        """
        Decode latent features back to images.

        Args:
            latent_features: (batch_size, num_tokens, embed_dim) tensor

        Returns:
            decoded_imgs: (batch_size, 3, H, W) tensor
        """
        B, num_tokens, C = latent_features.shape

        # Reshape from sequence to spatial: (B, H*W, C) -> (B, C, H, W)
        H = W = int(num_tokens ** 0.5)
        features = latent_features.permute(0, 2, 1).reshape(B, C, H, W)

        # Forward through decoder
        decoded_imgs = self.decoder(features)

        return decoded_imgs

    def forward(self, imgs, **kwargs):
        """
        Forward pass for training.

        Args:
            imgs: (batch_size, 3, H, W) tensor
            **kwargs: unused (for interface compatibility)

        Returns:
            reconstructed_imgs: (batch_size, 3, H, W) tensor
            latent_features: (batch_size, num_tokens, embed_dim) tensor
        """
        # Encode
        latent_features = self.encode(imgs)

        # Decode
        reconstructed_imgs = self.decode_from_latent(latent_features)

        return reconstructed_imgs, latent_features

    def reconstruct(self, imgs):
        """
        Full reconstruction: encode then decode.

        Args:
            imgs: (batch_size, 3, H, W) tensor

        Returns:
            reconstructed_imgs: (batch_size, 3, H, W) tensor
        """
        with torch.no_grad():
            reconstructed_imgs, _ = self.forward(imgs)
        return reconstructed_imgs

    def compute_reconstruction_loss(self, imgs, **kwargs):
        """
        Compute reconstruction loss for training.

        Args:
            imgs: (batch_size, 3, H, W) tensor
            **kwargs: unused (for interface compatibility)

        Returns:
            loss: scalar tensor
        """
        reconstructed_imgs, _ = self.forward(imgs)

        # MSE loss in pixel space
        loss = torch.nn.functional.mse_loss(reconstructed_imgs, imgs)

        return loss

    def train_step(self, imgs, optimizer, **kwargs):
        """
        Perform one training step on a batch of images.

        Args:
            imgs: (batch_size, 3, H, W) tensor of images
            optimizer: torch optimizer instance
            **kwargs: unused (for interface compatibility)

        Returns:
            loss_value: float, the loss value for this step
        """
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        reconstructed_imgs, _ = self.forward(imgs)

        # Calculate reconstruction loss
        loss = torch.nn.functional.mse_loss(reconstructed_imgs, imgs)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        return loss.item()
