"""
Masked Autoencoder Vision Transformer for visual encoding/decoding.
"""

import torch
import torch.nn as nn
import timm
from models.base_autoencoder import BaseAutoencoder


class MaskedAutoencoderViT(BaseAutoencoder):
    """
    A Vision Transformer (ViT) based Masked Autoencoder (MAE)
    with a POWERFUL encoder and a LIGHTWEIGHT MLP decoder.
    """
    def __init__(self, img_height=224, img_width=224, patch_size=16, embed_dim=256,
                 decoder_embed_dim=128, depth=5, num_heads=4, mlp_ratio=4.):
        super().__init__()

        # Store dimensions for base class interface
        self.embed_dim = embed_dim
        self.image_size = (img_height, img_width)
        self.patch_size = patch_size

        # --------------------------------------------------------------------------
        # MAE ENCODER (Powerful Transformer)
        # This part remains the same as before.
        self.patch_embed = timm.models.vision_transformer.PatchEmbed((img_height, img_width), patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_tokens = num_patches + 1  # +1 for CLS token
        self.grid_size = (img_height // patch_size, img_width // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE DECODER (Lightweight MLP) - NEW AND SIMPLER!
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        # A simple MLP head is enough for the decoder
        self.decoder_pred = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.ReLU(),
            nn.Linear(decoder_embed_dim, patch_size**2 * 3) # Predict the RGB values for each patch
        )
        # --------------------------------------------------------------------------

        # Initialize fixed 2D sin-cos positional embeddings for encoder/decoder
        self._init_positional_embeddings()

    # ----------------------------------------------------------------------
    # Positional embedding utilities (fixed 2D sin-cos, as in MAE)
    # ----------------------------------------------------------------------
    def _init_positional_embeddings(self):
        device = self.pos_embed.device
        # Encoder
        enc_pos = self._build_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size, device=device)
        # Decoder
        dec_pos = self._build_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, device=device)
        with torch.no_grad():
            self.pos_embed.copy_(enc_pos)
            self.decoder_pos_embed.copy_(dec_pos)

    def _build_2d_sincos_pos_embed(self, embed_dim: int, grid_size: tuple, device=None):
        """
        Create 2D sin-cos positional embeddings with a [CLS] token at the start.
        Returns a tensor of shape [1, grid_h*grid_w + 1, embed_dim].

        Args:
            embed_dim: Embedding dimension
            grid_size: Tuple of (grid_h, grid_w)
            device: torch device
        """
        assert embed_dim % 2 == 0, "embed_dim must be even for 2D sin-cos embeddings"

        grid_h_size, grid_w_size = grid_size

        # Positions along each axis
        grid_h = torch.arange(grid_h_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(grid_w_size, dtype=torch.float32, device=device)
        # Normalize to [0, 1]
        grid_h = grid_h / (grid_h_size - 1 if grid_h_size > 1 else 1)
        grid_w = grid_w / (grid_w_size - 1 if grid_w_size > 1 else 1)
        # Meshgrid: [H, W]
        gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij') if hasattr(torch, 'meshgrid') else torch.meshgrid(grid_h, grid_w)
        # Flatten to [N]
        gh = gh.reshape(-1)
        gw = gw.reshape(-1)
        # Half-dim for each axis
        dim_half = embed_dim // 2
        pos_h = self._get_1d_sincos_pos_embed(dim_half, gh)  # [N, dim_half]
        pos_w = self._get_1d_sincos_pos_embed(dim_half, gw)  # [N, dim_half]
        pos = torch.cat([pos_h, pos_w], dim=1)  # [N, embed_dim]
        # Add cls token pos (zeros) at the beginning
        cls_pos = torch.zeros(1, 1, embed_dim, dtype=pos.dtype, device=pos.device)
        pos = pos.unsqueeze(0)  # [1, N, D]
        pos = torch.cat([cls_pos, pos], dim=1)  # [1, N+1, D]
        return pos

    def _get_1d_sincos_pos_embed(self, embed_dim: int, positions: torch.Tensor):
        """Generate 1D sin-cos positional embeddings for given positions [N]."""
        assert embed_dim % 2 == 0, "1D embed_dim should be even"
        # Compute frequencies
        dim_t = torch.arange(embed_dim // 2, dtype=torch.float32, device=positions.device)
        dim_t = 1.0 / (10000 ** (dim_t / (embed_dim // 2)))  # [D/2]
        # Outer product: [N, D/2]
        out = positions.unsqueeze(1) * dim_t.unsqueeze(0)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [N, D]
        return emb

    def forward_encoder(self, x, mask_ratio=0.75):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        B, L, D = x.shape
        
        # Skip masking/shuffling when mask_ratio=0 for consistent ordering
        if mask_ratio == 0.0:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            # Create identity ids_restore (no shuffling occurred)
            ids_restore = torch.arange(L, device=x.device).unsqueeze(0).repeat(B, 1)
        else:
            len_keep = int(L * (1 - mask_ratio))
            noise = torch.rand(B, L, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x_masked), dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, ids_restore

    def forward_decoder(self, x, ids_restore):
        # (This function is now much simpler)
        x = self.decoder_embed(x)
        B, N, C = x.shape
        L = ids_restore.shape[1]
        len_keep = N - 1
        mask_tokens = self.mask_token.repeat(B, L - len_keep, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        
        # Pass through our simple MLP head
        x = self.decoder_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward(self, imgs, mask_ratio=0.75):
        latent, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, latent

    def encode(self, imgs):
        """Encode images to latent features (for use in world model)"""
        latent, _ = self.forward_encoder(imgs, mask_ratio=0.0)  # No masking for inference
        # Return all features including cls token and patch tokens
        return latent  # Shape: [batch_size, num_patches + 1, embed_dim]

    def decode(self, latent_features):
        """
        Decode latent features back to images (for visualization).

        Args:
            latent_features: (batch_size, num_patches+1, embed_dim) tensor from encode()

        Returns:
            decoded_imgs: (batch_size, 3, H, W) tensor
        """
        B = latent_features.shape[0]
        num_patches = self.patch_embed.num_patches

        # Create a dummy ids_restore (no masking - identity mapping)
        ids_restore = torch.arange(num_patches, device=latent_features.device).unsqueeze(0).repeat(B, 1)

        # latent_features already has shape [B, num_patches+1, embed_dim]
        # Pass directly to decoder
        pred = self.forward_decoder(latent_features, ids_restore)

        # Reshape patches to image
        pred = self.unpatchify(pred)
        return pred

    def unpatchify(self, x):
        """Convert patches back to image format"""
        patch_size = int(self.patch_embed.patch_size[0])
        h, w = self.grid_size  # Use stored grid_size tuple
        x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)
        return imgs

    def decode_from_latent(self, latent_features):
        """
        Decode latent features back to images (BaseAutoencoder interface).

        Args:
            latent_features: (batch_size, num_tokens, embed_dim) tensor

        Returns:
            decoded_imgs: (batch_size, 3, H, W) tensor
        """
        # Use the existing decode method
        return self.decode(latent_features)

    def compute_reconstruction_loss(self, imgs, **kwargs):
        """
        Compute reconstruction loss for training (BaseAutoencoder interface).

        Args:
            imgs: (batch_size, 3, H, W) tensor of images
            **kwargs: mask_ratio (float in [0, 1] for masked reconstruction)

        Returns:
            loss: scalar tensor
        """
        mask_ratio = kwargs.get('mask_ratio', 0.0)
        pred_patches, _ = self.forward(imgs, mask_ratio=mask_ratio)
        target_patches = self.patchify(imgs)
        loss = torch.nn.functional.mse_loss(pred_patches, target_patches)
        return loss

    def train_step(self, imgs, optimizer, **kwargs):
        """
        Perform one training step on a batch of images.

        Args:
            imgs: (batch_size, 3, H, W) tensor of images
            optimizer: torch optimizer instance
            **kwargs: Optional override parameters (mask_ratio_min, mask_ratio_max)

        Returns:
            loss_value: float, the loss value for this step
        """
        import random
        import config as cfg

        # Randomly sample mask ratio for this training step
        mask_ratio_min = kwargs.get('mask_ratio_min', cfg.AutoencoderLatentPredictorWorldModelConfig.MASK_RATIO_MIN)
        mask_ratio_max = kwargs.get('mask_ratio_max', cfg.AutoencoderLatentPredictorWorldModelConfig.MASK_RATIO_MAX)
        mask_ratio = random.uniform(mask_ratio_min, mask_ratio_max)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with masking
        pred_patches, latent = self.forward(imgs, mask_ratio=mask_ratio)

        # Calculate reconstruction loss
        target_patches = self.patchify(imgs)
        loss = torch.nn.functional.mse_loss(pred_patches, target_patches)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        return loss.item()

    def patchify(self, imgs):
        """
        Convert images to patches.

        Args:
            imgs: (batch_size, 3, H, W) tensor

        Returns:
            patches: (batch_size, num_patches, patch_size**2 * 3) tensor
        """
        patch_size = int(self.patch_embed.patch_size[0])
        B, C, H, W = imgs.shape
        h = H // patch_size
        w = W // patch_size
        x = imgs.reshape(B, C, h, patch_size, w, patch_size)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h * w, patch_size**2 * C)
        return x

    def reconstruct(self, imgs):
        """Full reconstruction using forward pass with no masking"""
        with torch.no_grad():
            pred_patches, _ = self.forward(imgs, mask_ratio=0.0)
            decoded_tensor = self.unpatchify(pred_patches)
            return decoded_tensor