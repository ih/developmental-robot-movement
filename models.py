"""
Neural network models for the developmental robot movement system.
"""

import torch
import torch.nn as nn
import timm


class MaskedAutoencoderViT(nn.Module):
    """
    A Vision Transformer (ViT) based Masked Autoencoder (MAE)
    with a POWERFUL encoder and a LIGHTWEIGHT MLP decoder.
    """
    def __init__(self, image_size=224, patch_size=16, embed_dim=256,
                 decoder_embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE ENCODER (Powerful Transformer)
        # This part remains the same as before.
        self.patch_embed = timm.models.vision_transformer.PatchEmbed(image_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = image_size // patch_size
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

    def _build_2d_sincos_pos_embed(self, embed_dim: int, grid_size: int, device=None):
        """
        Create 2D sin-cos positional embeddings with a [CLS] token at the start.
        Returns a tensor of shape [1, grid_size*grid_size + 1, embed_dim].
        """
        assert embed_dim % 2 == 0, "embed_dim must be even for 2D sin-cos embeddings"
        # Positions along each axis
        grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
        # Normalize to [0, 1]
        grid_h = grid_h / (grid_size - 1 if grid_size > 1 else 1)
        grid_w = grid_w / (grid_size - 1 if grid_size > 1 else 1)
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
        # (This function remains exactly the same)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        B, L, D = x.shape
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
        # Return cls token as the image representation
        return latent[:, 0, :]  # Shape: [batch_size, embed_dim]

    def decode(self, latent_features):
        """Decode latent features back to images (for visualization)"""
        # Expand cls token to full sequence for decoder
        B = latent_features.shape[0]
        num_patches = self.patch_embed.num_patches
        
        # Create a dummy ids_restore (no masking)
        ids_restore = torch.arange(num_patches, device=latent_features.device).unsqueeze(0).repeat(B, 1)
        
        # Expand latent to include position for all patches (simplified)
        x = latent_features.unsqueeze(1)  # Add sequence dim
        # Pad with zeros for patch positions (this is a simplification)
        x_padded = torch.cat([x, torch.zeros(B, num_patches, x.shape[-1], device=x.device)], dim=1)
        
        pred = self.forward_decoder(x_padded, ids_restore)
        
        # Reshape to image
        pred = self.unpatchify(pred)
        return pred

    def unpatchify(self, x):
        """Convert patches back to image format"""
        patch_size = int(self.patch_embed.patch_size[0])
        h = w = int(x.shape[1] ** 0.5)  # Assume square image
        x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)
        return imgs

    def reconstruct(self, imgs):
        """Full reconstruction using forward pass with no masking"""
        with torch.no_grad():
            pred_patches, _ = self.forward(imgs, mask_ratio=0.0)
            decoded_tensor = self.unpatchify(pred_patches)
            return decoded_tensor
