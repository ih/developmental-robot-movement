"""
Decoder-Only Vision Transformer for visual prediction.

A GPT-inspired single-stack transformer where all patches (visible and masked)
interact from the very first layer. Unlike the encoder-decoder MAE, there is no
separate encoder — masked patches are replaced with a learnable mask token before
any transformer processing, and the single stack handles both context understanding
and inpainting simultaneously.
"""

import torch
import torch.nn as nn
import timm
from models.base_autoencoder import BaseAutoencoder


class DecoderOnlyViT(BaseAutoencoder):
    """
    Decoder-only Vision Transformer for masked prediction.

    All patches (visible + masked) are processed through a single transformer stack.
    Masked patches start as learnable mask tokens and get refined through attention
    with visible context patches at every layer.
    """
    def __init__(self, img_height=224, img_width=224, patch_size=16, embed_dim=256,
                 depth=10, num_heads=4, mlp_ratio=4.):
        super().__init__()

        # Store dimensions for base class interface
        self.embed_dim = embed_dim
        self.image_size = (img_height, img_width)
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = timm.models.vision_transformer.PatchEmbed(
            (img_height, img_width), patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_tokens = num_patches + 1  # +1 for CLS token
        self.grid_size = (img_height // patch_size, img_width // patch_size)

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed 2D sin-cos positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # Single transformer stack
        self.blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head (projects to pixel values)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * 3, bias=True)

        # Alias for train_on_canvas gradient diagnostics compatibility
        self.decoder_blocks = self.blocks

        # Cache for attention re-run in decode()
        self._cached_patch_input = None

        # Initialize positional embeddings and weights
        self._init_positional_embeddings()
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialization (MAE convention)
    # ------------------------------------------------------------------
    def _init_weights(self):
        # Zero-init prediction head so initial outputs are near zero (stable for any embed_dim)
        nn.init.zeros_(self.decoder_pred.weight)
        nn.init.zeros_(self.decoder_pred.bias)
        # Small normal init for learnable tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    # ------------------------------------------------------------------
    # Positional embedding utilities (2D sin-cos, same as MAE)
    # ------------------------------------------------------------------
    def _init_positional_embeddings(self):
        pos = self._build_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_size, device=self.pos_embed.device)
        with torch.no_grad():
            self.pos_embed.copy_(pos)

    def _build_2d_sincos_pos_embed(self, embed_dim, grid_size, device=None):
        assert embed_dim % 2 == 0, "embed_dim must be even for 2D sin-cos embeddings"
        grid_h_size, grid_w_size = grid_size
        grid_h = torch.arange(grid_h_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(grid_w_size, dtype=torch.float32, device=device)
        grid_h = grid_h / (grid_h_size - 1 if grid_h_size > 1 else 1)
        grid_w = grid_w / (grid_w_size - 1 if grid_w_size > 1 else 1)
        gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij')
        gh = gh.reshape(-1)
        gw = gw.reshape(-1)
        dim_half = embed_dim // 2
        pos_h = self._get_1d_sincos_pos_embed(dim_half, gh)
        pos_w = self._get_1d_sincos_pos_embed(dim_half, gw)
        pos = torch.cat([pos_h, pos_w], dim=1)
        cls_pos = torch.zeros(1, 1, embed_dim, dtype=pos.dtype, device=pos.device)
        pos = pos.unsqueeze(0)
        pos = torch.cat([cls_pos, pos], dim=1)
        return pos

    def _get_1d_sincos_pos_embed(self, embed_dim, positions):
        assert embed_dim % 2 == 0, "1D embed_dim should be even"
        dim_t = torch.arange(embed_dim // 2, dtype=torch.float32, device=positions.device)
        dim_t = 1.0 / (10000 ** (dim_t / (embed_dim // 2)))
        out = positions.unsqueeze(1) * dim_t.unsqueeze(0)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        return emb

    # ------------------------------------------------------------------
    # Attention capture for visualization
    # ------------------------------------------------------------------
    def _forward_block_with_attn(self, block, x):
        """Forward through a transformer block while capturing attention weights."""
        x_normed = block.norm1(x)
        attn_module = block.attn
        B, N, C = x_normed.shape

        qkv = attn_module.qkv(x_normed)
        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = (q @ k.transpose(-2, -1)) * attn_module.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights_saved = attn_weights.clone()
        attn_weights = attn_module.attn_drop(attn_weights)

        x_attn = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = attn_module.proj(x_attn)
        x_attn = attn_module.proj_drop(x_attn)

        if hasattr(block, 'drop_path1'):
            x = x + block.drop_path1(x_attn)
        else:
            x = x + x_attn

        x_normed2 = block.norm2(x)
        x_mlp = block.mlp(x_normed2)

        if hasattr(block, 'drop_path2'):
            x = x + block.drop_path2(x_mlp)
        else:
            x = x + x_mlp

        return x, attn_weights_saved

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------
    def _forward_blocks(self, x, return_attn=False):
        """Process through transformer blocks, optionally capturing attention."""
        attn_weights = []
        if return_attn:
            for blk in self.blocks:
                x, attn = self._forward_block_with_attn(blk, x)
                attn_weights.append(attn)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm(x)
        if return_attn:
            return x, attn_weights
        return x

    def forward_with_patch_mask(self, imgs, patch_mask, return_attn=False):
        """
        Forward pass with custom binary mask for targeted inpainting.

        Args:
            imgs: [B, 3, H, W] input images
            patch_mask: [B, num_patches] boolean; True = masked
            return_attn: if True, return attention weights

        Returns:
            pred: [B, num_patches, patch_size^2 * 3] predicted patches
            latent: [B, 1+num_patches, embed_dim] transformer output
            attn_weights: (optional) list of [B, num_heads, N, N] per layer
            patch_mask: (optional) the input patch mask
        """
        # 1. Patch embed all patches
        x = self.patch_embed(imgs)  # [B, L, D]
        B, L, D = x.shape

        # 2. Replace masked patches with mask_token
        mask_tokens = self.mask_token.expand(B, L, -1)
        x = torch.where(patch_mask.unsqueeze(-1), mask_tokens, x)

        # 3. Add positional embeddings
        x = x + self.pos_embed[:, 1:, :]

        # 4. Prepend CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(B, -1, -1), x], dim=1)

        # 5. Process through single transformer stack
        if return_attn:
            x, attn_weights = self._forward_blocks(x, return_attn=True)
        else:
            x = self._forward_blocks(x, return_attn=False)

        latent = x

        # 6. Prediction head (skip CLS token)
        pred = self.decoder_pred(x[:, 1:, :])  # [B, L, P^2*3]

        if return_attn:
            return pred, latent, attn_weights, patch_mask
        return pred, latent

    def forward(self, imgs, mask_ratio=0.75):
        """Standard forward with random masking (BaseAutoencoder interface)."""
        import random as py_random
        B, _, H, W = imgs.shape
        L = self.patch_embed.num_patches

        # Create random mask
        if mask_ratio > 0:
            num_mask = int(L * mask_ratio)
            patch_mask = torch.zeros(B, L, dtype=torch.bool, device=imgs.device)
            for b in range(B):
                indices = torch.randperm(L, device=imgs.device)[:num_mask]
                patch_mask[b, indices] = True
        else:
            patch_mask = torch.zeros(B, L, dtype=torch.bool, device=imgs.device)

        pred, latent = self.forward_with_patch_mask(imgs, patch_mask)
        return pred, latent

    # ------------------------------------------------------------------
    # BaseAutoencoder interface
    # ------------------------------------------------------------------
    def encode(self, imgs):
        """Encode images to latent features (no masking)."""
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        # Cache for potential decode() attention re-run
        self._cached_patch_input = x.detach().clone()

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        x = self._forward_blocks(x, return_attn=False)
        return x  # [B, 1+num_patches, embed_dim]

    def decode(self, latent_features, return_attn=False):
        """
        Decode latent features back to images.

        For decoder-only: the latent already contains the full transformer output.
        If return_attn=True, we re-run the forward pass with attention capture
        using the cached input from encode().
        """
        if return_attn and self._cached_patch_input is not None:
            # Re-run blocks with attention capture
            x = self._cached_patch_input
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            x, attn_weights = self._forward_blocks(x, return_attn=True)
            pred = self.decoder_pred(x[:, 1:, :])
            pred_img = self.unpatchify(pred)
            return pred_img, attn_weights

        # Normal decode: apply prediction head to already-processed latent
        pred = self.decoder_pred(latent_features[:, 1:, :])
        pred_img = self.unpatchify(pred)
        return pred_img

    def decode_from_latent(self, latent_features):
        """Decode latent features (BaseAutoencoder interface)."""
        return self.decode(latent_features)

    def reconstruct(self, imgs):
        """Full reconstruction without masking."""
        with torch.no_grad():
            pred_patches, _ = self.forward(imgs, mask_ratio=0.0)
            return self.unpatchify(pred_patches)

    def compute_reconstruction_loss(self, imgs, **kwargs):
        """Compute reconstruction loss (BaseAutoencoder interface)."""
        mask_ratio = kwargs.get('mask_ratio', 0.0)
        pred_patches, _ = self.forward(imgs, mask_ratio=mask_ratio)
        target_patches = self.patchify(imgs)
        loss = torch.nn.functional.mse_loss(pred_patches, target_patches)
        return loss

    def train_step(self, imgs, optimizer, **kwargs):
        """Perform one training step (BaseAutoencoder interface)."""
        import random
        import config as cfg

        mask_ratio_min = kwargs.get('mask_ratio_min', cfg.MASK_RATIO_MIN)
        mask_ratio_max = kwargs.get('mask_ratio_max', cfg.MASK_RATIO_MAX)
        mask_ratio = random.uniform(mask_ratio_min, mask_ratio_max)

        optimizer.zero_grad()
        pred_patches, _ = self.forward(imgs, mask_ratio=mask_ratio)
        target_patches = self.patchify(imgs)
        loss = torch.nn.functional.mse_loss(pred_patches, target_patches)
        loss.backward()
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Patch utilities (identical to MaskedAutoencoderViT)
    # ------------------------------------------------------------------
    def patchify(self, imgs):
        """Convert images to patches: [B, 3, H, W] -> [B, num_patches, P^2*3]."""
        patch_size = int(self.patch_embed.patch_size[0])
        B, C, H, W = imgs.shape
        h = H // patch_size
        w = W // patch_size
        x = imgs.reshape(B, C, h, patch_size, w, patch_size)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h * w, patch_size**2 * C)
        return x

    def unpatchify(self, x):
        """Convert patches back to image: [B, num_patches, P^2*3] -> [B, 3, H, W]."""
        patch_size = int(self.patch_embed.patch_size[0])
        h, w = self.grid_size
        x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)
        return imgs
