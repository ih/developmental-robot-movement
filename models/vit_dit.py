"""
Diffusion Transformer (DiT) for latent-space visual prediction.

A transformer architecture with adaptive Layer Normalization (adaLN-Zero)
conditioning on diffusion timesteps. Operates on latent patches produced
by a frozen VAE/encoder, not on pixel patches directly.

Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.base_autoencoder import BaseAutoencoder


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    Maps integer timestep to a fixed-size vector using sinusoidal encoding,
    then projects through a 2-layer MLP for richer conditioning.

    Args:
        embed_dim: Output embedding dimension.
        max_period: Maximum period for the sinusoidal encoding.
    """

    def __init__(self, embed_dim: int, max_period: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

        # 2-layer MLP to project sinusoidal encoding
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] integer timesteps.

        Returns:
            Conditioning vectors [B, embed_dim].
        """
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
            / half_dim
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.embed_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return self.mlp(embedding)


# ---------------------------------------------------------------------------
# Modulation helper
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive LayerNorm modulation: scale * x + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# DiT Block with adaLN-Zero
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    Transformer block with adaptive LayerNorm-Zero (adaLN-Zero) conditioning.

    Replaces standard ViT block's learnable LayerNorm with conditioning-dependent
    modulation. Gate parameters are zero-initialized so each block starts as
    identity (same principle as depth growth zero-init).

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        # LayerNorms without learnable affine (modulation replaces them)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        # Standard multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Standard MLP
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

        # adaLN-Zero: conditioning -> 6 modulation parameters
        # (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim),
        )

        # Zero-initialize the modulation output so gates start at 0
        # This makes each block an identity function at initialization
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token sequence [B, N, D].
            c: Conditioning vector [B, D] (from timestep embedding).

        Returns:
            Updated token sequence [B, N, D].
        """
        # Get 6 modulation parameters from conditioning
        mod = self.adaLN_modulation(c)  # [B, 6*D]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Modulated self-attention
        x_norm = modulate(self.norm1(x), beta1, gamma1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + alpha1.unsqueeze(1) * attn_out

        # Modulated MLP
        x_norm = modulate(self.norm2(x), beta2, gamma2)
        x = x + alpha2.unsqueeze(1) * self.mlp(x_norm)

        return x

    def forward_with_attn(self, x: torch.Tensor, c: torch.Tensor):
        """Forward pass that also returns attention weights."""
        mod = self.adaLN_modulation(c)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        x_norm = modulate(self.norm1(x), beta1, gamma1)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        x = x + alpha1.unsqueeze(1) * attn_out

        x_norm = modulate(self.norm2(x), beta2, gamma2)
        x = x + alpha2.unsqueeze(1) * self.mlp(x_norm)

        return x, attn_weights


# ---------------------------------------------------------------------------
# Diffusion Vision Transformer
# ---------------------------------------------------------------------------

class DiffusionViT(BaseAutoencoder):
    """
    Diffusion Transformer for latent-space masked inpainting.

    Operates on latent patches (from a frozen VAE) with timestep conditioning
    via adaLN-Zero. All patches (visible + masked) are processed together
    through a single transformer stack.

    Args:
        img_height: Latent height (e.g., 28 for 224px with 8x VAE).
        img_width: Latent width (e.g., 88 for 704px with 8x VAE).
        in_channels: Number of latent channels (e.g., 4 for SD VAE).
        patch_size: Patch size in latent space (default 2).
        embed_dim: Token embedding dimension.
        depth: Number of DiT blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        prediction_type: What the model predicts ('epsilon' or 'sample').
    """

    def __init__(
        self,
        img_height: int = 28,
        img_width: int = 88,
        in_channels: int = 4,
        patch_size: int = 2,
        embed_dim: int = 256,
        depth: int = 12,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        prediction_type: str = "epsilon",
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.image_size = (img_height, img_width)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.prediction_type = prediction_type

        # Grid dimensions
        assert img_height % patch_size == 0, f"img_height ({img_height}) not divisible by patch_size ({patch_size})"
        assert img_width % patch_size == 0, f"img_width ({img_width}) not divisible by patch_size ({patch_size})"
        self.grid_size = (img_height // patch_size, img_width // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_tokens = num_patches + 1  # +1 for CLS

        # Patch embedding for latent input
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed 2D sin-cos positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # Timestep conditioning
        self.time_embed = SinusoidalTimestepEmbedding(embed_dim)

        # DiT transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Alias for gradient diagnostics compatibility with TargetedTrainingMixin
        self.decoder_blocks = self.blocks

        # Final layer: adaLN + linear prediction
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),  # gamma, beta
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        # Prediction head: projects to latent patch values
        patch_dim = patch_size ** 2 * in_channels
        self.decoder_pred = nn.Linear(embed_dim, patch_dim, bias=True)

        # Initialize
        self._init_positional_embeddings()
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------
    def _init_weights(self):
        # Zero-init prediction head (stable initial outputs)
        nn.init.zeros_(self.decoder_pred.weight)
        nn.init.zeros_(self.decoder_pred.bias)
        # Small normal init for learnable tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    # ------------------------------------------------------------------
    # Positional embedding (2D sin-cos, same as MAE/DecoderOnlyViT)
    # ------------------------------------------------------------------
    def _init_positional_embeddings(self):
        pos = self._build_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_size, device=self.pos_embed.device)
        with torch.no_grad():
            self.pos_embed.copy_(pos)

    def _build_2d_sincos_pos_embed(self, embed_dim, grid_size, device=None):
        assert embed_dim % 2 == 0
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
        assert embed_dim % 2 == 0
        dim_t = torch.arange(embed_dim // 2, dtype=torch.float32, device=positions.device)
        dim_t = 1.0 / (10000 ** (dim_t / (embed_dim // 2)))
        out = positions.unsqueeze(1) * dim_t.unsqueeze(0)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        return emb

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------
    def forward_with_patch_mask(
        self,
        imgs: torch.Tensor,
        patch_mask: torch.Tensor = None,
        return_attn: bool = False,
        timestep: torch.Tensor = None,
    ):
        """
        Forward pass with optional binary mask for targeted inpainting.

        Args:
            imgs: [B, C, H, W] latent images.
            patch_mask: [B, num_patches] boolean; True = masked. None = no masking
                        (unconditional mode: all patches processed as-is).
            return_attn: If True, return attention weights.
            timestep: [B] integer timesteps. Defaults to 0 (fully denoised).

        Returns:
            pred: [B, num_patches, patch_dim] predicted patches.
            latent: [B, 1+num_patches, embed_dim] transformer output.
            attn_weights: (optional) list of attention weight tensors.
            patch_mask: (optional) the input patch mask.
        """
        B = imgs.shape[0]
        device = imgs.device

        # Default timestep = 0 for backward compatibility
        if timestep is None:
            timestep = torch.zeros(B, dtype=torch.long, device=device)

        # 1. Patch embed
        x = self.patch_embed(imgs)  # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        L = x.shape[1]

        # 2. Replace masked patches with mask_token (skip in unconditional mode)
        if patch_mask is not None:
            mask_tokens = self.mask_token.expand(B, L, -1)
            x = torch.where(patch_mask.unsqueeze(-1), mask_tokens, x)

        # 3. Add positional embeddings
        x = x + self.pos_embed[:, 1:, :]

        # 4. Prepend CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(B, -1, -1), x], dim=1)

        # 5. Get timestep conditioning
        c = self.time_embed(timestep)  # [B, embed_dim]

        # 6. Process through DiT blocks
        attn_weights = []
        if return_attn:
            for blk in self.blocks:
                x, attn = blk.forward_with_attn(x, c)
                attn_weights.append(attn)
        else:
            for blk in self.blocks:
                x = blk(x, c)

        latent = x

        # 7. Final layer with conditioning
        final_mod = self.final_modulation(c)
        gamma, beta = final_mod.chunk(2, dim=-1)
        x = modulate(self.final_norm(x), beta, gamma)

        # 8. Prediction head (skip CLS)
        pred = self.decoder_pred(x[:, 1:, :])  # [B, num_patches, patch_dim]

        if return_attn:
            return pred, latent, attn_weights, patch_mask
        return pred, latent

    def forward_denoise(
        self,
        noisy_imgs: torch.Tensor,
        patch_mask: torch.Tensor = None,
        timestep: torch.Tensor = None,
    ):
        """
        Explicit denoising forward pass.

        Same as forward_with_patch_mask but with explicit timestep.
        Used by the diffusion training loop.
        """
        return self.forward_with_patch_mask(noisy_imgs, patch_mask, timestep=timestep)

    def forward(self, imgs, mask_ratio=0.75):
        """Standard forward with random masking (BaseAutoencoder interface)."""
        import random as py_random
        B = imgs.shape[0]
        L = self.grid_size[0] * self.grid_size[1]

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
        """Encode latent images to transformer features (no masking)."""
        B = imgs.shape[0]
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(B, -1, -1), x], dim=1)

        # Use t=0 conditioning for encode
        c = self.time_embed(torch.zeros(B, dtype=torch.long, device=imgs.device))
        for blk in self.blocks:
            x = blk(x, c)

        return x

    def decode(self, latent_features, return_attn=False):
        """Decode transformer features to latent patches."""
        B = latent_features.shape[0]
        c = self.time_embed(torch.zeros(B, dtype=torch.long, device=latent_features.device))

        final_mod = self.final_modulation(c)
        gamma, beta = final_mod.chunk(2, dim=-1)
        x = modulate(self.final_norm(latent_features), beta, gamma)

        pred = self.decoder_pred(x[:, 1:, :])
        pred_img = self.unpatchify(pred)

        if return_attn:
            return pred_img, []
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
        return F.mse_loss(pred_patches, target_patches)

    def train_step(self, imgs, optimizer, **kwargs):
        """Perform one training step (BaseAutoencoder interface)."""
        import random
        mask_ratio = random.uniform(
            kwargs.get('mask_ratio_min', 0.3),
            kwargs.get('mask_ratio_max', 0.85),
        )
        optimizer.zero_grad()
        pred_patches, _ = self.forward(imgs, mask_ratio=mask_ratio)
        target_patches = self.patchify(imgs)
        loss = F.mse_loss(pred_patches, target_patches)
        loss.backward()
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Patch utilities
    # ------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert latent images to patches: [B, C, H, W] -> [B, num_patches, P^2*C]."""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = H // p
        w = W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, h, w, p, p, C]
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to latent image: [B, num_patches, P^2*C] -> [B, C, H, W]."""
        p = self.patch_size
        C = self.in_channels
        h, w = self.grid_size
        x = x.reshape(x.shape[0], h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
        imgs = x.reshape(x.shape[0], C, h * p, w * p)
        return imgs
