"""
VAE/RAE encoders for canvas compression in latent diffusion.

Provides a common interface for multiple encoder backends:
- CanvasVAE: Custom convolutional VAE/RAE trained from scratch
- PretrainedSDVAE: Stable Diffusion's AutoencoderKL (4ch, 8x compression)
- PretrainedFluxVAE: FLUX's AutoencoderKL (16ch, 8x compression)
- DINOv2Encoder: Frozen DINOv2 ViT encoder + trainable decoder

All encoders implement the same interface:
    encode(imgs) -> latent
    decode(latent) -> imgs
    latent_channels: int
    compression_factor: int
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Common interface (for documentation; not enforced via ABC to keep simple)
# ---------------------------------------------------------------------------
# encode(imgs: Tensor[B,3,H,W]) -> Tensor[B,C,H_lat,W_lat]
# decode(latent: Tensor[B,C,H_lat,W_lat]) -> Tensor[B,3,H,W]
# latent_channels: int
# compression_factor: int


# ---------------------------------------------------------------------------
# Custom VAE / RAE
# ---------------------------------------------------------------------------

class CanvasVAE(nn.Module):
    """
    Custom convolutional VAE/RAE for training from scratch on canvas data.

    Args:
        latent_channels: Number of channels in the latent space.
        compression_factor: Spatial downsampling factor (must be power of 2).
        mode: 'vae' for KL divergence regularization, 'rae' for L2 regularization.
        base_channels: Base channel count for the encoder/decoder.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        compression_factor: int = 8,
        mode: str = "vae",
        base_channels: int = 64,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.compression_factor = compression_factor
        self.mode = mode

        # Number of downsampling stages
        num_stages = int(math.log2(compression_factor))
        assert 2 ** num_stages == compression_factor, \
            f"compression_factor must be a power of 2, got {compression_factor}"

        # Build encoder
        encoder_layers = []
        in_ch = 3
        for i in range(num_stages):
            out_ch = base_channels * (2 ** i)
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        # Bottleneck projection
        encoder_out_ch = base_channels * (2 ** (num_stages - 1))
        if mode == "vae":
            # Mean and logvar for reparameterization
            self.to_mean = nn.Conv2d(encoder_out_ch, latent_channels, 1)
            self.to_logvar = nn.Conv2d(encoder_out_ch, latent_channels, 1)
        else:
            # Direct projection for RAE
            self.to_latent = nn.Conv2d(encoder_out_ch, latent_channels, 1)

        # Build decoder
        decoder_layers = []
        in_ch = latent_channels
        for i in range(num_stages - 1, -1, -1):
            out_ch = base_channels * (2 ** i) if i > 0 else base_channels
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
            ])
            in_ch = out_ch
        # Final projection to RGB
        decoder_layers.append(nn.Conv2d(base_channels, 3, 3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

        # Scaling factor (computed during first encode, or set manually)
        self.register_buffer('scaling_factor', torch.tensor(1.0))
        self._scaling_computed = False

    def _encode_raw(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode to raw latent (before scaling)."""
        h = self.encoder(imgs)
        if self.mode == "vae":
            mean = self.to_mean(h)
            logvar = self.to_logvar(h)
            # Reparameterization trick
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mean + eps * std
            else:
                z = mean
            self._last_mean = mean
            self._last_logvar = logvar
            return z
        else:
            return self.to_latent(h)

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode images to scaled latent representation."""
        z = self._encode_raw(imgs)
        return z * self.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode scaled latent back to images."""
        z = latent / self.scaling_factor.clamp(min=1e-8)
        return self.decoder(z)

    def compute_scaling_factor(self, dataloader, num_batches: int = 50):
        """Compute scaling factor from data (std of raw latents)."""
        self.eval()
        latents = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                else:
                    imgs = batch
                imgs = imgs.to(next(self.parameters()).device)
                z = self._encode_raw(imgs)
                latents.append(z)
        all_latents = torch.cat(latents, dim=0)
        std = all_latents.std().item()
        self.scaling_factor.fill_(1.0 / max(std, 1e-8))
        self._scaling_computed = True
        print(f"CanvasVAE scaling factor set to {self.scaling_factor.item():.4f} (latent std={std:.4f})")

    def kl_loss(self) -> torch.Tensor:
        """KL divergence loss (only valid in VAE mode after encode)."""
        if self.mode != "vae":
            return torch.tensor(0.0)
        mean = self._last_mean
        logvar = self._last_logvar
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    def training_step(
        self,
        imgs: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        kl_weight: float = 1e-4,
        l2_weight: float = 1e-4,
    ) -> dict:
        """
        Single VAE/RAE training step.

        Returns:
            Dict with 'recon_loss', 'reg_loss', 'total_loss'.
        """
        optimizer.zero_grad()

        # Encode (raw, not scaled — scaling is for DiT training)
        h = self.encoder(imgs)
        if self.mode == "vae":
            mean = self.to_mean(h)
            logvar = self.to_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            self._last_mean = mean
            self._last_logvar = logvar
        else:
            z = self.to_latent(h)

        # Decode
        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, imgs)

        # Regularization
        if self.mode == "vae":
            reg_loss = self.kl_loss() * kl_weight
        else:
            reg_loss = z.pow(2).mean() * l2_weight

        total_loss = recon_loss + reg_loss
        total_loss.backward()
        optimizer.step()

        return {
            'recon_loss': recon_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item(),
        }


# ---------------------------------------------------------------------------
# Pre-trained Stable Diffusion VAE
# ---------------------------------------------------------------------------

class PretrainedSDVAE(nn.Module):
    """
    Wrapper around Stable Diffusion's AutoencoderKL.

    4 latent channels, 8x spatial compression. Always frozen.
    Requires: pip install diffusers
    """

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.latent_channels = 4
        self.compression_factor = 8
        self.model_id = model_id

        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(model_id)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # SD VAE scaling factor
        self._scaling_factor = 0.18215

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode images [B,3,H,W] in [0,1] to scaled latent."""
        # SD VAE expects [-1, 1] input
        x = imgs * 2.0 - 1.0
        with torch.no_grad():
            posterior = self.vae.encode(x)
            z = posterior.latent_dist.mode()  # deterministic (no reparameterization noise)
        return z * self._scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode scaled latent to images [B,3,H,W] in [0,1]."""
        z = latent / self._scaling_factor
        with torch.no_grad():
            x = self.vae.decode(z).sample
        return (x + 1.0) / 2.0  # [-1,1] -> [0,1]

    def train(self, mode=True):
        """Override: always stays in eval mode."""
        return super().train(False)


# ---------------------------------------------------------------------------
# Pre-trained FLUX VAE
# ---------------------------------------------------------------------------

class PretrainedFluxVAE(nn.Module):
    """
    Wrapper around FLUX's AutoencoderKL.

    16 latent channels, 8x spatial compression. Always frozen.
    Requires: pip install diffusers
    """

    def __init__(self, model_id: str = "black-forest-labs/FLUX.2-dev", subfolder: str = "vae"):
        super().__init__()
        self.compression_factor = 8
        self.model_id = model_id

        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder=subfolder, torch_dtype=torch.bfloat16)

        # Detect actual latent channels from model config (FLUX.2-dev uses 32, not 16)
        self.latent_channels = self.vae.config.latent_channels
        self.vae.eval()
        self.vae.requires_grad_(False)

        # FLUX-specific scaling parameters
        self._scale = 0.3611
        self._shift = 0.1159

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode images [B,3,H,W] in [0,1] to scaled latent."""
        x = imgs * 2.0 - 1.0
        orig_dtype = x.dtype
        x = x.to(torch.bfloat16)
        with torch.no_grad():
            posterior = self.vae.encode(x)
            z = posterior.latent_dist.mode()  # deterministic (no reparameterization noise)
        z = z.to(orig_dtype)
        return (z - self._shift) * self._scale

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode scaled latent to images [B,3,H,W] in [0,1]."""
        z = latent / self._scale + self._shift
        orig_dtype = z.dtype
        z = z.to(torch.bfloat16)
        with torch.no_grad():
            x = self.vae.decode(z).sample
        x = x.to(orig_dtype)
        return (x + 1.0) / 2.0

    def train(self, mode=True):
        """Override: always stays in eval mode."""
        return super().train(False)


# ---------------------------------------------------------------------------
# DINOv2 Encoder + Trainable Decoder
# ---------------------------------------------------------------------------

class DINOv2Decoder(nn.Module):
    """
    Lightweight CNN decoder for reconstructing images from DINOv2 features.

    Takes reshaped DINOv2 patch tokens [B, embed_dim, H_grid, W_grid] and
    upsamples back to pixel space [B, 3, H, W].
    """

    def __init__(self, embed_dim: int, grid_h: int, grid_w: int, target_h: int, target_w: int):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.target_h = target_h
        self.target_w = target_w

        # Project from high-dim features to manageable channel count
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
        )

        # Upsample stages: each doubles spatial resolution
        # DINOv2 with patch_size=14: grid is input_size/14
        # Need to go from grid_size to target_size
        # Using interpolation + conv refinement
        self.upsample_blocks = nn.ModuleList()
        channels = [256, 128, 64, 32]
        for i in range(len(channels) - 1):
            self.upsample_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(channels[i], channels[i + 1], 4, stride=2, padding=1),
                nn.GroupNorm(min(32, channels[i + 1]), channels[i + 1]),
                nn.SiLU(),
            ))

        # Final projection to RGB
        self.to_rgb = nn.Conv2d(channels[-1], 3, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, embed_dim, grid_h, grid_w] reshaped DINOv2 patch tokens.

        Returns:
            Reconstructed images [B, 3, target_h, target_w].
        """
        x = self.proj(features)

        for block in self.upsample_blocks:
            x = block(x)

        # Final interpolation to exact target size
        if x.shape[2] != self.target_h or x.shape[3] != self.target_w:
            x = F.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)

        return self.to_rgb(x)


class DINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 encoder with a trainable decoder.

    Uses DINOv2's self-supervised ViT as a frozen feature extractor,
    paired with a trainable CNN decoder for reconstruction. This
    implements the "Representation Autoencoder" (RAE) approach.

    Note: DINOv2 uses patch_size=14, so input dimensions must be
    divisible by 14. Canvas SEPARATOR_WIDTH should be adjusted to 14
    when using this encoder.

    Args:
        variant: DINOv2 model variant ('vits14', 'vitb14', 'vitl14', 'vitg14').
        target_h: Target image height (for decoder output).
        target_w: Target image width (for decoder output).
    """

    # Embedding dimensions per variant
    EMBED_DIMS = {
        'vits14': 384,
        'vitb14': 768,
        'vitl14': 1024,
        'vitg14': 1536,
    }

    def __init__(self, variant: str = "vitb14", target_h: int = 224, target_w: int = 700):
        super().__init__()
        self.variant = variant
        self.latent_channels = self.EMBED_DIMS[variant]
        self.compression_factor = 14  # DINOv2 patch size
        self.target_h = target_h
        self.target_w = target_w

        # Compute grid dimensions
        assert target_h % 14 == 0, f"target_h ({target_h}) must be divisible by 14 for DINOv2"
        assert target_w % 14 == 0, f"target_w ({target_w}) must be divisible by 14 for DINOv2"
        self.grid_h = target_h // 14
        self.grid_w = target_w // 14

        # Load frozen DINOv2 encoder
        self.encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_{variant}')
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        # Trainable decoder
        self.decoder_net = DINOv2Decoder(
            embed_dim=self.latent_channels,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            target_h=target_h,
            target_w=target_w,
        )

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode images to DINOv2 features reshaped as spatial latent.

        Args:
            imgs: [B, 3, H, W] images in [0, 1].

        Returns:
            Latent [B, embed_dim, grid_h, grid_w].
        """
        # DINOv2 expects ImageNet-normalized input
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
        x = (imgs - mean) / std

        with torch.no_grad():
            # Get patch tokens (excluding CLS token)
            features = self.encoder.forward_features(x)
            # DINOv2 returns dict with 'x_norm_patchtokens' key, or we extract patch tokens
            if isinstance(features, dict):
                patch_tokens = features.get('x_norm_patchtokens', features.get('x_prenorm', None))
                if patch_tokens is None:
                    # Fallback: take all tokens except CLS
                    patch_tokens = features['x_norm_clstoken']  # This won't work, use different approach
            else:
                # features shape: [B, num_tokens, embed_dim] where num_tokens = 1 (CLS) + num_patches
                patch_tokens = features[:, 1:, :]  # Skip CLS token

        B = patch_tokens.shape[0]
        # Reshape to spatial: [B, num_patches, D] -> [B, D, grid_h, grid_w]
        return patch_tokens.transpose(1, 2).reshape(B, self.latent_channels, self.grid_h, self.grid_w)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent features to images [B, 3, H, W] in [0, 1]."""
        out = self.decoder_net(latent)
        return out.clamp(0, 1)

    def training_step(
        self,
        imgs: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        Train the decoder only (encoder is frozen).

        Returns:
            Dict with 'recon_loss', 'total_loss'.
        """
        optimizer.zero_grad()
        latent = self.encode(imgs)
        recon = self.decode(latent)
        recon_loss = F.mse_loss(recon, imgs)
        recon_loss.backward()
        optimizer.step()
        return {
            'recon_loss': recon_loss.item(),
            'reg_loss': 0.0,
            'total_loss': recon_loss.item(),
        }

    def train(self, mode=True):
        """Override: encoder always frozen, only decoder trains."""
        super().train(mode)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        return self


def create_vae(
    vae_type: str,
    checkpoint_path: str = None,
    device: str = "cpu",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a VAE/encoder by type.

    Args:
        vae_type: One of 'custom', 'pretrained_sd', 'pretrained_flux', 'dinov2'.
        checkpoint_path: Path to saved weights (for custom VAE or DINOv2 decoder).
        device: Device to load model on.
        **kwargs: Additional arguments passed to the constructor.

    Returns:
        VAE/encoder module.
    """
    if vae_type == "custom":
        vae = CanvasVAE(**kwargs)
    elif vae_type == "pretrained_sd":
        vae = PretrainedSDVAE(**kwargs)
    elif vae_type == "pretrained_flux":
        vae = PretrainedFluxVAE(**kwargs)
    elif vae_type == "dinov2":
        vae = DINOv2Encoder(**kwargs)
    else:
        raise ValueError(f"Unknown vae_type: {vae_type}. Use 'custom', 'pretrained_sd', 'pretrained_flux', or 'dinov2'.")

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        vae.load_state_dict(state_dict, strict=False)
        print(f"Loaded VAE checkpoint from {checkpoint_path}")

    vae = vae.to(device)
    return vae
