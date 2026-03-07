"""
Latent Diffusion Wrapper combining a frozen VAE/encoder with a DiT.

Presents the same interface as TargetedMAEWrapper / TargetedDecoderOnlyWrapper:
- train_on_canvas(canvas_tensor, patch_mask, optimizer)
- forward_with_patch_mask(canvas_tensor, patch_mask)
- patchify(imgs) / unpatchify(patches)
- decoder_pred, decoder_blocks, mask_token attributes

Internally handles: pixel->latent encoding, diffusion training, iterative
DDIM denoising, and latent->pixel decoding.
"""

import torch
import torch.nn as nn

from models.vit_dit import DiffusionViT
from models.noise_scheduler import NoiseScheduler


class LatentDiffusionWrapper(nn.Module):
    """
    Combined VAE (frozen) + DiT (trainable) for canvas-based latent diffusion.

    The VAE encodes pixel canvases to compact latent representations.
    The DiT performs diffusion-based denoising on latent patches.
    The VAE decoder reconstructs pixels from denoised latents.

    Args:
        vae: Frozen VAE/encoder module (CanvasVAE, PretrainedSDVAE, etc.).
        dit: Trainable DiffusionViT.
        noise_scheduler: NoiseScheduler for diffusion process.
        num_inference_steps: Number of DDIM steps for inference.
    """

    def __init__(
        self,
        vae: nn.Module,
        dit: DiffusionViT,
        noise_scheduler: NoiseScheduler,
        num_inference_steps: int = 50,
        training_mode: str = "conditional",
    ):
        super().__init__()
        self.vae = vae
        self.dit = dit
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.training_mode = training_mode  # "conditional" or "unconditional"

        # Freeze VAE
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # Cache pixel-space canvas dimensions for mask mapping
        self._pixel_canvas_size = None  # Set on first forward

        # Proxy attributes for gradient diagnostics (TargetedTrainingMixin expects these)
        # These are accessed by train_on_canvas for gradient norm logging

    @property
    def decoder_pred(self):
        return self.dit.decoder_pred

    @property
    def decoder_blocks(self):
        return self.dit.decoder_blocks

    @property
    def mask_token(self):
        return self.dit.mask_token

    @property
    def embed_dim(self):
        return self.dit.embed_dim

    @property
    def blocks(self):
        return self.dit.blocks

    @property
    def image_size(self):
        return self.dit.image_size

    @property
    def grid_size(self):
        return self.dit.grid_size

    @property
    def patch_size(self):
        return self.dit.patch_size

    # ------------------------------------------------------------------
    # Pixel mask -> latent mask mapping
    # ------------------------------------------------------------------
    def _pixel_mask_to_latent_mask(
        self,
        pixel_patch_mask: torch.Tensor,
        pixel_canvas_h: int,
        pixel_canvas_w: int,
    ) -> torch.Tensor:
        """
        Map a pixel-space patch mask to the latent patch grid.

        For 8x VAE compression with pixel patch_size=16 and latent patch_size=2,
        the grids align 1:1 (both produce 14x44 patches for 224x704 canvas).

        For DINOv2 (14x compression), the grid is different and requires
        spatial remapping.

        Args:
            pixel_patch_mask: [B, num_pixel_patches] boolean mask.
            pixel_canvas_h: Pixel canvas height.
            pixel_canvas_w: Pixel canvas width.

        Returns:
            latent_mask: [B, num_latent_patches] boolean mask.
        """
        B = pixel_patch_mask.shape[0]
        device = pixel_patch_mask.device

        # Pixel patch grid
        pixel_patch_size = 16  # Standard pixel patch size used by mask generation
        pixel_grid_h = pixel_canvas_h // pixel_patch_size
        pixel_grid_w = pixel_canvas_w // pixel_patch_size

        # Latent grid
        cf = self.vae.compression_factor
        latent_h = pixel_canvas_h // cf
        latent_w = pixel_canvas_w // cf
        latent_ps = self.dit.patch_size
        latent_grid_h = latent_h // latent_ps
        latent_grid_w = latent_w // latent_ps

        # Check if grids align (common case: 8x VAE, pixel_ps=16, latent_ps=2)
        if pixel_grid_h == latent_grid_h and pixel_grid_w == latent_grid_w:
            # 1:1 mapping
            return pixel_patch_mask

        # Different grids: remap spatially
        # Reshape pixel mask to 2D grid
        pixel_mask_2d = pixel_patch_mask.reshape(B, pixel_grid_h, pixel_grid_w).float()

        # Resize to latent grid using nearest-neighbor
        latent_mask_2d = torch.nn.functional.interpolate(
            pixel_mask_2d.unsqueeze(1),  # [B, 1, pH, pW]
            size=(latent_grid_h, latent_grid_w),
            mode='nearest',
        ).squeeze(1)  # [B, lH, lW]

        # Convert back to boolean flat mask
        return (latent_mask_2d > 0.5).reshape(B, -1)

    # ------------------------------------------------------------------
    # train_on_canvas (same signature as TargetedTrainingMixin)
    # ------------------------------------------------------------------
    def train_on_canvas(
        self,
        canvas_tensor: torch.Tensor,
        patch_mask: torch.Tensor,
        optimizer,
        return_per_sample_losses: bool = False,
    ):
        """
        Train the DiT on a canvas using diffusion loss.

        Two modes controlled by self.training_mode:
        - "conditional": Noise on masked patches only, loss on masked patches only.
          DiT sees mask_token for masked patches (inpainting-aware training).
        - "unconditional": Noise on ALL patches, loss on ALL patches.
          DiT sees noisy patches directly (no mask_token). Mask used only at
          inference time via RePaint.

        Same signature as TargetedTrainingMixin.train_on_canvas() for compatibility.

        Args:
            canvas_tensor: [B, 3, H, W] pixel canvas in [0, 1].
            patch_mask: [B, num_pixel_patches] boolean; True = masked.
            optimizer: Optimizer for DiT parameters.
            return_per_sample_losses: If True, also return per-sample losses.

        Returns:
            tuple: (loss_value, grad_diagnostics) or with per_sample_losses.
        """
        from config import AutoencoderConcatPredictorWorldModelConfig as Config
        from models.autoencoder_concat_predictor import compute_hybrid_loss_on_masked_patches

        B = canvas_tensor.shape[0]
        pixel_h, pixel_w = canvas_tensor.shape[2], canvas_tensor.shape[3]

        # 1. Encode canvas to latent space (frozen VAE)
        with torch.no_grad():
            latent = self.vae.encode(canvas_tensor)  # [B, C, H_lat, W_lat]

        # 2. Patchify the clean latent
        target_patches = self.dit.patchify(latent)  # [B, num_patches, patch_dim]

        # 3. Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (B,), device=canvas_tensor.device,
        )

        # 4. Sample noise
        noise = torch.randn_like(target_patches)

        # 5. Add noise to create noisy patches
        noisy_patches = self.noise_scheduler.add_noise(target_patches, noise, timesteps)

        if self.training_mode == "unconditional":
            # Unconditional: noise ALL patches, no mask_token
            noisy_latent = self.dit.unpatchify(noisy_patches)

            # DiT forward pass without mask (no mask_token replacement)
            self.dit.train()
            pred_patches, _ = self.dit.forward_with_patch_mask(
                noisy_latent, None, timestep=timesteps
            )

            # Loss target
            if self.dit.prediction_type == 'epsilon':
                loss_target = noise
            else:
                loss_target = target_patches

            # Loss on ALL patches
            all_pred = pred_patches.reshape(-1, pred_patches.shape[-1])
            all_target = loss_target.reshape(-1, loss_target.shape[-1])

            loss_dict = compute_hybrid_loss_on_masked_patches(
                all_pred, all_target,
                focal_alpha=1.0, focal_beta=0,
            )

            loss = loss_dict['loss_hybrid']
            loss_hybrid_only = loss.item()

            # Per-sample losses over all patches
            per_sample_losses = None
            if return_per_sample_losses:
                per_sample_losses = []
                with torch.no_grad():
                    for b in range(B):
                        sample_loss_dict = compute_hybrid_loss_on_masked_patches(
                            pred_patches[b], loss_target[b],
                            focal_alpha=1.0, focal_beta=0,
                        )
                        per_sample_losses.append(sample_loss_dict['loss_hybrid'].item())
        else:
            # Conditional: noise on masked patches only, mask_token in DiT
            latent_mask = self._pixel_mask_to_latent_mask(patch_mask, pixel_h, pixel_w)

            # Build composite: unmasked = clean, masked = noisy
            composite_patches = torch.where(
                latent_mask.unsqueeze(-1), noisy_patches, target_patches
            )
            noisy_latent = self.dit.unpatchify(composite_patches)

            # DiT forward pass WITHOUT mask_token replacement.
            # The model must see the actual noisy patches to predict the noise
            # (epsilon prediction). Passing latent_mask would replace noisy
            # embeddings with mask_token, destroying the noise information.
            self.dit.train()
            pred_patches, _ = self.dit.forward_with_patch_mask(
                noisy_latent, None, timestep=timesteps
            )

            # Loss target
            if self.dit.prediction_type == 'epsilon':
                loss_target = noise
            else:
                loss_target = target_patches

            # Loss on masked patches only
            masked_pred = pred_patches[latent_mask]
            masked_target = loss_target[latent_mask]

            loss_dict = compute_hybrid_loss_on_masked_patches(
                masked_pred, masked_target,
                focal_alpha=1.0, focal_beta=0,
            )

            loss = loss_dict['loss_hybrid']
            loss_hybrid_only = loss.item()

            # Per-sample losses on masked patches
            per_sample_losses = None
            if return_per_sample_losses:
                per_sample_losses = []
                with torch.no_grad():
                    for b in range(B):
                        sample_mask = latent_mask[b]
                        sample_pred = pred_patches[b][sample_mask]
                        sample_target = loss_target[b][sample_mask]
                        sample_loss_dict = compute_hybrid_loss_on_masked_patches(
                            sample_pred, sample_target,
                            focal_alpha=1.0, focal_beta=0,
                        )
                        per_sample_losses.append(sample_loss_dict['loss_hybrid'].item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient diagnostics (same structure as TargetedTrainingMixin)
        def gnorm(p):
            return None if (p.grad is None) else p.grad.detach().norm().item()

        curr_lr = optimizer.param_groups[0]['lr']
        head_w = getattr(self.dit.decoder_pred, 'weight', None)
        head_b = getattr(self.dit.decoder_pred, 'bias', None)
        mask_tok = getattr(self.dit, 'mask_token', None)

        dec_blk = self.dit.blocks[0] if len(self.dit.blocks) > 0 else None
        qkv_w = None
        if dec_blk is not None:
            attn = getattr(dec_blk, 'attn', None)
            if attn is not None:
                qkv_w = getattr(attn, 'in_proj_weight', None)

        grad_diagnostics = {
            'lr': curr_lr,
            'head_weight_norm': gnorm(head_w),
            'head_bias_norm': gnorm(head_b),
            'mask_token_norm': gnorm(mask_tok),
            'qkv_weight_norm': gnorm(qkv_w),
            'loss_hybrid': loss_hybrid_only,
            'loss_plain': loss_dict['loss_plain'].item(),
            'loss_focal': loss_dict['loss_focal'].item(),
            'loss_standard': loss_dict['loss_standard'].item(),
            'loss_nonblack': 0.0,
            'black_baseline': 0.0,
            'frac_nonblack': 1.0,
            'focal_weight_mean': loss_dict['focal_weight_mean'],
            'focal_weight_max': loss_dict['focal_weight_max'],
            'focal_beta': Config.FOCAL_BETA,
            'focal_alpha': Config.FOCAL_LOSS_ALPHA,
            'loss_perceptual': 0.0,
            'perceptual_weight': 0.0,
        }

        optimizer.step()

        if return_per_sample_losses:
            return loss_hybrid_only, grad_diagnostics, per_sample_losses
        return loss_hybrid_only, grad_diagnostics

    # ------------------------------------------------------------------
    # forward_with_patch_mask (for inference / evaluation)
    # ------------------------------------------------------------------
    def forward_with_patch_mask(
        self,
        canvas_tensor: torch.Tensor,
        patch_mask: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        Full inference: encode, denoise in latent space, decode to pixels.

        Uses iterative DDIM denoising. Returns pixel-space predictions
        compatible with unpatchify() output format.

        Args:
            canvas_tensor: [B, 3, H, W] pixel canvas in [0, 1].
            patch_mask: [B, num_pixel_patches] boolean; True = masked.
            return_attn: If True, return attention weights from last denoising step.

        Returns:
            pred_patches: [B, num_pixel_patches, pixel_patch_dim] pixel-space predictions.
            latent: Transformer latent from final denoising step.
            attn_weights: (optional) attention weights.
            patch_mask: (optional) the input mask.
        """
        pixel_h, pixel_w = canvas_tensor.shape[2], canvas_tensor.shape[3]

        # 1. Encode to latent
        with torch.no_grad():
            clean_latent = self.vae.encode(canvas_tensor)

        # 2. Map pixel mask to latent mask
        latent_mask = self._pixel_mask_to_latent_mask(patch_mask, pixel_h, pixel_w)

        # 3. Denoise in latent space
        self.dit.eval()
        denoised_latent = self._denoise_inpaint(clean_latent, latent_mask)

        # 4. Decode to pixel space
        with torch.no_grad():
            pred_canvas = self.vae.decode(denoised_latent)  # [B, 3, H, W]

        # 5. Convert to pixel patches (for compatibility with existing code)
        pred_canvas = pred_canvas.clamp(0, 1)
        pred_pixel_patches = self._pixel_patchify(pred_canvas)

        # For latent output, use last transformer state
        latent = None

        if return_attn:
            return pred_pixel_patches, latent, [], patch_mask
        return pred_pixel_patches, latent

    def _denoise_inpaint(
        self,
        clean_latent: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Iterative DDIM denoising with inpainting.

        Two modes controlled by self.training_mode:
        - "conditional": DiT sees mask_token for masked patches. DDIM on masked
          patches only. Unmasked patches stay clean.
        - "unconditional": RePaint-style. DiT processes all patches without
          mask_token. DDIM on all patches. After each step, unmasked patches are
          repainted with clean values noised to the current timestep level.

        Args:
            clean_latent: [B, C, H_lat, W_lat] clean latent from VAE.
            latent_mask: [B, num_latent_patches] boolean; True = to denoise.

        Returns:
            Denoised latent [B, C, H_lat, W_lat].
        """
        device = clean_latent.device
        B = clean_latent.shape[0]

        # Patchify clean latent for repaint
        clean_patches = self.dit.patchify(clean_latent)

        # Set up DDIM schedule
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        self.noise_scheduler.to(device)

        if self.training_mode == "unconditional":
            # RePaint: initialize ALL patches as noise
            current_patches = torch.randn_like(clean_patches)

            with torch.no_grad():
                timesteps_list = self.noise_scheduler.timesteps.tolist()
                for i, t_val in enumerate(timesteps_list):
                    timestep = torch.full((B,), t_val, dtype=torch.long, device=device)

                    # Reconstruct latent image from patches
                    current_latent = self.dit.unpatchify(current_patches)

                    # DiT forward pass (unconditional - no mask_token)
                    pred_patches, _ = self.dit.forward_with_patch_mask(
                        current_latent, None, timestep=timestep
                    )

                    # DDIM step on ALL patches
                    updated_patches = self.noise_scheduler.step(
                        pred_patches, t_val, current_patches
                    )

                    # RePaint: replace unmasked patches with appropriately-noised
                    # clean values for the previous timestep
                    if i < len(timesteps_list) - 1:
                        t_prev = timesteps_list[i + 1]
                    else:
                        t_prev = 0

                    if t_prev > 0:
                        repaint_noise = torch.randn_like(clean_patches)
                        t_prev_tensor = torch.full(
                            (B,), t_prev, dtype=torch.long, device=device
                        )
                        noised_clean = self.noise_scheduler.add_noise(
                            clean_patches, repaint_noise, t_prev_tensor
                        )
                        current_patches = torch.where(
                            latent_mask.unsqueeze(-1), updated_patches, noised_clean
                        )
                    else:
                        # Last step: unmasked get exact clean values
                        current_patches = torch.where(
                            latent_mask.unsqueeze(-1), updated_patches, clean_patches
                        )
        else:
            # Conditional: initialize masked=noise, unmasked=clean
            noise = torch.randn_like(clean_patches)
            current_patches = torch.where(
                latent_mask.unsqueeze(-1), noise, clean_patches
            )

            with torch.no_grad():
                for t in self.noise_scheduler.timesteps:
                    t_val = t.item()
                    timestep = torch.full(
                        (B,), t_val, dtype=torch.long, device=device
                    )

                    # Reconstruct latent image from patches
                    current_latent = self.dit.unpatchify(current_patches)

                    # DiT forward pass WITHOUT mask_token (consistent with training)
                    pred_patches, _ = self.dit.forward_with_patch_mask(
                        current_latent, None, timestep=timestep
                    )

                    # DDIM step on masked patches only
                    masked_current = current_patches[latent_mask]
                    masked_pred = pred_patches[latent_mask]
                    updated_masked = self.noise_scheduler.step(
                        masked_pred, t_val, masked_current
                    )

                    # Repaint: update only masked patches
                    current_patches = current_patches.clone()
                    current_patches[latent_mask] = updated_masked

                    # Ensure unmasked patches stay clean
                    current_patches = torch.where(
                        latent_mask.unsqueeze(-1), current_patches, clean_patches
                    )

        return self.dit.unpatchify(current_patches)

    # ------------------------------------------------------------------
    # Pixel-space patchify/unpatchify (for compatibility)
    # ------------------------------------------------------------------
    def _pixel_patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert pixel images to 16x16 patches for compatibility."""
        p = 16  # Standard pixel patch size
        B, C, H, W = imgs.shape
        h = H // p
        w = W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def _pixel_unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert 16x16 pixel patches back to image."""
        p = 16
        h = H // p
        w = W // p
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Pixel-space patchify for compatibility with existing code."""
        return self._pixel_patchify(imgs)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel-space unpatchify for compatibility with existing code.

        Note: Requires knowing the pixel canvas dimensions. Uses stored size.
        """
        if self._pixel_canvas_size is not None:
            H, W = self._pixel_canvas_size
        else:
            # Infer from latent dims
            cf = self.vae.compression_factor
            H = self.dit.image_size[0] * cf
            W = self.dit.image_size[1] * cf
        return self._pixel_unpatchify(x, H, W)

    # ------------------------------------------------------------------
    # State dict management (only save DiT, not frozen VAE)
    # ------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        """Return only the DiT state dict (VAE is frozen/separate)."""
        return self.dit.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into the DiT only."""
        return self.dit.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse=True):
        """Only return DiT parameters (VAE is frozen)."""
        return self.dit.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        """Only return DiT named parameters."""
        return self.dit.named_parameters(prefix=prefix, recurse=recurse)

    def train(self, mode=True):
        """Set DiT to train mode, VAE stays frozen."""
        self.dit.train(mode)
        self.vae.eval()
        return self

    def eval(self):
        """Set everything to eval mode."""
        self.dit.eval()
        self.vae.eval()
        return self
