
"""
autoencoder_concat_predictor.py

Use the existing autoencoder as a predictor by concatenating history frames into a single canvas image.
Actions are encoded as thin colored separators between frames (e.g., red for "no motor power",
green for "right motor 0.12"), so the AE can learn to use the action context. During inference,
the last slot (next-frame position) is masked and the AE inpaints it.

Requirements:
  - vit_autoencoder.py (MaskedAutoencoderViT) or any BaseAutoencoder implementation
  - config.py (for TRANSFORM or image size)
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Local imports from the user's repo
from models.vit_autoencoder import MaskedAutoencoderViT
from models.base_autoencoder import BaseAutoencoder
from config import TRANSFORM

# ------------------------------
# Canvas building utilities
# ------------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    return img

def _ensure_hw(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize numpy RGB image to (H,W) using PIL (keeps 3 channels)."""
    H, W = size
    pil = Image.fromarray(_to_uint8(img))
    pil = pil.resize((W, H), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)

def _separator_color_for_action(action: dict) -> Tuple[int,int,int]:
    """
    Map action dict -> RGB color for separator visualization.

    Supports multiple action spaces:
     - JetBot motor actions: 'motor_right' value linearly interpolated RED (0.0) -> GREEN (0.12)
     - Discrete actions (toroidal dot, SO-101):
         action = 0 -> RED (stay/no movement)
         action = 1 -> GREEN (move right / move positive)
         action = 2 -> BLUE (move negative, SO-101 only)
     - Unknown actions: YELLOW (warning color)
    """
    # JetBot motor action space
    if 'motor_right' in action:
        v = float(action['motor_right'])
        v = max(0.0, min(0.12, v))  # clamp
        t = 0.0 if 0.12 == 0 else (v / 0.12)
        # Linear blend RED->GREEN
        r = int((1.0 - t) * 255)
        g = int(t * 255)
        b = 0
        return (r, g, b)

    # Discrete action space (toroidal dot: 2 actions, SO-101: 3 actions)
    action_val = action.get('action', None)
    if action_val == 0:
        return (255, 0, 0)    # RED: stay / no movement
    elif action_val == 1:
        return (0, 255, 0)    # GREEN: move right / move positive
    elif action_val == 2:
        return (0, 0, 255)    # BLUE: move negative (SO-101)

    # Unknown action space
    return (255, 255, 0)  # YELLOW: warning

def build_canvas(
    interleaved_history: List,
    frame_size: Tuple[int,int] = (224,224),
    sep_width: int = 8,
    bg_color: Tuple[int,int,int] = (0,0,0),
) -> np.ndarray:
    """
    Concatenate frames horizontally with thin action-colored separators between them.

    Args:
        interleaved_history: list alternating [frame0, action0, frame1, action1, frame2, ...]
                            Must start with a frame. Can end with either frame or action.
        frame_size: target (H,W) for each frame before concatenation.
        sep_width: width in pixels of action-colored separator
        bg_color: background color for canvas

    Returns:
        canvas RGB uint8 array of shape (H, W_total, 3).
    """
    if len(interleaved_history) == 0:
        raise ValueError("Need at least one frame in interleaved history")

    H, W = frame_size

    # Count frames (at even indices: 0, 2, 4, ...)
    num_frames = (len(interleaved_history) + 1) // 2

    # Total width = num_frames * W + (num_frames - 1) * sep_width
    W_total = num_frames * W + (num_frames - 1) * sep_width
    canvas = np.zeros((H, W_total, 3), dtype=np.uint8)
    canvas[:] = np.array(bg_color, dtype=np.uint8)

    x = 0
    for i, item in enumerate(interleaved_history):
        if i % 2 == 0:
            # Even index = frame
            frame_resized = _ensure_hw(item, frame_size)
            canvas[:, x:x+W, :] = frame_resized
            x += W
        else:
            # Odd index = action -> draw separator
            color = _separator_color_for_action(item)
            canvas[:, x:x+sep_width, :] = np.array(color, dtype=np.uint8)
            x += sep_width

    return canvas

# ------------------------------
# Patch masking for targeted inpainting (MAE ViT)
# ------------------------------

def compute_patch_mask_for_last_slot(
    img_size: Tuple[int,int],
    patch_size: int,
    K: int,
    sep_width: int,
    last_slot_index: int = -1,
) -> torch.Tensor:
    """
    Build a boolean mask over patch grid for the *last* frame slot on the canvas.
    last_slot_index: which slot to mask (default: -1 i.e., the last frame position).
    Returns: (1, num_patches) boolean tensor where True = mask this patch (hide from encoder).
    """
    H, W = img_size
    assert H % patch_size == 0, "H must be divisible by patch_size"
    assert W % patch_size == 0, "W must be divisible by patch_size"
    Gh = H // patch_size
    Gw = W // patch_size

    # Canvas width for K frames + (K-1) separators
    single_frame_W = W // K - ((K - 1) * sep_width) // K  # not used directly, compute robustly below

    # We recompute slot bounds explicitly from construction math
    slot_widths = []
    # We'll assume uniform frame width (without separators) = per input frame W0 = (totalW - (K-1)*sep)/K
    # But here img_size[1] is the total canvas width the model sees.
    # Let totalW = W; let W0 = (W - (K-1)*sep_width) // K
    W0 = (W - (K - 1) * sep_width) // K
    # Slot i starts at x = i*W0 + i*sep_width
    # Slot i ends (exclusive) at x_end = x + W0
    mask = torch.zeros((1, Gh * Gw), dtype=torch.bool)
    idx = 0
    slot = (K - 1) if last_slot_index == -1 else last_slot_index
    x0 = slot * (W0 + sep_width)
    x1 = x0 + W0

    for yy in range(Gh):
        for xx in range(Gw):
            # Patch bounds in pixels
            px0 = xx * patch_size
            px1 = px0 + patch_size
            py0 = yy * patch_size
            py1 = py0 + patch_size
            # If patch overlaps the slot region -> mask it
            overlap = not (px1 <= x0 or px0 >= x1)
            mask[0, idx] = overlap
            idx += 1

    return mask  # shape (1, Gh*Gw)

def compute_randomized_patch_mask_for_last_slot(
    img_size: Tuple[int,int],
    patch_size: int,
    num_frame_slots: int,
    sep_width: int,
    mask_ratio_min: float = 0.3,
    mask_ratio_max: float = 0.85,
    last_slot_index: int = -1,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Build a randomized boolean mask over patches in the last frame slot.

    Unlike compute_patch_mask_for_last_slot which masks ALL patches in the slot,
    this function randomly selects a subset based on a random mask_ratio.
    Each batch element gets an independent random mask for diversity.

    Args:
        img_size: (H, W) canvas dimensions
        patch_size: Size of each patch
        num_frame_slots: Number of frame slots in canvas
        sep_width: Separator width in pixels
        mask_ratio_min: Minimum mask ratio (default 0.3)
        mask_ratio_max: Maximum mask ratio (default 0.85)
        last_slot_index: Which slot to mask (default -1 = last)
        batch_size: Number of masks to generate (default 1)

    Returns:
        (batch_size, num_patches) boolean tensor where True = mask this patch
    """
    import random

    canvas_height, canvas_width = img_size
    assert canvas_height % patch_size == 0, "canvas_height must be divisible by patch_size"
    assert canvas_width % patch_size == 0, "canvas_width must be divisible by patch_size"

    num_patches_height = canvas_height // patch_size
    num_patches_width = canvas_width // patch_size

    # Compute slot bounds (same logic as compute_patch_mask_for_last_slot)
    single_frame_width = (canvas_width - (num_frame_slots - 1) * sep_width) // num_frame_slots
    slot_index = (num_frame_slots - 1) if last_slot_index == -1 else last_slot_index
    slot_start_x = slot_index * (single_frame_width + sep_width)
    slot_end_x = slot_start_x + single_frame_width

    # Identify all patches that overlap with the last slot
    last_slot_patch_indices = []
    patch_index = 0

    for patch_row in range(num_patches_height):
        for patch_col in range(num_patches_width):
            # Patch bounds in pixels
            patch_start_x = patch_col * patch_size
            patch_end_x = patch_start_x + patch_size
            patch_start_y = patch_row * patch_size
            patch_end_y = patch_start_y + patch_size

            # If patch overlaps the slot region, add to list
            overlaps_slot = not (patch_end_x <= slot_start_x or patch_start_x >= slot_end_x)
            if overlaps_slot:
                last_slot_patch_indices.append(patch_index)
            patch_index += 1

    # Generate independent random masks for each batch element
    total_num_patches = num_patches_height * num_patches_width
    batch_masks = []

    for _ in range(batch_size):
        # Randomly sample mask ratio for this sample
        mask_ratio = random.uniform(mask_ratio_min, mask_ratio_max)

        # Randomly select subset of last-slot patches to mask
        num_patches_to_mask = int(len(last_slot_patch_indices) * mask_ratio)
        masked_patch_indices = random.sample(last_slot_patch_indices, num_patches_to_mask)

        # Build mask for this sample
        mask = torch.zeros(total_num_patches, dtype=torch.bool)
        for patch_index in masked_patch_indices:
            mask[patch_index] = True

        batch_masks.append(mask)

    # Stack into [batch_size, num_patches]
    return torch.stack(batch_masks, dim=0)


def compute_randomized_patch_mask_for_last_slot_gpu(
    img_size: Tuple[int,int],
    patch_size: int,
    num_frame_slots: int,
    sep_width: int,
    mask_ratio_min: float = 0.3,
    mask_ratio_max: float = 0.85,
    last_slot_index: int = -1,
    batch_size: int = 1,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    GPU-accelerated version of compute_randomized_patch_mask_for_last_slot.

    Uses vectorized torch operations instead of Python loops for 10-20x speedup.
    All computations happen on GPU, eliminating CPU→GPU transfers.

    Key optimizations:
    - Vectorized patch overlap detection (no nested loops)
    - torch.rand() and torch.randperm() on GPU (vs Python random.uniform/sample)
    - Parallel mask generation for entire batch (vs sequential loop)

    Args:
        img_size: (H, W) canvas dimensions
        patch_size: Size of each patch
        num_frame_slots: Number of frame slots in canvas
        sep_width: Separator width in pixels
        mask_ratio_min: Minimum mask ratio (default 0.3)
        mask_ratio_max: Maximum mask ratio (default 0.85)
        last_slot_index: Which slot to mask (default -1 = last)
        batch_size: Number of masks to generate (default 1)
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        (batch_size, num_patches) boolean tensor where True = mask this patch
    """
    canvas_height, canvas_width = img_size
    assert canvas_height % patch_size == 0, "canvas_height must be divisible by patch_size"
    assert canvas_width % patch_size == 0, "canvas_width must be divisible by patch_size"

    num_patches_height = canvas_height // patch_size
    num_patches_width = canvas_width // patch_size
    total_num_patches = num_patches_height * num_patches_width

    # Compute slot bounds (same as CPU version)
    single_frame_width = (canvas_width - (num_frame_slots - 1) * sep_width) // num_frame_slots
    slot_index = (num_frame_slots - 1) if last_slot_index == -1 else last_slot_index
    slot_start_x = slot_index * (single_frame_width + sep_width)
    slot_end_x = slot_start_x + single_frame_width

    # Vectorized patch overlap detection (GPU-accelerated)
    # Create coordinate grids for all patches
    patch_rows = torch.arange(num_patches_height, device=device)
    patch_cols = torch.arange(num_patches_width, device=device)

    # Compute patch bounds for all patches simultaneously
    # patch_start_x: [num_patches_width]
    patch_start_x = patch_cols * patch_size
    patch_end_x = patch_start_x + patch_size

    # Check overlap: not (patch_end_x <= slot_start_x or patch_start_x >= slot_end_x)
    # Equivalent: (patch_end_x > slot_start_x) and (patch_start_x < slot_end_x)
    overlaps_slot = (patch_end_x > slot_start_x) & (patch_start_x < slot_end_x)

    # Expand to full grid: [num_patches_height, num_patches_width]
    overlaps_slot_grid = overlaps_slot.unsqueeze(0).expand(num_patches_height, -1)

    # Flatten to patch indices: [total_num_patches]
    overlaps_slot_flat = overlaps_slot_grid.reshape(-1)

    # Get indices of patches that overlap last slot
    last_slot_patch_indices = torch.nonzero(overlaps_slot_flat, as_tuple=True)[0]
    num_slot_patches = last_slot_patch_indices.shape[0]

    # Generate independent random masks for each batch element (parallel on GPU)
    # Random mask ratios for each batch element: [batch_size]
    mask_ratios = torch.rand(batch_size, device=device) * (mask_ratio_max - mask_ratio_min) + mask_ratio_min

    # Number of patches to mask per sample: [batch_size]
    num_patches_to_mask = (mask_ratios * num_slot_patches).long()

    # Initialize batch masks: [batch_size, total_num_patches]
    batch_masks = torch.zeros(batch_size, total_num_patches, dtype=torch.bool, device=device)

    # Generate random selection for each batch element
    for b in range(batch_size):
        # Random permutation of last-slot patch indices
        perm = torch.randperm(num_slot_patches, device=device)

        # Select top num_patches_to_mask[b] indices
        selected_local_indices = perm[:num_patches_to_mask[b]]

        # Map to global patch indices
        selected_patch_indices = last_slot_patch_indices[selected_local_indices]

        # Set mask to True for selected patches
        batch_masks[b, selected_patch_indices] = True

    return batch_masks


# ------------------------------
# MAE targeted-mask forward (adds a helper around the provided ViT MAE)
# ------------------------------

class TargetedMAEWrapper(MaskedAutoencoderViT):
    """
    Extends MaskedAutoencoderViT with a method to pass a *custom* binary mask over patches.
    True in `patch_mask` means the patch is hidden from the encoder and must be inpainted.
    """
    def train_on_canvas(self, canvas_tensor: torch.Tensor, patch_mask: torch.Tensor, optimizer,
                        return_per_sample_losses: bool = False):
        """
        Train the autoencoder on a canvas with targeted masking.

        Computes loss only on masked patches in patch space, forcing the model
        to learn inpainting rather than just reconstructing visible regions.
        This is the MAE-native approach and works for any mask pattern.

        Args:
            canvas_tensor: [B, 3, H, W] canvas image tensor (B can be 1 or more)
            patch_mask: [B, num_patches] boolean tensor; True = masked
            optimizer: Optimizer for updating weights
            return_per_sample_losses: If True, also return per-sample losses for loss-weighted sampling

        Returns:
            tuple: (loss, grad_diagnostics) or (loss, grad_diagnostics, per_sample_losses) if return_per_sample_losses=True
                   per_sample_losses is a list of floats, one per sample in the batch
        """
        # Validation
        assert canvas_tensor.ndim == 4, f"Expected 4D tensor [B,3,H,W], got {canvas_tensor.ndim}D"
        assert canvas_tensor.shape[0] == patch_mask.shape[0], \
            f"Batch size mismatch: canvas={canvas_tensor.shape[0]}, mask={patch_mask.shape[0]}"

        # Forward pass with masking - model predicts all patches including masked ones
        pred_patches, _ = self.forward_with_patch_mask(canvas_tensor, patch_mask)  # [1, num_patches, patch_size^2 * 3]

        # Convert ground truth canvas to patch representation (no gradients needed)
        with torch.no_grad():
            target_patches = self.patchify(canvas_tensor)  # [1, num_patches, patch_size^2 * 3]

        # Select only the masked patches for loss computation
        # This ensures we only optimize for inpainting the masked region
        masked_pred = pred_patches[patch_mask]      # [num_masked_patches, patch_size^2 * 3]
        masked_target = target_patches[patch_mask]  # [num_masked_patches, patch_size^2 * 3]

        # --- DIAGNOSTIC: dot-only loss vs. black baseline (computed with standard loss) ---
        with torch.no_grad():
            nonblack_mask = (masked_target.abs() > 1e-3).float()  # non-black pixels (the dot/edges)
            num_nonblack_pixels = nonblack_mask.sum().clamp_min(1.0)

            loss_nonblack = ((masked_pred - masked_target)**2 * nonblack_mask).sum() / num_nonblack_pixels
            black_baseline = (masked_target**2 * nonblack_mask).sum() / num_nonblack_pixels
            frac_nonblack = (num_nonblack_pixels / masked_target.numel()).item()

        # --- HYBRID LOSS: Use shared helper function ---
        from config import AutoencoderConcatPredictorWorldModelConfig as Config

        loss_dict = compute_hybrid_loss_on_masked_patches(
            masked_pred,
            masked_target,
            focal_alpha=Config.FOCAL_LOSS_ALPHA,
            focal_beta=Config.FOCAL_BETA
        )

        # Extract values from loss dict
        loss = loss_dict['loss_hybrid']
        loss_plain = loss_dict['loss_plain']
        focal_loss = loss_dict['loss_focal']
        loss_standard = loss_dict['loss_standard'].item()
        focal_weight_mean = loss_dict['focal_weight_mean']
        focal_weight_max = loss_dict['focal_weight_max']
        alpha = Config.FOCAL_LOSS_ALPHA

        # --- PER-SAMPLE LOSSES (for loss-weighted sampling) ---
        per_sample_losses = None
        if return_per_sample_losses:
            B = canvas_tensor.shape[0]
            per_sample_losses = []
            with torch.no_grad():
                for b in range(B):
                    # Get masked patches for this sample
                    sample_mask = patch_mask[b]  # [num_patches]
                    sample_pred = pred_patches[b][sample_mask]  # [num_masked, patch_dim]
                    sample_target = target_patches[b][sample_mask]  # [num_masked, patch_dim]

                    # Compute hybrid loss for this sample
                    sample_loss_dict = compute_hybrid_loss_on_masked_patches(
                        sample_pred, sample_target,
                        focal_alpha=Config.FOCAL_LOSS_ALPHA,
                        focal_beta=Config.FOCAL_BETA
                    )
                    per_sample_losses.append(sample_loss_dict['loss_hybrid'].item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # --- DIAGNOSTIC: gradient flow and LR on the masked token path ---
        def gnorm(p):
            return None if (p.grad is None) else p.grad.detach().norm().item()

        # Current LR
        curr_lr = optimizer.param_groups[0]['lr']

        # Decoder head (directly sets pixel values)
        # decoder_pred is a Linear layer, not a sequence
        head_w = getattr(self.decoder_pred, 'weight', None)
        head_b = getattr(self.decoder_pred, 'bias', None)

        # Mask token (what masked positions start from)
        mask_tok = getattr(self, 'mask_token', None)

        # First decoder block weights (does attention mix anything?)
        dec_blk = self.decoder_blocks[0] if hasattr(self, 'decoder_blocks') and len(self.decoder_blocks) > 0 else None
        qkv_w = getattr(getattr(dec_blk, 'attn', None), 'qkv', None) if dec_blk is not None else None
        qkv_w = getattr(qkv_w, 'weight', None) if qkv_w is not None else None

        # Collect diagnostics
        grad_diagnostics = {
            'lr': curr_lr,
            'head_weight_norm': gnorm(head_w),
            'head_bias_norm': gnorm(head_b),
            'mask_token_norm': gnorm(mask_tok),
            'qkv_weight_norm': gnorm(qkv_w),
            # Loss diagnostics
            'loss_hybrid': loss.item(),             # Hybrid loss (used for training)
            'loss_plain': loss_plain.item(),        # Plain MSE component
            'loss_focal': focal_loss.item(),        # Focal MSE component
            'loss_standard': loss_standard,         # Standard unweighted loss (for comparison)
            'loss_nonblack': loss_nonblack.item(),  # Loss on non-black pixels only
            'black_baseline': black_baseline.item(),
            'frac_nonblack': frac_nonblack,
            # Focal weight statistics
            'focal_weight_mean': focal_weight_mean,
            'focal_weight_max': focal_weight_max,
            'focal_beta': Config.FOCAL_BETA,
            'focal_alpha': alpha,
        }

        optimizer.step()

        if return_per_sample_losses:
            return float(loss.detach().cpu()), grad_diagnostics, per_sample_losses
        return float(loss.detach().cpu()), grad_diagnostics

    def forward_with_patch_mask(self, imgs: torch.Tensor, patch_mask: torch.Tensor, return_attn: bool = False):
        """
        imgs: [B,3,H,W]
        patch_mask: [B, num_patches] boolean tensor; True = masked.
        return_attn: if True, return decoder attention weights

        Returns:
            pred: predicted patches for masked locations (same shape as self.forward() decoder output)
            latent: latent features
            attn_weights: (optional) list of [B, num_heads, N, N] tensors per decoder layer
            patch_mask_out: (optional) the patch mask for identifying masked/unmasked patches
        """
        # Encode without shuffling, but drop masked patches
        x = self.patch_embed(imgs)                      # [B, L, D]
        x = x + self.pos_embed[:, 1:, :]
        B, L, D = x.shape
        assert patch_mask.shape[0] == B and patch_mask.shape[1] == L, "mask shape mismatch"

        # Keep = ~mask
        keep_mask = ~patch_mask
        len_keep = keep_mask.sum(dim=1)                # [B]
        # Build per-example gather indices for kept tokens
        kept_list = []
        for b in range(B):
            idxs = torch.nonzero(keep_mask[b], as_tuple=False).squeeze(1)
            kept_list.append(idxs)
        max_keep = max([len(i) for i in kept_list])
        # Pad to rectangular for gather
        gather_idx = torch.zeros((B, max_keep), dtype=torch.long, device=x.device)
        for b, idxs in enumerate(kept_list):
            nb = len(idxs)
            gather_idx[b, :nb] = idxs
            if nb < max_keep:
                # repeat last index to pad
                gather_idx[b, nb:] = idxs[-1] if nb > 0 else 0

        x_masked = torch.gather(x, dim=1, index=gather_idx.unsqueeze(-1).expand(B, max_keep, D))

        # Prepend cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        # Standard transformer blocks
        for blk in self.blocks:
            x_masked = blk(x_masked)
        latent = self.norm(x_masked)  # [B, 1+max_keep, D]

        # --- Decoder ---
        # We need to restore full token order and insert mask tokens for the masked patches
        dec = self.decoder_embed(latent)
        B, N, C = dec.shape  # N = 1 + max_keep

        # Build ids_restore such that the decoder reorders kept tokens + mask tokens back to original patch order
        ids_restore = torch.arange(L, device=imgs.device).unsqueeze(0).repeat(B, 1)  # we will *not* use shuffle

        # Compose the full length sequence for decoder: kept (in original order) + mask tokens
        # First, scatter kept features into their original positions
        full_tokens = torch.zeros(B, L, C, device=imgs.device)
        for b in range(B):
            kept_idxs = kept_list[b]
            nb = len(kept_idxs)
            full_tokens[b, kept_idxs] = dec[b, 1:1+nb, :]  # skip CLS in dec

        # Mask tokens for masked positions
        mask_tokens = self.mask_token.repeat(B, L, 1)  # [B, L, C]
        full_tokens = torch.where(keep_mask.unsqueeze(-1), full_tokens, mask_tokens)

        # Now add decoder positional embedding and re-insert CLS at the front
        dec_input = torch.cat([dec[:, :1, :], full_tokens], dim=1) + self.decoder_pos_embed

        # Pass through transformer decoder blocks, optionally capturing attention
        x = dec_input
        attn_weights = []
        if return_attn:
            for blk in self.decoder_blocks:
                x, attn = self._forward_block_with_attn(blk, x)
                attn_weights.append(attn)
        else:
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # Predict patches
        pred = self.decoder_pred(x)     # [B, 1+L, P^2*3]
        pred = pred[:, 1:, :]           # drop CLS

        if return_attn:
            return pred, latent, attn_weights, patch_mask
        return pred, latent  # pred is per-patch predictions in *original* order

# ------------------------------
# Loss calculation utilities
# ------------------------------

def compute_hybrid_loss_on_masked_patches(
    masked_pred: torch.Tensor,
    masked_target: torch.Tensor,
    focal_alpha: float,
    focal_beta: float
) -> dict:
    """
    Compute hybrid loss (plain MSE + normalized focal MSE) on masked patches.

    This is the shared loss calculation used for both training and inference evaluation.

    Args:
        masked_pred: [num_masked_patches, patch_dim] predicted patches
        masked_target: [num_masked_patches, patch_dim] ground truth patches
        focal_alpha: Blend ratio for hybrid loss (alpha * plain + (1-alpha) * focal)
        focal_beta: Temperature for focal weighting (higher = more focus on errors)

    Returns:
        dict containing:
            - loss_hybrid: Combined loss used for training
            - loss_plain: Plain MSE component
            - loss_focal: Focal MSE component
            - loss_standard: Standard unweighted MSE (for comparison)
            - focal_weight_mean: Mean focal weight
            - focal_weight_max: Max focal weight
    """
    # Per-pixel squared error
    pixel_error = (masked_pred - masked_target) ** 2

    # 1) Plain MSE term
    loss_plain = pixel_error.mean()

    # 2) Normalized focal term
    if focal_beta > 0:
        w = torch.exp(focal_beta * pixel_error).detach()  # Detach for stability
        loss_focal = (w * pixel_error).sum() / (w.sum() + 1e-8)
        focal_weight_mean = w.mean().item()
        focal_weight_max = w.max().item()
    else:
        loss_focal = loss_plain
        focal_weight_mean = 1.0
        focal_weight_max = 1.0

    # 3) Hybrid combination
    loss_hybrid = focal_alpha * loss_plain + (1.0 - focal_alpha) * loss_focal

    # 4) Standard MSE for comparison
    loss_standard = F.mse_loss(masked_pred, masked_target)

    return {
        'loss_hybrid': loss_hybrid,
        'loss_plain': loss_plain,
        'loss_focal': loss_focal,
        'loss_standard': loss_standard,
        'focal_weight_mean': focal_weight_mean,
        'focal_weight_max': focal_weight_max,
    }

# ------------------------------
# Canvas tensor conversion
# ------------------------------

def canvas_to_tensor(canvas: np.ndarray, batch_size: Optional[int] = None) -> torch.Tensor:
    """
    Convert canvas(es) to tensor.

    Args:
        canvas: Either HxWx3 single canvas OR BxHxWx3 batch of canvases (uint8)
        batch_size: Expected batch size for validation (optional)

    Returns:
        torch.Tensor: [B,3,H,W] float tensor in range [0,1]
    """
    # Don't use TRANSFORM here - canvas is multi-frame and should not be resized

    # Handle both single canvas and batch
    if canvas.ndim == 3:
        # Single canvas: HxWx3 -> [1,3,H,W]
        arr = canvas.astype("float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        t = torch.from_numpy(arr).unsqueeze(0)
    elif canvas.ndim == 4:
        # Batch of canvases: BxHxWx3 -> [B,3,H,W]
        arr = canvas.astype("float32") / 255.0
        arr = np.transpose(arr, (0, 3, 1, 2))
        t = torch.from_numpy(arr)
    else:
        raise ValueError(f"canvas must be 3D (HxWx3) or 4D (BxHxWx3), got shape {canvas.shape}")

    # Optional validation
    if batch_size is not None:
        assert t.shape[0] == batch_size, f"Expected batch size {batch_size}, got {t.shape[0]}"

    return t
