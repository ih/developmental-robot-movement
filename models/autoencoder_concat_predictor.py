
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
     - Toroidal dot actions: 'action' = 0 -> RED, 'action' = 1 -> GREEN
     - Unknown actions: BLUE
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

    # Toroidal dot binary action space
    if action.get('action', None) in (0, 1):
        return (255, 0, 0) if action['action'] == 0 else (0, 255, 0)

    # Unknown action space
    return (0, 0, 255)

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

# ------------------------------
# MAE targeted-mask forward (adds a helper around the provided ViT MAE)
# ------------------------------

class TargetedMAEWrapper(MaskedAutoencoderViT):
    """
    Extends MaskedAutoencoderViT with a method to pass a *custom* binary mask over patches.
    True in `patch_mask` means the patch is hidden from the encoder and must be inpainted.
    """
    def train_on_canvas(self, canvas_tensor: torch.Tensor, patch_mask: torch.Tensor, optimizer):
        """
        Train the autoencoder on a canvas with targeted masking.

        Args:
            canvas_tensor: [1, 3, H, W] canvas image tensor
            patch_mask: [1, num_patches] boolean tensor; True = masked
            optimizer: Optimizer for updating weights

        Returns:
            loss: Reconstruction loss value
        """
        # Forward pass with patch masking
        pred_patches, _ = self.forward_with_patch_mask(canvas_tensor, patch_mask)

        # Compute reconstruction loss only on masked patches
        # Unpatchify to get full image prediction
        img_pred = self.unpatchify(pred_patches)  # [1, 3, H, W]

        # MSE loss on masked regions
        loss = torch.nn.functional.mse_loss(img_pred, canvas_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def forward_with_patch_mask(self, imgs: torch.Tensor, patch_mask: torch.Tensor):
        """
        imgs: [B,3,H,W]
        patch_mask: [B, num_patches] boolean tensor; True = masked.
        Returns predicted patches for masked locations (same shape as self.forward() decoder output),
        and latent features.
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

        # Predict patches
        pred = self.decoder_pred(dec_input)  # [B, 1+L, P^2*3]
        pred = pred[:, 1:, :]                # drop CLS
        return pred, latent  # pred is per-patch predictions in *original* order

# ------------------------------
# Training & Inference helpers
# ------------------------------

def canvas_to_tensor(canvas: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8 canvas -> [1,3,H,W] float tensor 0..1 using config.TRANSFORM if available."""
    pil = Image.fromarray(canvas)
    if TRANSFORM is not None:
        t = TRANSFORM(pil)              # [3,H,W], 0..1
    else:
        arr = np.asarray(pil).astype("float32") / 255.0
        arr = np.transpose(arr, (2,0,1))
        t = torch.from_numpy(arr)
    return t.unsqueeze(0)

@torch.no_grad()
def predict_next_frame_with_ae(
    model: TargetedMAEWrapper,
    frames: List[np.ndarray],
    actions: Optional[List[dict]],
    sep_width: int = 8,
) -> np.ndarray:
    """
    Build canvas with K history frames + one empty slot (the next frame position),
    then inpaint that last slot with a targeted masked forward pass.
    Returns the predicted next frame as an HxWx3 uint8 image.
    """
    K = len(frames) + 1  # we will create K slots: frames[:-1] history + 1 blank (future)
    H, W = 224, 224
    # Build a temporary list with the last slot as zeros (or mid-gray)
    future_blank = np.zeros((H, W, 3), dtype=np.uint8)
    canv = build_canvas(frames + [future_blank], actions, frame_size=(H,W), sep_width=sep_width)

    # Convert to tensor and prepare patch mask that covers the *last* slot
    x = canvas_to_tensor(canv).to(next(model.parameters()).device)  # [1,3,H,W] after TRANSFORM
    # The model's patch size is model.patch_size (int)
    patch = int(model.patch_size)
    # H_total and W_total are determined by TRANSFORM: ensure they equal canvas size
    Htot, Wtot = x.shape[-2], x.shape[-1]
    patch_mask = compute_patch_mask_for_last_slot(
        img_size=(Htot, Wtot),
        patch_size=patch,
        K=K,
        sep_width=sep_width
    ).to(x.device)

    # Inference: targeted inpainting
    pred_patches, _ = model.forward_with_patch_mask(x, patch_mask)  # [1, L, P^2*3]
    # Unpatchify helper from original MAE (we can reuse model.unpatchify via a tiny wrapper)
    # We'll mimic the decode() pathway to get the full canvas image predicted
    # First, reconstruct the full canvas image from predicted patches
    img_pred = model.unpatchify(pred_patches)  # [1, 3, Htot, Wtot]
    img_np = (img_pred.clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)

    # Extract the last slot region (without separators)
    W0 = (Wtot - (K - 1) * sep_width) // K
    x0 = (K - 1) * (W0 + sep_width)
    x1 = x0 + W0
    next_frame = img_np[:, x0:x1, :]
    # Resize back to (H,W) if needed (should already match)
    if next_frame.shape[0] != H or next_frame.shape[1] != W:
        next_frame = _ensure_hw(next_frame, (H,W))
    return next_frame

# ------------------------------
# Minimal usage example (pseudo)
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize MAE
    mae = TargetedMAEWrapper(image_size=224, patch_size=16, embed_dim=256, decoder_embed_dim=128).to(device)
    mae.eval()

    # Fake data for a quick smoke test (3 history frames, 2 actions between them)
    Khist = 3
    H, W = 224, 224
    frames = [(np.random.rand(H, W, 3) * 255).astype(np.uint8) for _ in range(Khist)]
    actions = [{'motor_right': 0.0}, {'motor_right': 0.12}]  # len = Khist-1
    # Predict next
    pred = predict_next_frame_with_ae(mae, frames, actions, sep_width=8)
    print("Pred next frame:", pred.shape, pred.dtype)
