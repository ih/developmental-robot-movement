"""
PyTorch Dataset for pre-built canvas data.

Enables efficient parallel batch loading with DataLoader for training speedup.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PrebuiltCanvasDataset(Dataset):
    """
    PyTorch Dataset for pre-built training canvases.

    Uses pre-built canvas cache to avoid expensive on-the-fly canvas construction.
    Designed for use with DataLoader's multi-worker parallel loading.

    Args:
        canvas_cache: Dictionary mapping frame_idx -> {'canvas': np.array, ...}
        frame_indices: List of valid frame indices to sample from

    Returns:
        Tuple of (canvas_numpy, frame_idx) for each sample
    """

    def __init__(self, canvas_cache, frame_indices):
        """
        Initialize dataset with pre-built canvas cache.

        Args:
            canvas_cache: Dict {frame_idx -> {'canvas': np.array HxWx3, ...}}
            frame_indices: List of valid frame indices to include in dataset
        """
        self.canvas_cache = canvas_cache
        self.frame_indices = frame_indices

        # Validate all indices exist in cache
        missing = [idx for idx in frame_indices if idx not in canvas_cache]
        if missing:
            raise ValueError(f"Canvas cache missing {len(missing)} indices: {missing[:10]}...")

    def __len__(self):
        """Return number of samples in dataset"""
        return len(self.frame_indices)

    def __getitem__(self, idx):
        """
        Get a single canvas sample.

        Args:
            idx: Dataset index (0 to len-1)

        Returns:
            Tuple of (canvas_numpy, frame_idx)
            - canvas_numpy: np.array of shape (H, W, 3) uint8
            - frame_idx: Original frame index in session
        """
        frame_idx = self.frame_indices[idx]
        canvas_data = self.canvas_cache[frame_idx]
        canvas = canvas_data['canvas']

        return canvas, frame_idx


class CanvasCollateFn:
    """
    Picklable collate function for batching pre-built canvases.

    Class-based implementation is required for Windows multiprocessing compatibility,
    as local functions cannot be pickled.
    """

    def __init__(self, config, device='cuda', transfer_to_device=True):
        """
        Initialize collate function with config.

        Args:
            config: AutoencoderConcatPredictorWorldModelConfig
            device: Target device for tensors
            transfer_to_device: If False, keep tensors on CPU for manual transfer (Phase 4)
        """
        self.config = config
        self.device = device
        self.transfer_to_device = transfer_to_device

    def __call__(self, batch):
        """
        Collate a batch of canvas samples.

        Args:
            batch: List of (canvas_numpy, frame_idx) tuples from dataset

        Returns:
            Tuple of (canvas_tensor, patch_mask, frame_indices)
        """
        from models.autoencoder_concat_predictor import (
            canvas_to_tensor,
            compute_randomized_patch_mask_for_last_slot_gpu,
        )

        # Unpack batch
        canvases, frame_indices = zip(*batch)
        batch_size = len(canvases)

        # Stack canvases: [B, H, W, 3]
        canvas_batch = np.stack(canvases, axis=0)

        # Convert to tensor: [B, 3, H, W]
        canvas_tensor = canvas_to_tensor(canvas_batch, batch_size=batch_size)

        # Conditionally transfer to device (Phase 4: skip transfer for manual stream management)
        if self.transfer_to_device:
            canvas_tensor = canvas_tensor.to(self.device)

        # Generate patch masks on GPU (Phase 3 optimization)
        canvas_height, canvas_width = canvas_tensor.shape[-2:]
        num_frames = self.config.CANVAS_HISTORY_SIZE

        # For Phase 4 with manual transfer, generate mask on CPU first
        mask_device = self.device if self.transfer_to_device else 'cpu'

        patch_mask = compute_randomized_patch_mask_for_last_slot_gpu(
            img_size=(canvas_height, canvas_width),
            patch_size=self.config.PATCH_SIZE,
            num_frame_slots=num_frames,
            sep_width=self.config.SEPARATOR_WIDTH,
            mask_ratio_min=getattr(self.config, 'MASK_RATIO_MIN', 1.0),
            mask_ratio_max=getattr(self.config, 'MASK_RATIO_MAX', 1.0),
            batch_size=batch_size,
            device=mask_device,
        )

        return canvas_tensor, patch_mask, list(frame_indices)


# Keep old function signature for backward compatibility
def canvas_collate_fn(batch, config, device='cuda'):
    """
    Legacy function signature - creates CanvasCollateFn instance.

    For direct use, prefer creating CanvasCollateFn directly.
    """
    collate_obj = CanvasCollateFn(config, device)
    return collate_obj(batch)


def create_canvas_dataloader(canvas_cache, frame_indices, batch_size, config,
                             device='cuda', num_workers=2, shuffle=True,
                             pin_memory=True, persistent_workers=None,
                             transfer_to_device=True):
    """
    Create a DataLoader for pre-built canvases with optimized settings.

    Args:
        canvas_cache: Dictionary mapping frame_idx -> canvas data
        frame_indices: List of valid frame indices to load
        batch_size: Batch size for training
        config: AutoencoderConcatPredictorWorldModelConfig
        device: Target device ('cuda' or 'cpu')
        num_workers: Number of parallel data loading workers (default: 2)
                    - 0: Single-process loading (safe but slower)
                    - 2-4: Multi-process loading (faster, recommended for Windows)
                    - 4-8: High parallelism (recommended for Linux)
        shuffle: Whether to shuffle data (default: True)
        pin_memory: Use pinned memory for faster GPU transfer (default: True)
        persistent_workers: Keep workers alive between epochs (default: auto-detect)
                          - True: Faster for multiple epochs (avoids spawn overhead)
                          - False: Lower memory usage
                          - None: Auto (True if num_workers > 0, False otherwise)
        transfer_to_device: If True, collate_fn transfers to device; if False, caller
                          handles transfer manually with CUDA streams (Phase 4)

    Returns:
        torch.utils.data.DataLoader configured for efficient batch loading
    """
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = PrebuiltCanvasDataset(canvas_cache, frame_indices)

    # Auto-detect persistent_workers setting
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # Create picklable collate function (required for Windows multiprocessing)
    collate_fn = CanvasCollateFn(config, device, transfer_to_device=transfer_to_device)

    # Create DataLoader with optimized settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and (device == 'cuda'),
        persistent_workers=persistent_workers,
        drop_last=False,  # Keep incomplete final batch
    )

    return dataloader
