"""
world_model_utils.py

Common utility functions for world model implementations.
Provides shared functionality for model training, history management, and tensor operations.
"""

import numpy as np
import torch
import torch.nn as nn


def to_model_tensor(frame_np, device):
    """
    Convert frame to properly scaled tensor for model input.

    Args:
        frame_np: HxWx3 RGB frame (uint8 or float)
        device: torch.device to place tensor on

    Returns:
        torch.Tensor: [1,3,H,W] tensor in range [0,1]
    """
    if frame_np.dtype == np.uint8:
        img = frame_np.astype(np.float32) / 255.0
    else:
        img = np.clip(frame_np.astype(np.float32), 0.0, 1.0)
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)


def create_param_groups(model, weight_decay):
    """
    Create parameter groups for AdamW optimizer.

    Separates parameters into two groups:
    1. Parameters with weight decay (weights in Linear/Conv layers)
    2. Parameters without weight decay (biases, LayerNorm params)

    Args:
        model: PyTorch model
        weight_decay: float, weight decay value for group 1

    Returns:
        List of parameter groups for optimizer
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Exclude bias and LayerNorm parameters from weight decay
        # Common patterns: 'bias', 'norm', 'ln', 'bn' (batch norm)
        if 'bias' in name or 'norm' in name.lower() or 'ln_' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def create_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps=100000, lr_min_ratio=0.01, lr_min=None):
    """
    Create warmup + cosine decay learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps (linear increase from 0 to base_lr)
        total_steps: Total training steps (default: 100000)
        lr_min_ratio: Minimum LR as ratio of base LR (default: 0.01), used if lr_min not specified
        lr_min: Absolute minimum LR (overrides lr_min_ratio if specified)

    Returns:
        Learning rate scheduler (torch.optim.lr_scheduler.LambdaLR)
    """
    # Calculate minimum learning rate
    base_lr = optimizer.param_groups[0]['lr']
    if lr_min is not None:
        # Use absolute minimum LR
        actual_lr_min = max(lr_min, 1e-8)
    else:
        # Use ratio-based minimum LR
        actual_lr_min = max(base_lr * lr_min_ratio, 1e-8)

    # Create warmup + cosine decay scheduler using LambdaLR
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay from 1 to lr_min_ratio
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
            # Scale between min_ratio and 1.0
            min_ratio = actual_lr_min / base_lr
            return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_reduce_on_plateau_scheduler(optimizer, patience=5, factor=0.1, min_lr=1e-7):
    """
    Create ReduceLROnPlateau scheduler for validation-driven LR adjustment.
    Used when training length is unknown (e.g., train-until-divergence mode).

    Unlike cosine schedulers, this adapts based on validation loss and doesn't
    require knowing total_steps ahead of time.

    Args:
        optimizer: The optimizer to schedule
        patience: Number of validation checks with no improvement before LR reduction
        factor: Factor to reduce LR by (new_lr = old_lr * factor)
        min_lr: Minimum LR floor

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience,
        min_lr=min_lr
    )


def create_constant_lr_scheduler(optimizer):
    """
    Create a scheduler that keeps LR constant (no decay).

    Used for plateau sweep mode where LR adaptation happens through
    sweep-triggered restarts, not through scheduler decay.

    Args:
        optimizer: The optimizer to schedule

    Returns:
        Learning rate scheduler (torch.optim.lr_scheduler.LambdaLR)
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def create_resume_scheduler(optimizer, warmup_from_lr, warmup_to_lr, warmup_steps,
                             decay_to_lr, total_steps):
    """
    Create scheduler for resuming training with gradual LR warmup.

    This scheduler:
    1. Warmup: Ramps from warmup_from_lr to warmup_to_lr over warmup_steps
    2. Decay: Cosine decay from warmup_to_lr to decay_to_lr over remaining steps

    Used when resuming from a checkpoint where the LR needs to "jump up" to match
    the global schedule - the warmup gives momentum time to adapt.

    Args:
        optimizer: The optimizer to schedule (should have LR set to warmup_from_lr)
        warmup_from_lr: Starting LR (typically checkpoint's ending LR)
        warmup_to_lr: Target LR after warmup (computed from global schedule)
        warmup_steps: Number of steps to ramp from warmup_from_lr to warmup_to_lr
        decay_to_lr: Final minimum LR after decay
        total_steps: Total training steps for this session

    Returns:
        Learning rate scheduler (torch.optim.lr_scheduler.LambdaLR)
    """
    decay_steps = max(1, total_steps - warmup_steps)

    # The optimizer's current LR should be set to warmup_from_lr before calling this
    base_lr = optimizer.param_groups[0]['lr']

    def lr_lambda(step):
        if step < warmup_steps and warmup_steps > 0:
            # Linear warmup from warmup_from_lr to warmup_to_lr
            progress = float(step) / float(warmup_steps)
            current_lr = warmup_from_lr + progress * (warmup_to_lr - warmup_from_lr)
        else:
            # Cosine decay from warmup_to_lr to decay_to_lr
            decay_progress = float(step - warmup_steps) / float(decay_steps)
            decay_progress = min(decay_progress, 1.0)  # Clamp to [0, 1]
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * decay_progress))
            current_lr = decay_to_lr + (warmup_to_lr - decay_to_lr) * cosine_decay

        # Return multiplier (scheduler applies this to base_lr)
        return current_lr / base_lr if base_lr > 0 else 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maintain_history_window(history_list, max_size):
    """
    Keep history list at reasonable size by trimming from the front.

    Args:
        history_list: List to trim
        max_size: Maximum number of elements to keep

    Returns:
        Trimmed list (or original if under max_size)
    """
    if len(history_list) > max_size:
        return history_list[-max_size:]
    return history_list


def _detect_block_depth(state_dict, prefix):
    """Detect the number of transformer blocks in a state dict by prefix (e.g. 'blocks')."""
    indices = set()
    for key in state_dict:
        if key.startswith(prefix + '.'):
            parts = key[len(prefix) + 1:].split('.', 1)
            if parts[0].isdigit():
                indices.add(int(parts[0]))
    return max(indices) + 1 if indices else 0


def _zero_init_block_residual(block):
    """Zero-initialize output projections so the block acts as identity via residual."""
    nn.init.zeros_(block.attn.proj.weight)
    nn.init.zeros_(block.attn.proj.bias)
    nn.init.zeros_(block.mlp.fc2.weight)
    nn.init.zeros_(block.mlp.fc2.bias)


def load_state_dict_with_depth_growth(model, state_dict, zero_init_new_blocks=True):
    """
    Load a state dict into a model that may have a different depth (number of layers).

    Matching block weights (by index) are loaded. Extra blocks in the model
    get zero-init residual paths so they approximate identity initially,
    preserving the prediction head's trained input distribution.

    Works for decoder-only (single 'blocks' stack), encoder-decoder
    (separate 'blocks' encoder + 'decoder_blocks' decoder), and DiT
    (single 'blocks' stack with adaLN_modulation parameters).

    Args:
        model: The target model (DecoderOnlyViT, MaskedAutoencoderViT, or DiffusionViT)
        state_dict: The saved state dict to load from
        zero_init_new_blocks: If True, zero-init residual paths of new blocks

    Returns:
        dict with depth growth info:
            'depth_changed': bool, whether any depth mismatch was detected
            'blocks_saved': int, saved encoder/main block count
            'blocks_current': int, current encoder/main block count
            'decoder_blocks_saved': int, saved decoder block count (encoder-decoder only)
            'decoder_blocks_current': int, current decoder block count (encoder-decoder only)
            'loaded_keys': int, number of keys loaded
            'skipped_keys': list of str, keys that couldn't be loaded
    """
    model_state = model.state_dict()

    saved_blocks_depth = _detect_block_depth(state_dict, 'blocks')
    current_blocks_depth = len(model.blocks)

    # For decoder-only, decoder_blocks is an alias for blocks, so state dict
    # has duplicate keys under both prefixes pointing to the same parameters.
    # Detect true encoder-decoder by checking if decoder_blocks is a separate module.
    has_separate_decoder = (
        hasattr(model, 'decoder_blocks')
        and model.decoder_blocks is not model.blocks
    )

    saved_decoder_depth = 0
    current_decoder_depth = 0
    if has_separate_decoder:
        saved_decoder_depth = _detect_block_depth(state_dict, 'decoder_blocks')
        current_decoder_depth = len(model.decoder_blocks)

    # Build filtered state dict: only include keys present in model with matching shape
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(f"{key}: shape {value.shape} -> {model_state[key].shape}")
        else:
            skipped.append(key)

    # Load matching weights
    model.load_state_dict(filtered, strict=False)

    # Zero-init residual paths for new main/encoder blocks
    if zero_init_new_blocks and current_blocks_depth > saved_blocks_depth:
        for i in range(saved_blocks_depth, current_blocks_depth):
            _zero_init_block_residual(model.blocks[i])

    # Zero-init residual paths for new decoder blocks (encoder-decoder only)
    if zero_init_new_blocks and has_separate_decoder and current_decoder_depth > saved_decoder_depth:
        for i in range(saved_decoder_depth, current_decoder_depth):
            _zero_init_block_residual(model.decoder_blocks[i])

    depth_changed = (
        current_blocks_depth != saved_blocks_depth
        or (has_separate_decoder and current_decoder_depth != saved_decoder_depth)
    )

    # Log DiT-specific parameters for clarity
    dit_keys = [k for k in skipped if any(p in k for p in ['adaLN_modulation', 'time_embed', 'final_modulation'])]
    if dit_keys:
        print(f"  DiT-specific parameters not in checkpoint ({len(dit_keys)} keys): "
              f"{', '.join(dit_keys[:5])}{'...' if len(dit_keys) > 5 else ''}")

    return {
        'depth_changed': depth_changed,
        'blocks_saved': saved_blocks_depth,
        'blocks_current': current_blocks_depth,
        'decoder_blocks_saved': saved_decoder_depth,
        'decoder_blocks_current': current_decoder_depth,
        'loaded_keys': len(filtered),
        'skipped_keys': skipped,
    }
