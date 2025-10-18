"""
world_model_utils.py

Common utility functions for world model implementations.
Provides shared functionality for model training, history management, and tensor operations.
"""

import numpy as np
import torch


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


def create_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps=100000, lr_min_ratio=0.01):
    """
    Create warmup + cosine decay learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps (linear increase from 0 to base_lr)
        total_steps: Total training steps (default: 100000)
        lr_min_ratio: Minimum LR as ratio of base LR (default: 0.01)

    Returns:
        Learning rate scheduler (torch.optim.lr_scheduler.LambdaLR)
    """
    # Calculate minimum learning rate
    base_lr = optimizer.param_groups[0]['lr']
    lr_min = max(base_lr * lr_min_ratio, 1e-6)

    # Create warmup + cosine decay scheduler using LambdaLR
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay from 1 to lr_min_ratio
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
            # Scale between lr_min_ratio and 1.0
            min_ratio = lr_min / base_lr
            return min_ratio + (1.0 - min_ratio) * cosine_decay

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
