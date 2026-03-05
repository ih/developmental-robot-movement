"""
Standalone VAE/RAE training script.

Trains a CanvasVAE or DINOv2 decoder on pre-built canvas data from a session.
The trained VAE is then used as the frozen encoder for DiT latent diffusion.

Usage:
    # Train custom VAE
    python train_vae.py --session-path saved/sessions/toroidal_dot/my_session \
        --vae-type custom --latent-channels 4 --compression-factor 8 --mode vae \
        --epochs 100 --batch-size 32

    # Train DINOv2 decoder
    python train_vae.py --session-path saved/sessions/toroidal_dot/my_session \
        --vae-type dinov2 --dinov2-variant vitb14 --epochs 50
"""

import argparse
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from session_explorer_lib import prebuild_all_canvases
from config import AutoencoderConcatPredictorWorldModelConfig as Config
from recording_reader import RecordingReader


class CanvasDataset(Dataset):
    """Simple dataset wrapping pre-built canvases."""

    def __init__(self, canvas_cache: dict):
        self.indices = sorted(canvas_cache.keys())
        self.canvas_cache = canvas_cache

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        canvas_np = self.canvas_cache[frame_idx]  # HxWx3 uint8
        # Convert to float tensor [3, H, W] in [0, 1]
        tensor = torch.from_numpy(canvas_np).float().permute(2, 0, 1) / 255.0
        return tensor


def load_session(session_path: str):
    """Load session and return observations, actions, action_space."""
    reader = RecordingReader(session_path)
    observations = reader.load_observations()
    actions = reader.load_actions()
    events = reader.load_events()
    action_space = events.get("action_space", [])
    return observations, actions, action_space, reader


def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load session
    print(f"Loading session from {args.session_path}...")
    observations, actions, action_space, reader = load_session(args.session_path)
    print(f"  {len(observations)} observations, {len(actions)} actions")

    # Pre-build canvases
    print("Pre-building canvases...")
    canvas_cache, detected_frame_size = prebuild_all_canvases(
        args.session_path, observations, actions, Config
    )
    print(f"  {len(canvas_cache)} canvases built, frame size: {detected_frame_size}")

    # Create dataset and dataloader
    dataset = CanvasDataset(canvas_cache)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Get canvas dimensions from first canvas
    sample = dataset[0]
    canvas_h, canvas_w = sample.shape[1], sample.shape[2]
    print(f"  Canvas dimensions: {canvas_h}x{canvas_w}")

    # Create VAE
    from models.vae import create_vae
    vae_kwargs = {}
    if args.vae_type == "custom":
        vae_kwargs = {
            'latent_channels': args.latent_channels,
            'compression_factor': args.compression_factor,
            'mode': args.mode,
        }
    elif args.vae_type == "dinov2":
        vae_kwargs = {
            'variant': args.dinov2_variant,
            'target_h': canvas_h,
            'target_w': canvas_w,
        }
    else:
        print(f"Error: VAE type '{args.vae_type}' does not support training. "
              f"Only 'custom' and 'dinov2' can be trained.")
        return

    vae = create_vae(vae_type=args.vae_type, device=str(device), **vae_kwargs)

    # Count trainable parameters
    trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vae.parameters())
    print(f"  VAE parameters: {trainable:,} trainable / {total:,} total")

    # Create optimizer (only trainable parameters)
    optimizer = optim.AdamW(
        [p for p in vae.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        vae.train()
        epoch_recon_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)
            loss_dict = vae.training_step(batch, optimizer)
            epoch_recon_loss += loss_dict['recon_loss']
            epoch_reg_loss += loss_dict['reg_loss']
            num_batches += 1

        avg_recon = epoch_recon_loss / num_batches
        avg_reg = epoch_reg_loss / num_batches
        avg_total = avg_recon + avg_reg

        if (epoch + 1) % max(1, args.epochs // 20) == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{args.epochs} | "
                  f"recon={avg_recon:.6f} reg={avg_reg:.6f} total={avg_total:.6f} | "
                  f"{elapsed:.0f}s")

        # Save best model
        if avg_total < best_loss:
            best_loss = avg_total
            save_path = _get_save_path(args)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': vae.state_dict(),
                'vae_type': args.vae_type,
                'epoch': epoch + 1,
                'loss': avg_total,
                'recon_loss': avg_recon,
                'config': {
                    'latent_channels': getattr(vae, 'latent_channels', None),
                    'compression_factor': getattr(vae, 'compression_factor', None),
                    'canvas_size': (canvas_h, canvas_w),
                },
            }, save_path)

    # Compute scaling factor for custom VAE
    if args.vae_type == "custom" and hasattr(vae, 'compute_scaling_factor'):
        print("\nComputing scaling factor...")
        vae.compute_scaling_factor(dataloader)
        # Re-save with scaling factor
        save_path = _get_save_path(args)
        torch.save({
            'model_state_dict': vae.state_dict(),
            'vae_type': args.vae_type,
            'epoch': args.epochs,
            'loss': best_loss,
            'config': {
                'latent_channels': vae.latent_channels,
                'compression_factor': vae.compression_factor,
                'canvas_size': (canvas_h, canvas_w),
                'scaling_factor': vae.scaling_factor.item(),
            },
        }, save_path)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoint saved to: {_get_save_path(args)}")


def _get_save_path(args) -> str:
    """Determine save path for VAE checkpoint."""
    session_name = Path(args.session_path).name

    # Determine robot type from session path
    if "toroidal_dot" in args.session_path:
        robot_type = "toroidal_dot"
    elif "so101" in args.session_path:
        robot_type = "so101"
    elif "jetbot" in args.session_path:
        robot_type = "jetbot"
    else:
        robot_type = "unknown"

    checkpoint_dir = f"saved/checkpoints/{robot_type}/vae"
    return os.path.join(checkpoint_dir, f"vae_{args.vae_type}_{session_name}.pth")


def main():
    parser = argparse.ArgumentParser(description="Train VAE/RAE for latent diffusion")

    parser.add_argument("--session-path", required=True, help="Path to session directory")
    parser.add_argument("--vae-type", default="custom", choices=["custom", "dinov2"],
                        help="VAE type to train")

    # Custom VAE args
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--compression-factor", type=int, default=8)
    parser.add_argument("--mode", default="vae", choices=["vae", "rae"])

    # DINOv2 args
    parser.add_argument("--dinov2-variant", default="vitb14",
                        choices=["vits14", "vitb14", "vitl14", "vitg14"])

    # Training args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    train_vae(args)


if __name__ == "__main__":
    main()
