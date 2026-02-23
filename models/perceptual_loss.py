"""
VGG Perceptual Loss for sharper image predictions.

Computes loss in VGG feature space rather than pixel space,
encouraging structurally plausible, sharp outputs over blurry MSE means.

Usage:
    loss_fn = VGGPerceptualLoss(device='cuda')
    loss = loss_fn(predicted_images, target_images)

Set PERCEPTUAL_LOSS_WEIGHT = 0.0 in config to disable entirely (no VGG loaded).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG16 features.

    Extracts features from relu1_2, relu2_2, relu3_3 and computes
    L1 loss between predicted and target feature maps.
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, device='cpu', layer_weights=None):
        super().__init__()

        # Load pretrained VGG16 features
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device)
        model.eval()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Extract feature blocks up to relu1_2 (idx 4), relu2_2 (idx 9), relu3_3 (idx 16)
        self.block1 = nn.Sequential(*list(model.children())[:4]).to(device)   # -> relu1_2
        self.block2 = nn.Sequential(*list(model.children())[4:9]).to(device)  # -> relu2_2
        self.block3 = nn.Sequential(*list(model.children())[9:16]).to(device) # -> relu3_3

        # Layer weights (how much each feature level contributes)
        self.layer_weights = layer_weights or [1.0, 1.0, 1.0]

        # Register normalization buffers
        self.register_buffer(
            'mean', torch.tensor(self.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor(self.IMAGENET_STD, device=device).view(1, 3, 1, 1)
        )

    def _normalize(self, x):
        """Apply ImageNet normalization. Input should be [B, 3, H, W] in [0, 1]."""
        return (x - self.mean) / self.std

    def _extract_features(self, x):
        """Extract multi-scale VGG features."""
        x = self._normalize(x)
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        return [f1, f2, f3]

    def forward(self, pred, target):
        """
        Compute perceptual loss.

        Args:
            pred: [B, 3, H, W] predicted images in [0, 1] range
            target: [B, 3, H, W] target images in [0, 1] range

        Returns:
            Scalar perceptual loss
        """
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = 0.0
        for pf, tf, w in zip(pred_features, target_features, self.layer_weights):
            loss += w * F.l1_loss(pf, tf)

        return loss
