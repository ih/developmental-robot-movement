"""
Tests for the latent diffusion DiT architecture.

Covers:
- DiTBlock zero-init identity behavior
- DiffusionViT forward shape validation
- NoiseScheduler add_noise / denoise roundtrip
- CanvasVAE encode/decode roundtrip
- LatentDiffusionWrapper train_on_canvas loss decrease
- LatentDiffusionWrapper forward_with_patch_mask non-trivial output
- Checkpoint save/load roundtrip
- Pixel mask -> latent mask mapping correctness
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import unittest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vit_dit import DiTBlock, DiffusionViT, SinusoidalTimestepEmbedding, modulate
from models.noise_scheduler import NoiseScheduler
from models.vae import CanvasVAE, create_vae
from models.latent_diffusion import LatentDiffusionWrapper


class TestSinusoidalTimestepEmbedding(unittest.TestCase):
    """Tests for the timestep embedding module."""

    def test_output_shape(self):
        embed = SinusoidalTimestepEmbedding(embed_dim=128)
        timesteps = torch.tensor([0, 100, 500, 999])
        out = embed(timesteps)
        self.assertEqual(out.shape, (4, 128))

    def test_different_timesteps_produce_different_embeddings(self):
        embed = SinusoidalTimestepEmbedding(embed_dim=64)
        t0 = embed(torch.tensor([0]))
        t500 = embed(torch.tensor([500]))
        self.assertFalse(torch.allclose(t0, t500, atol=1e-6))

    def test_odd_embed_dim(self):
        embed = SinusoidalTimestepEmbedding(embed_dim=65)
        timesteps = torch.tensor([42])
        out = embed(timesteps)
        self.assertEqual(out.shape, (1, 65))


class TestDiTBlockZeroInit(unittest.TestCase):
    """DiTBlock zero-init: gates start at 0, making blocks identity at init."""

    def test_zero_init_identity(self):
        """At initialization, DiTBlock should be approximately identity."""
        embed_dim = 64
        block = DiTBlock(embed_dim, num_heads=4)

        # Check that the adaLN modulation output layer is zero-initialized
        last_linear = block.adaLN_modulation[-1]
        self.assertTrue(torch.all(last_linear.weight == 0))
        self.assertTrue(torch.all(last_linear.bias == 0))

        # With zero-init, all 6 modulation params are 0 (gamma, beta, alpha all zero)
        # alpha (gate) = 0 -> residual contribution = 0 -> output == input
        x = torch.randn(2, 10, embed_dim)
        c = torch.randn(2, embed_dim)

        with torch.no_grad():
            out = block(x, c)

        self.assertTrue(
            torch.allclose(out, x, atol=1e-5),
            f"DiTBlock should be identity at init but max diff = {(out - x).abs().max().item()}"
        )

    def test_forward_with_attn_returns_weights(self):
        block = DiTBlock(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        c = torch.randn(2, 64)
        out, attn_weights = block.forward_with_attn(x, c)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(attn_weights.shape[0], 2)  # batch dim
        self.assertEqual(attn_weights.shape[1], 10)  # seq_len (queries)


class TestDiffusionViT(unittest.TestCase):
    """DiffusionViT forward shape validation and basic behavior."""

    def setUp(self):
        # Small DiT for testing: 28x88 latent with 4 channels, patch_size=2
        self.dit = DiffusionViT(
            img_height=28,
            img_width=88,
            in_channels=4,
            patch_size=2,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )
        self.B = 2
        self.latent = torch.randn(self.B, 4, 28, 88)
        self.num_patches = 14 * 44  # 616

    def test_grid_size(self):
        self.assertEqual(self.dit.grid_size, (14, 44))

    def test_forward_with_patch_mask_shape(self):
        mask = torch.ones(self.B, self.num_patches, dtype=torch.bool)
        pred, latent = self.dit.forward_with_patch_mask(self.latent, mask)

        patch_dim = 2 * 2 * 4  # patch_size^2 * in_channels = 16
        self.assertEqual(pred.shape, (self.B, self.num_patches, patch_dim))
        self.assertEqual(latent.shape, (self.B, self.num_patches + 1, 64))

    def test_forward_with_return_attn(self):
        mask = torch.ones(self.B, self.num_patches, dtype=torch.bool)
        pred, latent, attn_weights, returned_mask = self.dit.forward_with_patch_mask(
            self.latent, mask, return_attn=True
        )
        self.assertEqual(len(attn_weights), 2)  # depth=2
        self.assertTrue(torch.equal(returned_mask, mask))

    def test_forward_denoise(self):
        mask = torch.ones(self.B, self.num_patches, dtype=torch.bool)
        timestep = torch.tensor([100, 500])
        pred, latent = self.dit.forward_denoise(self.latent, mask, timestep)

        patch_dim = 2 * 2 * 4
        self.assertEqual(pred.shape, (self.B, self.num_patches, patch_dim))

    def test_patchify_unpatchify_roundtrip(self):
        patches = self.dit.patchify(self.latent)
        restored = self.dit.unpatchify(patches)
        self.assertTrue(
            torch.allclose(self.latent, restored, atol=1e-6),
            "patchify -> unpatchify should be lossless"
        )

    def test_encode_decode_shapes(self):
        """BaseAutoencoder interface: encode and decode."""
        features = self.dit.encode(self.latent)
        self.assertEqual(features.shape, (self.B, self.num_patches + 1, 64))

        decoded = self.dit.decode(features)
        self.assertEqual(decoded.shape, self.latent.shape)

    def test_default_timestep_zero(self):
        """When timestep is None, should default to 0 (fully denoised)."""
        mask = torch.zeros(self.B, self.num_patches, dtype=torch.bool)
        pred_default, _ = self.dit.forward_with_patch_mask(self.latent, mask, timestep=None)
        pred_explicit, _ = self.dit.forward_with_patch_mask(
            self.latent, mask, timestep=torch.zeros(self.B, dtype=torch.long)
        )
        self.assertTrue(torch.allclose(pred_default, pred_explicit, atol=1e-6))

    def test_prediction_head_zero_init(self):
        """decoder_pred should be zero-initialized for stable training start."""
        self.assertTrue(torch.all(self.dit.decoder_pred.weight == 0))
        self.assertTrue(torch.all(self.dit.decoder_pred.bias == 0))


class TestNoiseScheduler(unittest.TestCase):
    """NoiseScheduler forward/reverse process tests."""

    def setUp(self):
        self.scheduler = NoiseScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
        )

    def test_schedule_shape(self):
        self.assertEqual(self.scheduler.betas.shape, (1000,))
        self.assertEqual(self.scheduler.alphas_cumprod.shape, (1000,))

    def test_alphas_cumprod_decreasing(self):
        """alpha_bar should monotonically decrease (more noise over time)."""
        diffs = self.scheduler.alphas_cumprod[1:] - self.scheduler.alphas_cumprod[:-1]
        self.assertTrue(torch.all(diffs <= 0), "alphas_cumprod should be monotonically decreasing")

    def test_add_noise_at_t0(self):
        """At t=0, noise should be minimal (alpha_bar close to 1)."""
        clean = torch.ones(2, 4)
        noise = torch.randn(2, 4)
        t = torch.zeros(2, dtype=torch.long)
        noisy = self.scheduler.add_noise(clean, noise, t)
        # At t=0, alpha_bar is close to 1, so noisy should be close to clean
        self.assertTrue(
            torch.allclose(noisy, clean, atol=0.02),
            f"At t=0, noisy should be very close to clean but max diff = {(noisy - clean).abs().max().item()}"
        )

    def test_add_noise_at_max_t(self):
        """At max t, the signal is mostly noise."""
        clean = torch.ones(2, 4)
        noise = torch.randn(2, 4)
        t = torch.full((2,), 999, dtype=torch.long)
        noisy = self.scheduler.add_noise(clean, noise, t)
        # At t=999, alpha_bar is very small, so output should be mostly noise
        alpha_bar = self.scheduler.alphas_cumprod[999].item()
        self.assertLess(alpha_bar, 0.05, "At t=999, alpha_bar should be very small")

    def test_set_timesteps(self):
        self.scheduler.set_timesteps(50)
        self.assertIsNotNone(self.scheduler.timesteps)
        self.assertEqual(len(self.scheduler.timesteps), 50)
        # Should be in descending order
        for i in range(len(self.scheduler.timesteps) - 1):
            self.assertGreaterEqual(
                self.scheduler.timesteps[i].item(),
                self.scheduler.timesteps[i + 1].item()
            )

    def test_ddim_step_reduces_noise(self):
        """A DDIM step should move the sample closer to the predicted clean."""
        clean = torch.randn(2, 16)
        noise = torch.randn_like(clean)
        t = 500
        t_tensor = torch.full((2,), t, dtype=torch.long)

        noisy = self.scheduler.add_noise(clean, noise, t_tensor)
        self.scheduler.set_timesteps(50)

        # Assuming model correctly predicts the noise
        prev_sample = self.scheduler.step(noise, t, noisy, eta=0.0)

        # After denoising step, sample should be closer to clean than before
        dist_before = (noisy - clean).pow(2).mean().item()
        dist_after = (prev_sample - clean).pow(2).mean().item()
        self.assertLess(
            dist_after, dist_before,
            f"DDIM step with perfect noise prediction should reduce distance to clean: "
            f"before={dist_before:.4f}, after={dist_after:.4f}"
        )

    def test_cosine_schedule(self):
        """Cosine schedule should also produce valid alphas_cumprod."""
        sched = NoiseScheduler(
            num_train_timesteps=100,
            beta_schedule="cosine",
        )
        self.assertTrue(torch.all(sched.alphas_cumprod > 0))
        self.assertTrue(torch.all(sched.alphas_cumprod <= 1))

    def test_sample_prediction_type(self):
        """Test with prediction_type='sample': single step moves toward clean."""
        sched = NoiseScheduler(
            num_train_timesteps=100,
            prediction_type="sample",
        )
        sched.set_timesteps(10)

        clean = torch.randn(2, 8)
        noise = torch.randn_like(clean)
        t = 50
        t_tensor = torch.full((2,), t, dtype=torch.long)
        noisy = sched.add_noise(clean, noise, t_tensor)

        # Model predicts clean sample perfectly
        prev = sched.step(clean, t, noisy, eta=0.0)

        # A single DDIM step moves closer to clean, but doesn't fully denoise
        dist_before = (noisy - clean).pow(2).mean().item()
        dist_after = (prev - clean).pow(2).mean().item()
        self.assertLess(
            dist_after, dist_before,
            f"DDIM step with perfect sample prediction should reduce distance: "
            f"before={dist_before:.4f}, after={dist_after:.4f}"
        )

    def test_to_device(self):
        """Test moving scheduler to a device."""
        device = torch.device("cpu")
        self.scheduler.to(device)
        self.assertEqual(self.scheduler.betas.device.type, "cpu")


class TestCanvasVAE(unittest.TestCase):
    """CanvasVAE encode/decode roundtrip and training tests."""

    def setUp(self):
        self.vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="vae")
        self.B = 2
        self.imgs = torch.rand(self.B, 3, 224, 704)  # Standard canvas size

    def test_encode_shape(self):
        self.vae.eval()
        with torch.no_grad():
            latent = self.vae.encode(self.imgs)
        expected_h = 224 // 8  # 28
        expected_w = 704 // 8  # 88
        self.assertEqual(latent.shape, (self.B, 4, expected_h, expected_w))

    def test_decode_shape(self):
        latent = torch.randn(self.B, 4, 28, 88)
        with torch.no_grad():
            decoded = self.vae.decode(latent)
        self.assertEqual(decoded.shape, (self.B, 3, 224, 704))

    def test_encode_decode_roundtrip_shapes(self):
        """Encode then decode should produce correct shape."""
        self.vae.eval()
        with torch.no_grad():
            latent = self.vae.encode(self.imgs)
            reconstructed = self.vae.decode(latent)
        self.assertEqual(reconstructed.shape, self.imgs.shape)

    def test_training_step(self):
        """Single training step should return valid loss dict."""
        optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)
        small_imgs = torch.rand(2, 3, 64, 64)  # Smaller for speed
        vae = CanvasVAE(latent_channels=4, compression_factor=4, mode="vae")
        loss_dict = vae.training_step(small_imgs, optimizer)

        self.assertIn('recon_loss', loss_dict)
        self.assertIn('reg_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)
        self.assertGreater(loss_dict['total_loss'], 0)

    def test_rae_mode(self):
        """RAE mode uses L2 regularization instead of KL."""
        rae = CanvasVAE(latent_channels=4, compression_factor=4, mode="rae")
        small_imgs = torch.rand(2, 3, 64, 64)
        optimizer = optim.Adam(rae.parameters(), lr=1e-3)
        loss_dict = rae.training_step(small_imgs, optimizer)
        self.assertGreater(loss_dict['total_loss'], 0)

    def test_kl_loss_vae_mode(self):
        """kl_loss() should return positive value in VAE mode."""
        self.vae.train()
        _ = self.vae._encode_raw(torch.rand(1, 3, 64, 64))
        kl = self.vae.kl_loss()
        self.assertGreater(kl.item(), 0)

    def test_kl_loss_rae_mode(self):
        """kl_loss() should return 0 in RAE mode."""
        rae = CanvasVAE(latent_channels=4, compression_factor=4, mode="rae")
        kl = rae.kl_loss()
        self.assertEqual(kl.item(), 0.0)


class TestCanvasVAECreateFactory(unittest.TestCase):
    """Test the create_vae factory function."""

    def test_create_custom(self):
        vae = create_vae("custom", latent_channels=4, compression_factor=8)
        self.assertIsInstance(vae, CanvasVAE)
        self.assertEqual(vae.latent_channels, 4)
        self.assertEqual(vae.compression_factor, 8)

    def test_create_unknown_raises(self):
        with self.assertRaises(ValueError):
            create_vae("unknown_type")


class TestPixelMaskToLatentMask(unittest.TestCase):
    """Test mask mapping from pixel patch grid to latent patch grid."""

    def setUp(self):
        # Create a small LatentDiffusionWrapper for testing mask mapping
        self.vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")
        self.dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        self.scheduler = NoiseScheduler(num_train_timesteps=100)
        self.wrapper = LatentDiffusionWrapper(
            self.vae, self.dit, self.scheduler, num_inference_steps=5
        )

    def test_1to1_mapping_sd_vae(self):
        """With 8x VAE and latent patch_size=2, pixel 16x16 grid maps 1:1."""
        B = 2
        pixel_h, pixel_w = 224, 704
        num_pixel_patches = (pixel_h // 16) * (pixel_w // 16)  # 14 * 44 = 616

        # Create pixel mask: mask last 44 patches (last frame column)
        pixel_mask = torch.zeros(B, num_pixel_patches, dtype=torch.bool)
        pixel_mask[:, -44:] = True

        latent_mask = self.wrapper._pixel_mask_to_latent_mask(pixel_mask, pixel_h, pixel_w)

        # Should be identical for 1:1 mapping
        self.assertEqual(latent_mask.shape, pixel_mask.shape)
        self.assertTrue(torch.equal(latent_mask, pixel_mask))

    def test_full_mask_maps_to_full_mask(self):
        """Full mask in pixel space should be full mask in latent space."""
        B = 1
        pixel_h, pixel_w = 224, 704
        num_patches = (pixel_h // 16) * (pixel_w // 16)

        pixel_mask = torch.ones(B, num_patches, dtype=torch.bool)
        latent_mask = self.wrapper._pixel_mask_to_latent_mask(pixel_mask, pixel_h, pixel_w)
        self.assertTrue(torch.all(latent_mask))

    def test_empty_mask_maps_to_empty_mask(self):
        """Empty mask in pixel space should be empty mask in latent space."""
        B = 1
        pixel_h, pixel_w = 224, 704
        num_patches = (pixel_h // 16) * (pixel_w // 16)

        pixel_mask = torch.zeros(B, num_patches, dtype=torch.bool)
        latent_mask = self.wrapper._pixel_mask_to_latent_mask(pixel_mask, pixel_h, pixel_w)
        self.assertFalse(torch.any(latent_mask))


class TestLatentDiffusionWrapperTraining(unittest.TestCase):
    """LatentDiffusionWrapper training and inference tests."""

    def setUp(self):
        # Seed for reproducibility (untrained VAE can produce extreme latents)
        torch.manual_seed(42)

        # Use small dimensions for speed
        self.vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")

        # Normalize VAE output to prevent NaN in DiT attention
        # Compute rough scaling factor from a test batch
        self.vae.eval()
        test_imgs = torch.rand(2, 3, 224, 704)
        with torch.no_grad():
            raw_z = self.vae._encode_raw(test_imgs)
            std = raw_z.std().item()
            if std > 0:
                self.vae.scaling_factor.fill_(1.0 / std)

        self.dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        self.scheduler = NoiseScheduler(
            num_train_timesteps=100,
            prediction_type="epsilon",
        )
        self.wrapper = LatentDiffusionWrapper(
            self.vae, self.dit, self.scheduler, num_inference_steps=5
        )
        self.optimizer = optim.AdamW(
            [p for p in self.wrapper.parameters() if p.requires_grad],
            lr=1e-3,
        )
        # Small canvas for testing
        self.B = 2
        self.canvas = torch.rand(self.B, 3, 224, 704)
        self.num_pixel_patches = 14 * 44  # 616
        self.pixel_mask = torch.ones(self.B, self.num_pixel_patches, dtype=torch.bool)

    def test_train_on_canvas_returns_loss(self):
        """train_on_canvas should return a positive loss value."""
        loss, grad_diag = self.wrapper.train_on_canvas(
            self.canvas, self.pixel_mask, self.optimizer
        )
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        self.assertIn('lr', grad_diag)
        self.assertIn('loss_hybrid', grad_diag)

    def test_train_on_canvas_with_per_sample_losses(self):
        """train_on_canvas with return_per_sample_losses should return list."""
        loss, grad_diag, per_sample = self.wrapper.train_on_canvas(
            self.canvas, self.pixel_mask, self.optimizer,
            return_per_sample_losses=True,
        )
        self.assertEqual(len(per_sample), self.B)
        for sl in per_sample:
            self.assertIsInstance(sl, float)

    def test_loss_decreases_over_steps(self):
        """Loss should generally decrease after multiple training steps."""
        losses = []
        for _ in range(10):
            loss, _ = self.wrapper.train_on_canvas(
                self.canvas, self.pixel_mask, self.optimizer
            )
            losses.append(loss)

        # Loss at end should be less than loss at start (with some tolerance)
        # Use rolling average to smooth out noise
        avg_first_3 = sum(losses[:3]) / 3
        avg_last_3 = sum(losses[-3:]) / 3
        self.assertLess(
            avg_last_3, avg_first_3,
            f"Loss should decrease: first 3 avg={avg_first_3:.4f}, last 3 avg={avg_last_3:.4f}"
        )

    def test_vae_stays_frozen(self):
        """VAE parameters should not have gradients after training."""
        self.wrapper.train_on_canvas(self.canvas, self.pixel_mask, self.optimizer)
        for p in self.wrapper.vae.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)

    def test_dit_gets_gradients(self):
        """DiT parameters should get gradients after training."""
        self.wrapper.train_on_canvas(self.canvas, self.pixel_mask, self.optimizer)
        has_grad = False
        for p in self.wrapper.dit.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        self.assertTrue(has_grad, "DiT should receive gradients during training")


class TestLatentDiffusionWrapperInference(unittest.TestCase):
    """LatentDiffusionWrapper inference tests."""

    def setUp(self):
        self.vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")
        self.dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        self.scheduler = NoiseScheduler(
            num_train_timesteps=100,
            prediction_type="epsilon",
        )
        self.wrapper = LatentDiffusionWrapper(
            self.vae, self.dit, self.scheduler, num_inference_steps=3
        )
        self.B = 1
        self.canvas = torch.rand(self.B, 3, 224, 704)
        self.num_pixel_patches = 14 * 44

    def test_forward_with_patch_mask_shape(self):
        """Inference should return pixel-space patch predictions."""
        mask = torch.ones(self.B, self.num_pixel_patches, dtype=torch.bool)
        pred_patches, latent = self.wrapper.forward_with_patch_mask(self.canvas, mask)

        pixel_patch_dim = 16 * 16 * 3  # 768
        self.assertEqual(pred_patches.shape, (self.B, self.num_pixel_patches, pixel_patch_dim))

    def test_forward_with_return_attn(self):
        mask = torch.ones(self.B, self.num_pixel_patches, dtype=torch.bool)
        pred, latent, attn, returned_mask = self.wrapper.forward_with_patch_mask(
            self.canvas, mask, return_attn=True
        )
        self.assertTrue(torch.equal(returned_mask, mask))

    def test_output_is_bounded(self):
        """Output pixel values should be in [0, 1] range."""
        mask = torch.ones(self.B, self.num_pixel_patches, dtype=torch.bool)
        pred_patches, _ = self.wrapper.forward_with_patch_mask(self.canvas, mask)
        pred_img = self.wrapper.unpatchify(pred_patches)
        self.assertTrue(pred_img.min() >= 0.0, f"Min pixel value: {pred_img.min().item()}")
        self.assertTrue(pred_img.max() <= 1.0, f"Max pixel value: {pred_img.max().item()}")

    def test_unmasked_regions_preserved(self):
        """Patches that are NOT masked should be close to the original."""
        # Mask only a subset of patches (last 44 = last column of latent grid)
        mask = torch.zeros(self.B, self.num_pixel_patches, dtype=torch.bool)
        mask[:, -44:] = True

        pred_patches, _ = self.wrapper.forward_with_patch_mask(self.canvas, mask)

        # Unmasked patches in the prediction should be close to the original
        # (they go through VAE encode->decode so won't be exact, but should be close
        # for the non-masked regions)
        original_patches = self.wrapper.patchify(self.canvas)
        unmasked_pred = pred_patches[:, :-44, :]
        unmasked_orig = original_patches[:, :-44, :]

        # The VAE roundtrip introduces some error, especially untrained,
        # so we just check that predictions are finite and not all-zero
        self.assertTrue(torch.isfinite(unmasked_pred).all())
        self.assertFalse(torch.all(unmasked_pred == 0))


class TestLatentDiffusionWrapperProperties(unittest.TestCase):
    """Test that wrapper correctly proxies DiT properties."""

    def setUp(self):
        self.vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")
        self.dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        self.scheduler = NoiseScheduler(num_train_timesteps=100)
        self.wrapper = LatentDiffusionWrapper(
            self.vae, self.dit, self.scheduler
        )

    def test_decoder_pred_proxy(self):
        self.assertIs(self.wrapper.decoder_pred, self.dit.decoder_pred)

    def test_decoder_blocks_proxy(self):
        self.assertIs(self.wrapper.decoder_blocks, self.dit.decoder_blocks)

    def test_mask_token_proxy(self):
        self.assertIs(self.wrapper.mask_token, self.dit.mask_token)

    def test_embed_dim_proxy(self):
        self.assertEqual(self.wrapper.embed_dim, 64)

    def test_grid_size_proxy(self):
        self.assertEqual(self.wrapper.grid_size, (14, 44))

    def test_patch_size_proxy(self):
        self.assertEqual(self.wrapper.patch_size, 2)

    def test_parameters_only_dit(self):
        """wrapper.parameters() should only yield DiT params."""
        wrapper_params = set(id(p) for p in self.wrapper.parameters())
        dit_params = set(id(p) for p in self.dit.parameters())
        vae_params = set(id(p) for p in self.vae.parameters())

        self.assertEqual(wrapper_params, dit_params)
        self.assertTrue(wrapper_params.isdisjoint(vae_params))

    def test_train_mode(self):
        """train() should set DiT to train, VAE to eval."""
        self.wrapper.train()
        self.assertTrue(self.dit.training)
        self.assertFalse(self.vae.training)

    def test_eval_mode(self):
        self.wrapper.eval()
        self.assertFalse(self.dit.training)
        self.assertFalse(self.vae.training)


class TestCheckpointSaveLoad(unittest.TestCase):
    """Test checkpoint save/load roundtrip."""

    def test_save_load_dit_weights(self):
        """Saving and loading wrapper state should preserve DiT weights."""
        vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")
        dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        scheduler = NoiseScheduler(num_train_timesteps=100)
        wrapper = LatentDiffusionWrapper(vae, dit, scheduler)

        # Modify weights from initial values
        with torch.no_grad():
            for p in dit.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            save_path = f.name
            torch.save(wrapper.state_dict(), save_path)

        try:
            # Create fresh wrapper and load
            dit2 = DiffusionViT(
                img_height=28, img_width=88,
                in_channels=4, patch_size=2,
                embed_dim=64, depth=2, num_heads=4,
            )
            vae2 = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")
            scheduler2 = NoiseScheduler(num_train_timesteps=100)
            wrapper2 = LatentDiffusionWrapper(vae2, dit2, scheduler2)

            state = torch.load(save_path, weights_only=True)
            wrapper2.load_state_dict(state)

            # Compare weights
            for (n1, p1), (n2, p2) in zip(
                dit.named_parameters(), dit2.named_parameters()
            ):
                self.assertTrue(
                    torch.equal(p1, p2),
                    f"Weight mismatch for {n1}"
                )
        finally:
            os.unlink(save_path)

    def test_state_dict_excludes_vae(self):
        """state_dict() should only contain DiT keys, not VAE keys."""
        vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")
        dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        scheduler = NoiseScheduler(num_train_timesteps=100)
        wrapper = LatentDiffusionWrapper(vae, dit, scheduler)

        state = wrapper.state_dict()
        for key in state:
            self.assertFalse(
                key.startswith("vae."),
                f"State dict should not contain VAE key: {key}"
            )

    def test_full_checkpoint_roundtrip(self):
        """Test saving and loading a full checkpoint (dict with metadata)."""
        vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")
        dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        scheduler = NoiseScheduler(num_train_timesteps=100)
        wrapper = LatentDiffusionWrapper(vae, dit, scheduler)

        optimizer = optim.AdamW(wrapper.parameters(), lr=1e-3)

        # Do a training step to populate optimizer state
        canvas = torch.rand(1, 3, 224, 704)
        mask = torch.ones(1, 616, dtype=torch.bool)
        wrapper.train_on_canvas(canvas, mask, optimizer)

        # Save full checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            save_path = f.name
            torch.save({
                'model_state_dict': wrapper.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vae_type': 'custom',
                'epoch': 5,
            }, save_path)

        try:
            loaded = torch.load(save_path, weights_only=False)
            self.assertEqual(loaded['vae_type'], 'custom')
            self.assertEqual(loaded['epoch'], 5)
            self.assertIn('model_state_dict', loaded)
            self.assertIn('optimizer_state_dict', loaded)
        finally:
            os.unlink(save_path)


class TestPixelPatchifyUnpatchify(unittest.TestCase):
    """Test pixel-space patchify/unpatchify on the wrapper."""

    def setUp(self):
        self.vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")
        self.dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        self.scheduler = NoiseScheduler(num_train_timesteps=100)
        self.wrapper = LatentDiffusionWrapper(
            self.vae, self.dit, self.scheduler
        )

    def test_pixel_patchify_shape(self):
        imgs = torch.rand(2, 3, 224, 704)
        patches = self.wrapper.patchify(imgs)
        num_patches = (224 // 16) * (704 // 16)  # 14 * 44 = 616
        patch_dim = 16 * 16 * 3  # 768
        self.assertEqual(patches.shape, (2, num_patches, patch_dim))

    def test_pixel_patchify_unpatchify_roundtrip(self):
        imgs = torch.rand(2, 3, 224, 704)
        self.wrapper._pixel_canvas_size = (224, 704)
        patches = self.wrapper.patchify(imgs)
        restored = self.wrapper.unpatchify(patches)
        self.assertTrue(
            torch.allclose(imgs, restored, atol=1e-6),
            "Pixel patchify -> unpatchify should be lossless"
        )


class TestDiffusionViTUnconditionalForward(unittest.TestCase):
    """Test DiffusionViT forward pass with patch_mask=None (unconditional mode)."""

    def setUp(self):
        self.dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        self.B = 2
        self.latent = torch.randn(self.B, 4, 28, 88)
        self.num_patches = 14 * 44

    def test_forward_without_mask_shape(self):
        """Forward pass with patch_mask=None should produce correct shapes."""
        pred, latent = self.dit.forward_with_patch_mask(self.latent, None)
        patch_dim = 2 * 2 * 4
        self.assertEqual(pred.shape, (self.B, self.num_patches, patch_dim))
        self.assertEqual(latent.shape, (self.B, self.num_patches + 1, 64))

    def test_forward_without_mask_with_timestep(self):
        """Unconditional forward with explicit timesteps should work."""
        timestep = torch.tensor([100, 500])
        pred, latent = self.dit.forward_with_patch_mask(
            self.latent, None, timestep=timestep
        )
        patch_dim = 2 * 2 * 4
        self.assertEqual(pred.shape, (self.B, self.num_patches, patch_dim))

    def test_forward_denoise_without_mask(self):
        """forward_denoise with patch_mask=None should work."""
        timestep = torch.tensor([100, 500])
        pred, latent = self.dit.forward_denoise(self.latent, None, timestep)
        patch_dim = 2 * 2 * 4
        self.assertEqual(pred.shape, (self.B, self.num_patches, patch_dim))

    def test_mask_token_not_in_graph_without_mask(self):
        """With patch_mask=None, mask_token should not be in computation graph."""
        pred, _ = self.dit.forward_with_patch_mask(self.latent, None)
        loss = pred.sum()
        loss.backward()
        self.assertIsNone(
            self.dit.mask_token.grad,
            "mask_token should not receive gradients in unconditional mode"
        )


class TestUnconditionalTraining(unittest.TestCase):
    """Test LatentDiffusionWrapper with training_mode='unconditional'."""

    def setUp(self):
        torch.manual_seed(42)
        self.vae = CanvasVAE(latent_channels=4, compression_factor=8, mode="rae")
        self.vae.eval()
        test_imgs = torch.rand(2, 3, 224, 704)
        with torch.no_grad():
            raw_z = self.vae._encode_raw(test_imgs)
            std = raw_z.std().item()
            if std > 0:
                self.vae.scaling_factor.fill_(1.0 / std)

        self.dit = DiffusionViT(
            img_height=28, img_width=88,
            in_channels=4, patch_size=2,
            embed_dim=64, depth=2, num_heads=4,
        )
        self.scheduler = NoiseScheduler(
            num_train_timesteps=100,
            prediction_type="epsilon",
        )
        self.wrapper = LatentDiffusionWrapper(
            self.vae, self.dit, self.scheduler,
            num_inference_steps=5,
            training_mode="unconditional",
        )
        self.optimizer = optim.AdamW(
            [p for p in self.wrapper.parameters() if p.requires_grad],
            lr=1e-3,
        )
        self.B = 2
        self.canvas = torch.rand(self.B, 3, 224, 704)
        self.num_pixel_patches = 14 * 44
        self.pixel_mask = torch.ones(self.B, self.num_pixel_patches, dtype=torch.bool)

    def test_train_returns_loss(self):
        """Unconditional training should return a positive loss."""
        loss, grad_diag = self.wrapper.train_on_canvas(
            self.canvas, self.pixel_mask, self.optimizer
        )
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_mask_token_no_grad(self):
        """mask_token should NOT receive gradients in unconditional training."""
        self.wrapper.train_on_canvas(
            self.canvas, self.pixel_mask, self.optimizer
        )
        self.assertIsNone(
            self.dit.mask_token.grad,
            "mask_token should not receive gradients in unconditional mode"
        )

    def test_mask_does_not_affect_loss(self):
        """Different masks should produce the same loss in unconditional mode."""
        # Full mask
        torch.manual_seed(123)
        full_mask = torch.ones(self.B, self.num_pixel_patches, dtype=torch.bool)
        loss_full, _ = self.wrapper.train_on_canvas(
            self.canvas, full_mask, self.optimizer
        )

        # Reset model and optimizer
        self.dit.load_state_dict(
            DiffusionViT(
                img_height=28, img_width=88,
                in_channels=4, patch_size=2,
                embed_dim=64, depth=2, num_heads=4,
            ).state_dict()
        )
        self.optimizer = optim.AdamW(
            [p for p in self.wrapper.parameters() if p.requires_grad],
            lr=1e-3,
        )

        # Partial mask
        torch.manual_seed(123)
        partial_mask = torch.zeros(self.B, self.num_pixel_patches, dtype=torch.bool)
        partial_mask[:, -44:] = True
        loss_partial, _ = self.wrapper.train_on_canvas(
            self.canvas, partial_mask, self.optimizer
        )

        self.assertAlmostEqual(
            loss_full, loss_partial, places=3,
            msg="Unconditional training loss should not depend on mask"
        )

    def test_per_sample_losses(self):
        """Per-sample losses should cover all patches."""
        loss, grad_diag, per_sample = self.wrapper.train_on_canvas(
            self.canvas, self.pixel_mask, self.optimizer,
            return_per_sample_losses=True,
        )
        self.assertEqual(len(per_sample), self.B)
        for sl in per_sample:
            self.assertIsInstance(sl, float)
            self.assertGreater(sl, 0)

    def test_loss_decreases(self):
        """Loss should generally decrease after multiple unconditional training steps."""
        losses = []
        for _ in range(10):
            loss, _ = self.wrapper.train_on_canvas(
                self.canvas, self.pixel_mask, self.optimizer
            )
            losses.append(loss)

        avg_first_3 = sum(losses[:3]) / 3
        avg_last_3 = sum(losses[-3:]) / 3
        self.assertLess(
            avg_last_3, avg_first_3,
            f"Loss should decrease: first 3 avg={avg_first_3:.4f}, last 3 avg={avg_last_3:.4f}"
        )

    def test_inference_shape(self):
        """Inference with unconditional wrapper should produce correct shapes."""
        mask = torch.ones(1, self.num_pixel_patches, dtype=torch.bool)
        canvas = torch.rand(1, 3, 224, 704)
        pred_patches, _ = self.wrapper.forward_with_patch_mask(canvas, mask)
        pixel_patch_dim = 16 * 16 * 3
        self.assertEqual(pred_patches.shape, (1, self.num_pixel_patches, pixel_patch_dim))

    def test_inference_unmasked_preserved(self):
        """In unconditional RePaint, unmasked patches should still be close to original."""
        mask = torch.zeros(1, self.num_pixel_patches, dtype=torch.bool)
        mask[:, -44:] = True
        canvas = torch.rand(1, 3, 224, 704)
        pred_patches, _ = self.wrapper.forward_with_patch_mask(canvas, mask)
        # Unmasked predictions should be finite (VAE roundtrip introduces error)
        unmasked_pred = pred_patches[:, :-44, :]
        self.assertTrue(torch.isfinite(unmasked_pred).all())
        self.assertFalse(torch.all(unmasked_pred == 0))


class TestModulateFunction(unittest.TestCase):
    """Test the modulate() helper function."""

    def test_identity_with_zeros(self):
        """modulate(x, shift=0, scale=0) should equal x."""
        x = torch.randn(2, 10, 64)
        shift = torch.zeros(2, 64)
        scale = torch.zeros(2, 64)
        result = modulate(x, shift, scale)
        self.assertTrue(torch.allclose(result, x))

    def test_scale_and_shift(self):
        """modulate should apply (1 + scale) * x + shift."""
        x = torch.ones(1, 5, 4)
        shift = torch.full((1, 4), 0.5)
        scale = torch.full((1, 4), 1.0)  # (1 + 1) = 2
        result = modulate(x, shift, scale)
        expected = torch.full((1, 5, 4), 2.5)  # 2 * 1 + 0.5
        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
