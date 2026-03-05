"""
Diffusion noise scheduler for DDPM and DDIM sampling.

Handles the forward diffusion process (adding noise) and reverse process
(denoising) for canvas-based latent diffusion models.
"""

import math
import torch
import numpy as np


class NoiseScheduler:
    """
    DDPM/DDIM noise scheduler for canvas-based diffusion.

    Precomputes noise schedule parameters (alphas, betas) and provides
    methods for adding noise (forward process) and denoising (reverse process).

    Args:
        num_train_timesteps: Number of diffusion timesteps during training.
        beta_start: Start value for linear/cosine beta schedule.
        beta_end: End value for linear/cosine beta schedule.
        beta_schedule: Type of beta schedule ('linear' or 'cosine').
        prediction_type: What the model predicts ('epsilon' for noise, 'sample' for clean).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        # Compute betas
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}. Use 'linear' or 'cosine'.")

        # Precompute alpha schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute frequently used values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For DDPM reverse process
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        # Convert to float32 for GPU use
        self.betas = self.betas.float()
        self.alphas = self.alphas.float()
        self.alphas_cumprod = self.alphas_cumprod.float()
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.float()
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.float()
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.float()
        self.posterior_variance = self.posterior_variance.float()

        # Inference timesteps (set by set_timesteps())
        self.timesteps = None

    @staticmethod
    def _cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule as proposed in 'Improved DDPM'."""
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0, 0.999)

    def _get_schedule_values(self, timesteps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Index into a schedule tensor using timesteps, broadcasting to match batch dims."""
        device = timesteps.device
        vals = values.to(device)
        out = vals[timesteps]
        # Reshape for broadcasting: [B] -> [B, 1, 1, ...] or [B, 1]
        while out.ndim < 2:
            out = out.unsqueeze(-1)
        return out

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise at given timesteps.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            clean: Clean samples [B, ...] (patches or latents).
            noise: Noise tensor, same shape as clean.
            timesteps: Integer timesteps [B].

        Returns:
            Noisy samples at timestep t, same shape as clean.
        """
        sqrt_alpha = self._get_schedule_values(timesteps, self.sqrt_alphas_cumprod)
        sqrt_one_minus_alpha = self._get_schedule_values(timesteps, self.sqrt_one_minus_alphas_cumprod)

        # Broadcast to match clean tensor dims
        while sqrt_alpha.ndim < clean.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        while sqrt_one_minus_alpha.ndim < clean.ndim:
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * clean + sqrt_one_minus_alpha * noise

    def set_timesteps(self, num_inference_steps: int):
        """
        Configure DDIM timestep schedule for inference.

        Creates evenly spaced timesteps from T-1 down to 0.

        Args:
            num_inference_steps: Number of denoising steps to use.
        """
        step_ratio = self.num_train_timesteps / num_inference_steps
        # Evenly spaced timesteps in reverse order (T-1, ..., 0)
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).long()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Single DDIM denoising step.

        Args:
            model_output: Model prediction (noise or clean, depending on prediction_type).
            timestep: Current timestep (integer).
            sample: Current noisy sample [B, ...].
            eta: DDIM stochasticity parameter (0.0 = deterministic DDIM, 1.0 = DDPM).

        Returns:
            Denoised sample at previous timestep, same shape as sample.
        """
        device = sample.device
        t = timestep

        # Current and previous alpha_bar
        alpha_bar_t = self.alphas_cumprod[t].to(device)

        # Find previous timestep
        if self.timesteps is not None:
            timestep_list = self.timesteps.tolist()
            if t in timestep_list:
                idx = timestep_list.index(t)
                if idx < len(timestep_list) - 1:
                    t_prev = timestep_list[idx + 1]
                else:
                    t_prev = 0
            else:
                t_prev = max(0, t - 1)
        else:
            t_prev = max(0, t - 1)

        alpha_bar_t_prev = self.alphas_cumprod[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0, device=device)

        # Predict x_0 from model output
        if self.prediction_type == "epsilon":
            # model predicts noise
            sqrt_alpha_bar = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).sqrt()
            pred_x0 = (sample - sqrt_one_minus_alpha_bar * model_output) / sqrt_alpha_bar.clamp(min=1e-8)
        elif self.prediction_type == "sample":
            # model predicts clean sample directly
            pred_x0 = model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # DDIM formula
        sigma_t = eta * ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)).sqrt()

        # Direction pointing to x_t
        pred_dir = (1 - alpha_bar_t_prev - sigma_t ** 2).clamp(min=0).sqrt() * \
            ((sample - alpha_bar_t.sqrt() * pred_x0) / (1 - alpha_bar_t).sqrt().clamp(min=1e-8))

        # Compute previous sample
        prev_sample = alpha_bar_t_prev.sqrt() * pred_x0 + pred_dir

        if eta > 0 and t_prev > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + sigma_t * noise

        return prev_sample

    def to(self, device):
        """Move schedule tensors to device (for convenience)."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        if self.timesteps is not None:
            self.timesteps = self.timesteps.to(device)
        return self
