#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Diffusion utilities including noise scheduling and sampling
'''
import torch
import numpy as np
from typing import Optional, Union, Tuple


class NoiseScheduler:
    """
    Noise scheduler for diffusion process (DDPM/DDIM)
    """
    def __init__(
        self,
        num_diffusion_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon"  # epsilon or sample
    ):
        self.num_diffusion_steps = num_diffusion_steps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        elif beta_schedule == "squaredcos_cap_v2":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = num_diffusion_steps + 1
            x = torch.linspace(0, num_diffusion_steps, steps)
            alphas_cumprod = torch.cos(((x / num_diffusion_steps) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        # Calculate alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples according to the noise schedule
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict the sample at the previous timestep by reversing the SDE
        """
        t = timestep
        prev_t = self._get_prev_timestep(t)
        
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        
        # 2. compute predicted original sample from predicted noise also called "predicted x_0"
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"prediction_type {self.prediction_type} is not supported")
            
        # 3. Clip predicted x_0
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
            
        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        pred_sample_direction = (1 - alpha_prod_t_prev - eta ** 2 * beta_prod_t) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=generator).to(device)
            variance = eta * beta_prod_t ** 0.5 * torch.randn_like(model_output)
            prev_sample = prev_sample + variance
            
        if not return_dict:
            return (prev_sample,)
            
        return prev_sample
    
    def _get_prev_timestep(self, timestep: int) -> int:
        """
        Get previous timestep
        """
        return timestep - self.num_diffusion_steps // self.num_diffusion_steps


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999) 