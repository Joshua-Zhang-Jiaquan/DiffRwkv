# File: latentDLM_mmdit/continuous_diffusion.py

import torch
import torch.nn.functional as F
import math
import numpy as np


class ContinuousDiffusion:
    """
    Improved continuous diffusion for latent vectors.
    
    Features:
    - Multiple beta schedules (linear, cosine, sigmoid)
    - Multiple parameterizations (epsilon, x0, v_param)
    - Proper timestep handling (both discrete and continuous)
    - GPU-optimized precomputation
    """
    
    def __init__(self, 
                 num_timesteps=1000,
                 beta_schedule="cosine",
                 beta_start=0.0001,
                 beta_end=0.02,
                 parameterization="epsilon",  # "epsilon", "v_param", "x0"
                 device="cuda"):
        
        self.num_timesteps = num_timesteps
        self.parameterization = parameterization
        self.device = device
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == "sigmoid":
            self.betas = self._sigmoid_beta_schedule(num_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Move to device
        self.betas = self.betas.to(device)
        
        # Precompute useful values (all on GPU)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling (noise prediction)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse process
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # For v-parameterization
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - self.alphas_cumprod_prev)
        
        print("ContinuousDiffusion initialized")
        print(f"  Schedule: {beta_schedule}")
        print(f"  Timesteps: {num_timesteps}")
        print(f"  Parameterization: {parameterization}")
        print(f"  Beta range: [{self.betas.min():.6f}, {self.betas.max():.6f}]")
        print(f"  Device: {device}")
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        Better than linear schedule for most applications.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _sigmoid_beta_schedule(self, timesteps, beta_start, beta_end):
        """Sigmoid schedule - smooth transitions at beginning/end."""
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    
    def sample_timesteps(self, batch_size, device=None):
        """Sample random discrete timesteps [0, num_timesteps-1]."""
        if device is None:
            device = self.device
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
    
    def sample_continuous_timesteps(self, batch_size, device=None):
        """Sample continuous timesteps [0, 1] - your preferred approach."""
        if device is None:
            device = self.device
        return torch.rand(batch_size, device=device)
    
    def get_schedule_values(self, t, device=None):
        """
        Get schedule values for given timesteps.
        Handles both discrete timesteps and continuous [0,1] timesteps.
        """
        if device is None:
            device = t.device
            
        # Ensure our schedules are on the right device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        if t.max() <= 1.0:  # Continuous timesteps [0, 1]
            # Convert to discrete timesteps
            discrete_t = (t * (self.num_timesteps - 1)).long().clamp(0, self.num_timesteps - 1)
        else:  # Already discrete
            discrete_t = t.long().clamp(0, self.num_timesteps - 1)
        
        sqrt_alpha_bar = sqrt_alphas_cumprod[discrete_t]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[discrete_t]
        
        return sqrt_alpha_bar, sqrt_one_minus_alpha_bar
    
    def add_noise(self, x0, t, noise=None):
        """
        Add noise to clean samples using the forward diffusion process.
        
        Args:
            x0: Clean samples [batch, ...]
            t: Timesteps [batch]
            noise: Optional noise tensor, if None will sample random noise
            
        Returns:
            xt: Noisy samples
            target: Target for training (depends on parameterization)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Get schedule values
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_schedule_values(t, x0.device)
        
        # Reshape for broadcasting
        while sqrt_alpha_bar.dim() < x0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        # Forward diffusion: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        
        # Return target based on parameterization
        if self.parameterization == "epsilon":
            target = noise
        elif self.parameterization == "x0":
            target = x0
        elif self.parameterization == "v_param":
            # v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x0
            target = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * x0
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        
        return xt, target
    
    def predict_x0_from_eps(self, xt, t, eps):
        """Predict x0 from epsilon prediction."""
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_schedule_values(t, xt.device)
        
        while sqrt_alpha_bar.dim() < xt.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        return (xt - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar
    
    def predict_eps_from_x0(self, xt, t, x0):
        """Predict epsilon from x0 prediction."""
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_schedule_values(t, xt.device)
        
        while sqrt_alpha_bar.dim() < xt.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        return (xt - sqrt_alpha_bar * x0) / sqrt_one_minus_alpha_bar
    
    def predict_v_from_x0_eps(self, x0, eps, t):
        """Predict v-param from x0 and epsilon."""
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_schedule_values(t, x0.device)
        
        while sqrt_alpha_bar.dim() < x0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        return sqrt_alpha_bar * eps - sqrt_one_minus_alpha_bar * x0
    
    def predict_x0_from_v(self, xt, t, v):
        """Predict x0 from v-param prediction."""
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_schedule_values(t, xt.device)
        
        while sqrt_alpha_bar.dim() < xt.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        return sqrt_alpha_bar * xt - sqrt_one_minus_alpha_bar * v
    
    def predict_eps_from_v(self, xt, t, v):
        """Predict epsilon from v-param prediction."""
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_schedule_values(t, xt.device)
        
        while sqrt_alpha_bar.dim() < xt.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        return sqrt_alpha_bar * v + sqrt_one_minus_alpha_bar * xt
    
    def get_loss(self, model_output, target, loss_type="mse"):
        """Compute loss between model output and target."""
        if loss_type == "mse":
            return F.mse_loss(model_output, target)
        elif loss_type == "l1":
            return F.l1_loss(model_output, target)
        elif loss_type == "huber":
            return F.huber_loss(model_output, target)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def ddim_step(self, xt, t, t_prev, pred_noise, eta=0.0):
        """
        DDIM sampling step for inference.
        
        Args:
            xt: Current noisy sample
            t: Current timestep  
            t_prev: Previous timestep
            pred_noise: Predicted noise from model
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)
        """
        # Get schedule values
        sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = self.get_schedule_values(t, xt.device)
        sqrt_alpha_bar_prev, sqrt_one_minus_alpha_bar_prev = self.get_schedule_values(t_prev, xt.device)
        
        # Reshape for broadcasting
        while sqrt_alpha_bar_t.dim() < xt.dim():
            sqrt_alpha_bar_t = sqrt_alpha_bar_t.unsqueeze(-1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.unsqueeze(-1)
            sqrt_alpha_bar_prev = sqrt_alpha_bar_prev.unsqueeze(-1)
            sqrt_one_minus_alpha_bar_prev = sqrt_one_minus_alpha_bar_prev.unsqueeze(-1)
        
        # Predict x0
        x0_pred = (xt - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
        
        # Direction pointing towards xt
        dir_xt = sqrt_one_minus_alpha_bar_prev * pred_noise
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(xt)
            sigma_t = eta * torch.sqrt(sqrt_one_minus_alpha_bar_prev / sqrt_one_minus_alpha_bar_t) * torch.sqrt(1 - sqrt_alpha_bar_t / sqrt_alpha_bar_prev)
            dir_xt += sigma_t * noise
        
        # Compute x_{t-1}
        x_prev = sqrt_alpha_bar_prev * x0_pred + dir_xt
        
        return x_prev, x0_pred


# Wrapper for backward compatibility
class SimpleContinuousDiffusion(ContinuousDiffusion):
    """Simplified interface matching your current usage."""
    
    def __init__(self, beta_min=0.0001, beta_max=0.02, device="cuda"):
        super().__init__(
            num_timesteps=1000,
            beta_schedule="cosine",  # Better than linear
            beta_start=beta_min,
            beta_end=beta_max,
            parameterization="epsilon",
            device=device
        )
    
    def get_alpha_beta(self, t):
        """Legacy method for compatibility."""
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_schedule_values(t)
        alpha_bar = sqrt_alpha_bar ** 2
        return alpha_bar, 1 - alpha_bar
