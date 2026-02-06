import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(dtype=t.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DualModalityConditioning(nn.Module):
    """Combine text and latent timesteps into a single conditioning vector."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.text_embed = TimestepEmbedder(hidden_size)
        self.latent_embed = TimestepEmbedder(hidden_size)
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self,
        text_timesteps: torch.Tensor | None,
        latent_timesteps: torch.Tensor | None,
    ) -> torch.Tensor:
        model_param = next(self.parameters())
        model_dtype = model_param.dtype
        device = model_param.device

        if text_timesteps is None:
            if latent_timesteps is not None:
                batch_size = latent_timesteps.shape[0]
            else:
                batch_size = 1
            text_timesteps = torch.zeros(batch_size, device=device, dtype=model_dtype)
        else:
            text_timesteps = text_timesteps.to(dtype=model_dtype, device=device)

        if latent_timesteps is None:
            batch_size = text_timesteps.shape[0]
            latent_timesteps = torch.zeros(batch_size, device=device, dtype=model_dtype)
        else:
            latent_timesteps = latent_timesteps.to(dtype=model_dtype, device=device)

        text_cond = F.silu(self.text_embed(text_timesteps))
        latent_cond = F.silu(self.latent_embed(latent_timesteps))
        combined = torch.cat([text_cond, latent_cond], dim=-1)
        return self.combine(combined)
