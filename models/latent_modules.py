import torch
import torch.nn as nn


class TextLatentEncoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        self.mu_head = nn.Linear(hidden_size, latent_dim)
        self.logvar_head = nn.Linear(hidden_size, latent_dim)
    
    def forward(self, text_hidden: torch.Tensor):
        encoded = self.encoder(text_hidden)
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)
        return mu, logvar


class LatentPrior(nn.Module):
    def __init__(self, latent_dim: int, prior_type: str = "fixed_gaussian"):
        super().__init__()
        self.latent_dim = latent_dim
        self.prior_type = prior_type
        
        if prior_type == "learned_gaussian":
            self.mu = nn.Parameter(torch.zeros(latent_dim))
            self.logvar = nn.Parameter(torch.zeros(latent_dim))
        elif prior_type == "fixed_gaussian":
            self.register_buffer("mu", torch.zeros(latent_dim))
            self.register_buffer("logvar", torch.zeros(latent_dim))
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")
    
    def forward(self, batch_size: int, device: torch.device):
        mu = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)
        logvar = self.logvar.unsqueeze(0).expand(batch_size, -1).to(device)
        return mu, logvar


def kl_divergence(mu_q: torch.Tensor, logvar_q: torch.Tensor, 
                  mu_p: torch.Tensor, logvar_p: torch.Tensor):
    kl = 0.5 * (logvar_p - logvar_q + 
                (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p) - 1)
    return kl.sum(dim=-1).mean()


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
