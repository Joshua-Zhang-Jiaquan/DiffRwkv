import torch
import torch.nn.functional as F
from typing import Optional, Tuple


@torch.no_grad()
def sample_text_from_latent(
    model,
    latent: torch.Tensor,
    seq_len: int,
    num_steps: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: str = "cuda"
) -> torch.Tensor:
    model.eval()
    batch_size = latent.shape[0]
    
    if latent.dim() == 2:
        latent = latent.unsqueeze(1)
    
    latent = latent.to(device)
    
    noisy_tokens = torch.randint(
        0, model.vocab_size, 
        (batch_size, seq_len), 
        device=device
    )
    
    timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)
    
    for i, t in enumerate(timesteps):
        t_batch = t.expand(batch_size)
        
        logits = model.rwkv_denoiser(
            noisy_tokens,
            t_batch,
            latent_condition=latent,
            mode="l2t"
        )
        
        if temperature > 0:
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()
                sorted_indices_to_remove[:, :, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, seq_len)
        else:
            next_tokens = torch.argmax(logits, dim=-1)
        
        if i < num_steps - 1:
            alpha = (num_steps - i - 1) / num_steps
            noisy_tokens = (alpha * noisy_tokens + (1 - alpha) * next_tokens).long()
        else:
            noisy_tokens = next_tokens
    
    return noisy_tokens


@torch.no_grad()
def sample_latent_from_text(
    model,
    input_ids: torch.Tensor,
    latent_dim: int,
    num_steps: int = 100,
    guidance_scale: float = 1.0,
    device: str = "cuda"
) -> torch.Tensor:
    model.eval()
    batch_size = input_ids.shape[0]
    
    input_ids = input_ids.to(device)
    
    noisy_latent = torch.randn(batch_size, 1, latent_dim, device=device)
    
    timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)
    
    for i, t in enumerate(timesteps):
        t_batch = t.expand(batch_size)
        
        pred_latent = model.rwkv_denoiser(
            input_ids,
            t_batch,
            latent_input=noisy_latent,
            mode="t2l"
        )
        
        if guidance_scale != 1.0:
            pred_latent_uncond = model.rwkv_denoiser(
                input_ids,
                t_batch,
                latent_input=noisy_latent,
                mode="unconditional"
            )
            pred_latent = pred_latent_uncond + guidance_scale * (pred_latent - pred_latent_uncond)
        
        if i < num_steps - 1:
            alpha = (num_steps - i - 1) / num_steps
            noisy_latent = alpha * noisy_latent + (1 - alpha) * pred_latent
        else:
            noisy_latent = pred_latent
    
    return noisy_latent


@torch.no_grad()
def sample_joint(
    model,
    batch_size: int,
    seq_len: int,
    latent_dim: int,
    num_steps: int = 100,
    temperature: float = 1.0,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    
    noisy_tokens = torch.randint(
        0, model.vocab_size,
        (batch_size, seq_len),
        device=device
    )
    noisy_latent = torch.randn(batch_size, 1, latent_dim, device=device)
    
    timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)
    
    for i, t in enumerate(timesteps):
        t_batch = t.expand(batch_size)
        
        text_logits, pred_latent = model.rwkv_denoiser(
            noisy_tokens,
            t_batch,
            latent_input=noisy_latent,
            mode="unconditional"
        )
        
        if temperature > 0:
            text_logits = text_logits / temperature
            probs = F.softmax(text_logits, dim=-1)
            next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, seq_len)
        else:
            next_tokens = torch.argmax(text_logits, dim=-1)
        
        if i < num_steps - 1:
            alpha = (num_steps - i - 1) / num_steps
            noisy_tokens = (alpha * noisy_tokens + (1 - alpha) * next_tokens).long()
            noisy_latent = alpha * noisy_latent + (1 - alpha) * pred_latent
        else:
            noisy_tokens = next_tokens
            noisy_latent = pred_latent
    
    return noisy_tokens, noisy_latent
