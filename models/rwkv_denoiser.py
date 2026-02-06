from __future__ import annotations

import torch
import torch.nn as nn

from rwkv_diffusion_rnn.time_conditioning import DualModalityConditioning


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class LatentPrefixProjector(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, prefix_len: int, dropout: float = 0.0):
        super().__init__()
        self.prefix_len = prefix_len
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * prefix_len),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B, D] or [B, L, D]
        if latents.dim() == 3:
            latents = latents.mean(dim=1)
        projected = self.net(latents)
        projected = projected.view(latents.shape[0], self.prefix_len, self.hidden_size)
        return self.norm(projected)


class MultimodalRWKVDenoiser(nn.Module):
    """RWKV-based denoiser with latent prefix + text logits."""

    def __init__(
        self,
        config,
        rwkv_model,
        vocab_size: int,
        latent_dim: int,
        cluster_size: int = 0,
    ):
        super().__init__()
        self.config = config
        self.rwkv_model = rwkv_model
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.cluster_size = cluster_size

        hidden_size = _cfg_get(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(rwkv_model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("RWKV hidden size not found; set model.hidden_size in config")
        self.hidden_size = hidden_size

        self.latent_prefix_len = _cfg_get(config, "latent_prefix_len", 1)
        self.time_conditioning = _cfg_get(config, "time_conditioning", "film")
        self.use_rwkv_lm_head = _cfg_get(config, "use_rwkv_lm_head", True)
        self.latent_pred_from = _cfg_get(config, "latent_pred_from", "prefix")

        dropout = _cfg_get(config, "dropout", 0.0)

        self.latent_projector = LatentPrefixProjector(
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            prefix_len=self.latent_prefix_len,
            dropout=dropout,
        )
        self.latent_pos_embed = nn.Parameter(
            torch.zeros(1, self.latent_prefix_len, hidden_size)
        )

        # Token type embeddings: 0=text, 1=latent
        self.token_type_embed = nn.Embedding(2, hidden_size)

        self.conditioning = DualModalityConditioning(hidden_size)
        if self.time_conditioning == "film":
            self.time_film = nn.Linear(hidden_size, hidden_size * 2)
        else:
            self.time_film = None

        # Heads
        self.text_head = nn.Linear(hidden_size, vocab_size)
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, latent_dim),
        )

        if cluster_size > 0:
            self.text_head_clusters = nn.Linear(hidden_size, vocab_size)
        else:
            self.text_head_clusters = None

        self._maybe_freeze_backbone()

    def _maybe_freeze_backbone(self) -> None:
        freeze_backbone = _cfg_get(self.config, "freeze_backbone", True)
        train_lm_head = _cfg_get(self.config, "train_lm_head", False)

        if freeze_backbone:
            for param in self.rwkv_model.parameters():
                param.requires_grad = False

        if train_lm_head:
            lm_head = getattr(self.rwkv_model, "lm_head", None)
            if lm_head is not None:
                for param in lm_head.parameters():
                    param.requires_grad = True

    def _apply_time_conditioning(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = cond.to(dtype=x.dtype)
        if self.time_conditioning == "film" and self.time_film is not None:
            gamma_beta = self.time_film(cond)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            return x * (1 + gamma[:, None, :]) + beta[:, None, :]
        return x + cond[:, None, :]

    def forward(
        self,
        text_tokens: torch.Tensor,
        latents: torch.Tensor | None,
        text_timesteps: torch.Tensor | None,
        latent_timesteps: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ):
        if text_tokens is None:
            raise ValueError("text_tokens must be provided for RWKV denoiser")

        device = text_tokens.device
        text_emb = self.rwkv_model.get_input_embeddings()(text_tokens)
        text_emb = text_emb.to(dtype=next(self.parameters()).dtype)

        prefix_emb = None
        if latents is not None:
            latents = latents.to(device=device, dtype=text_emb.dtype)
            prefix_emb = self.latent_projector(latents)
            prefix_emb = prefix_emb + self.latent_pos_embed.to(dtype=text_emb.dtype)

        # Token type embeddings
        if prefix_emb is not None:
            prefix_type = self.token_type_embed(
                torch.ones(prefix_emb.shape[:2], device=device, dtype=torch.long)
            )
            prefix_emb = prefix_emb + prefix_type
        text_type = self.token_type_embed(
            torch.zeros(text_emb.shape[:2], device=device, dtype=torch.long)
        )
        text_emb = text_emb + text_type

        if prefix_emb is not None:
            combined = torch.cat([prefix_emb, text_emb], dim=1)
        else:
            combined = text_emb

        cond = self.conditioning(text_timesteps, latent_timesteps)
        combined = self._apply_time_conditioning(combined, cond)

        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(text_tokens.shape, device=device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if prefix_emb is not None:
            prefix_mask = torch.ones(prefix_emb.shape[:2], device=device, dtype=torch.bool)
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            combined_mask = attention_mask

        try:
            outputs = self.rwkv_model(
                inputs_embeds=combined,
                attention_mask=combined_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        except TypeError as exc:
            raise TypeError(
                "RWKV model did not accept inputs_embeds/attention_mask. "
                "Use a checkpoint that supports inputs_embeds or adjust the wrapper."
            ) from exc

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            raise RuntimeError("RWKV outputs missing hidden states")

        prefix_len = prefix_emb.shape[1] if prefix_emb is not None else 0
        if hasattr(outputs, "logits") and outputs.logits is not None and self.use_rwkv_lm_head:
            text_logits = outputs.logits[:, prefix_len:, :]
        else:
            text_logits = self.text_head(hidden_states[:, prefix_len:, :])

        latent_pred = None
        if prefix_len > 0:
            if self.latent_pred_from == "prefix":
                latent_hidden = hidden_states[:, :prefix_len, :].mean(dim=1)
            elif self.latent_pred_from == "text":
                text_seq_len = hidden_states.shape[1] - prefix_len
                if text_seq_len > 0:
                    latent_hidden = hidden_states[:, prefix_len:, :].mean(dim=1)
                else:
                    latent_hidden = hidden_states[:, :prefix_len, :].mean(dim=1)
            elif self.latent_pred_from == "both":
                prefix_hidden = hidden_states[:, :prefix_len, :].mean(dim=1)
                text_seq_len = hidden_states.shape[1] - prefix_len
                if text_seq_len > 0:
                    text_hidden = hidden_states[:, prefix_len:, :].mean(dim=1)
                    latent_hidden = prefix_hidden + text_hidden
                else:
                    latent_hidden = prefix_hidden
            else:
                latent_hidden = hidden_states[:, :prefix_len, :].mean(dim=1)
            
            latent_pred = self.latent_head(latent_hidden)

        if self.cluster_size > 0 and self.text_head_clusters is not None:
            text_cluster_logits = self.text_head_clusters(hidden_states[:, prefix_len:, :])
            return text_logits, latent_pred, text_cluster_logits

        return text_logits, latent_pred
