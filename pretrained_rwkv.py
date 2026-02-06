import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_source(model_name: str, local_path: str | None) -> str:
    return local_path if local_path else model_name


def load_rwkv_model(config, dtype: torch.dtype | None = None, device: torch.device | None = None):
    """Load RWKV model + tokenizer with offline-friendly behavior.

    Expects config fields:
      - model.rwkv_name (default BlinkDL/rwkv7-g1)
      - model.rwkv_revision
      - model.rwkv_local_path
      - model.rwkv_trust_remote_code (default False)
      - model.add_mask_token (default True)
      - model.pad_token_strategy (default "eos")
      - tokenizer.cache_dir / tokenizer.local_files_only
    """
    model_cfg = _cfg_get(config, "model", {})
    tok_cfg = _cfg_get(config, "tokenizer", {})
    data_cfg = _cfg_get(config, "data", {})

    model_name = _cfg_get(model_cfg, "rwkv_name", "BlinkDL/rwkv7-g1")
    revision = _cfg_get(model_cfg, "rwkv_revision", None)
    local_path = _cfg_get(model_cfg, "rwkv_local_path", None)
    trust_remote_code = _cfg_get(model_cfg, "rwkv_trust_remote_code", False)
    cache_dir = _cfg_get(tok_cfg, "cache_dir", None) or _cfg_get(model_cfg, "cache_dir", None)
    local_files_only = _cfg_get(tok_cfg, "local_files_only", False)

    add_mask_token = _cfg_get(model_cfg, "add_mask_token", True)
    pad_token_strategy = _cfg_get(model_cfg, "pad_token_strategy", "eos")

    source = _resolve_source(model_name, local_path)

    if local_files_only and local_path is not None and not os.path.exists(local_path):
        raise FileNotFoundError(
            f"RWKV local path not found: {local_path}. "
            "Set model.rwkv_local_path to a valid directory or disable tokenizer.local_files_only."
        )

    allow_dummy_tokenizer = _cfg_get(model_cfg, "allow_dummy_tokenizer", False)
    tokenizer = None
    tokenizer_error = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    except OSError as exc:
        tokenizer_error = exc
        if not allow_dummy_tokenizer:
            if local_files_only:
                raise FileNotFoundError(
                    f"RWKV tokenizer not found locally for source '{source}'. "
                    "Set tokenizer.local_files_only=false or download weights to cache_dir."
                ) from exc
            raise

    try:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
    except OSError as exc:
        if local_files_only:
            raise FileNotFoundError(
                f"RWKV weights not found locally for source '{source}'. "
                "Set tokenizer.local_files_only=false or download weights to cache_dir."
            ) from exc
        raise

    if _cfg_get(model_cfg, "rwkv_use_cache", False):
        model.config.use_cache = True
    else:
        model.config.use_cache = False
    model.config.output_hidden_states = True

    if tokenizer is None:
        vocab_size = getattr(model.config, "vocab_size", None) or _cfg_get(model_cfg, "vocab_size", 50277)

        class _DummyTokenizer:
            def __init__(self, vocab_size: int, max_length: int | None):
                self.vocab_size = vocab_size
                self.mask_token_id = 0
                self.pad_token_id = 0
                self.eos_token_id = 0
                self.mask_token = "<mask>"
                self.pad_token = "<pad>"
                self.eos_token = "</s>"
                self.model_max_length = max_length or 512
                self.padding_side = "right"
                self.truncation_side = "right"

            def __len__(self):
                return self.vocab_size

            def save_pretrained(self, *args, **kwargs):
                return None

        max_length = _cfg_get(data_cfg, "max_length", None)
        tokenizer = _DummyTokenizer(vocab_size, max_length)
        print(f"Using dummy tokenizer (vocab_size={vocab_size}) due to tokenizer load error: {tokenizer_error}")
    else:
        # Ensure padding token exists for batching
        if tokenizer.pad_token_id is None:
            if pad_token_strategy == "eos" and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                model.resize_token_embeddings(len(tokenizer))

        # Ensure mask token exists for masked diffusion
        if add_mask_token and tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})
            model.resize_token_embeddings(len(tokenizer))

        max_length = _cfg_get(data_cfg, "max_length", None)
        if max_length is not None:
            tokenizer.model_max_length = max_length
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

    if device is not None:
        model = model.to(device=device)

    print(
        "Loaded RWKV model from "
        f"{'local path' if local_path else 'HF repo'}: {source}"
    )

    return model, tokenizer
