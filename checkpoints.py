import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


@dataclass
class TrainingState:
    epoch: int = 0
    epoch_start_step: int = 0
    step: int = 0
    total_tokens: int = 0
    total_flops: float = 0.0
    start_time: float = -1
    curr_time: float = -1


def save_checkpoint(path, trainer, optimizer, state: TrainingState):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    if hasattr(trainer, "config"):
        OmegaConf.save(config=trainer.config, f=path / "config.yaml", resolve=True)

    torch.save(trainer.model.state_dict(), path / "model.pt")

    if hasattr(trainer, "tokenizer") and trainer.tokenizer is not None:
        try:
            trainer.tokenizer.save_pretrained(path)
        except Exception:
            pass

    if optimizer is not None:
        torch.save(optimizer.state_dict(), path / "optimizer.pt")

    with open(path / "state.json", "w") as f:
        json.dump(asdict(state), f)


def load_checkpoint_for_training(path, config=None, device=None, dtype=None):
    """Load a checkpoint produced by save_checkpoint().

    Expected structure under `path/`:
      - model.pt
      - optimizer.pt (optional)
      - state.json
      - config.yaml (optional)
      - tokenizer files (optional)

    Notes:
    - RWKV backbone is reloaded from config (HF/local) and then the full denoiser
      state_dict is restored.
    """
    ckpt_dir = Path(path)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_dir}")

    if config is None:
        raise ValueError("load_checkpoint_for_training requires `config` for RWKV model reconstruction")

    # Local imports to avoid circular dependencies.
    from rwkv_diffusion_rnn import load_rwkv_model, MultimodalRWKVDenoiser
    from rwkv_diffusion_rnn.diffusion_process import MaskedDiffusion
    from rwkv_diffusion_rnn.improved_trainer import MultimodalDiffusionTrainer_new
    from rwkv_diffusion_rnn.optimizer import get_optimizer

    if dtype is None:
        dtype = torch.float32

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load RWKV backbone + tokenizer (config-driven; offline-friendly).
    rwkv_model, tokenizer = load_rwkv_model(config, dtype=dtype, device=device)

    vocab_size = len(tokenizer)
    model = MultimodalRWKVDenoiser(
        config=config.model,
        rwkv_model=rwkv_model,
        vocab_size=vocab_size,
        latent_dim=config.model.get("latent_dim", 32),
        cluster_size=config.model.get("cluster_size", 0),
    ).to(device=device, dtype=dtype)

    state_path = ckpt_dir / "state.json"
    with open(state_path, "r") as f:
        state_payload = json.load(f)
    state = TrainingState(**{k: state_payload.get(k, getattr(TrainingState(), k)) for k in TrainingState.__dataclass_fields__})

    # Older checkpoints may not track timing fields; keep them consistent.
    if state.start_time < 0:
        state.start_time = -1
    if state.curr_time < 0:
        state.curr_time = state.start_time

    model_path = ckpt_dir / "model.pt"
    model_state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(model_state, strict=False)

    text_noise_schedule = MaskedDiffusion(tokenizer)
    trainer = MultimodalDiffusionTrainer_new(
        model=model,
        tokenizer=tokenizer,
        text_noise_schedule=text_noise_schedule,
        dtype=dtype,
        config=config,
    ).to(device=device)

    optimizer = get_optimizer(config, trainer)
    opt_path = ckpt_dir / "optimizer.pt"
    if opt_path.exists():
        optimizer_state = torch.load(opt_path, map_location="cpu", weights_only=False)
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception:
            # Optimizer schema may change across runs; keep the run usable.
            pass

    old_config = None
    cfg_path = ckpt_dir / "config.yaml"
    if cfg_path.exists():
        try:
            old_config = OmegaConf.load(cfg_path)
        except Exception:
            old_config = None

    return model, text_noise_schedule, tokenizer, old_config, trainer, optimizer, state


def save_rng_state(path: Path, rank: int):
    rng_state_dict = {
        "cpu_rng_state": torch.get_rng_state(),
        "gpu_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "py_rng_state": random.getstate(),
    }
    torch.save(rng_state_dict, Path(path, f"rng_state_{rank}.pt"))


def load_rng_state(path: Path, rank: int):
    rng_state_dict = torch.load(Path(path, f"rng_state_{rank}.pt"), map_location="cpu", weights_only=False)
    torch.set_rng_state(rng_state_dict["cpu_rng_state"])
    if torch.cuda.is_available() and rng_state_dict.get("gpu_rng_state") is not None:
        torch.cuda.set_rng_state(rng_state_dict["gpu_rng_state"])
    np.random.set_state(rng_state_dict["numpy_rng_state"])
    random.setstate(rng_state_dict["py_rng_state"])
