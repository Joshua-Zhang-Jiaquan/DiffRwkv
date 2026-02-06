import datetime
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import wandb
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

# Allow running from repo root or subdirs
sys.path.append("..")
sys.path.append(".")
os.environ["WANDB_MODE"] = "disabled"

from rwkv_diffusion_rnn import load_rwkv_model, MultimodalRWKVDenoiser
from rwkv_diffusion_rnn.checkpoints import (
    save_checkpoint,
    load_checkpoint_for_training,
    TrainingState,
    save_rng_state,
    load_rng_state,
)
from rwkv_diffusion_rnn.optimizer import get_optimizer
from rwkv_diffusion_rnn.utils import (
    get_lr,
    parse_dtype,
)
from rwkv_diffusion_rnn.data_simple import get_simple_dataloaders
from rwkv_diffusion_rnn.diffusion_process import MaskedDiffusion
from rwkv_diffusion_rnn.improved_trainer import MultimodalDiffusionTrainer_new


class Logger:
    def __init__(self, is_main_process: bool):
        self.is_main_process = is_main_process

    def init(self, *args, **kwargs):
        if self.is_main_process:
            wandb.init(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.is_main_process:
            wandb.log(*args, **kwargs)


def safe_barrier(local_rank: int | None = None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        if local_rank is not None:
            dist.barrier(device_ids=[local_rank])  # type: ignore[arg-type]
        else:
            dist.barrier()
    except TypeError:
        dist.barrier()


@contextmanager
def main_process_first(local_rank: int | None = None):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            yield
            safe_barrier(local_rank)
        else:
            safe_barrier(local_rank)
            yield
    else:
        yield


def _init_distributed() -> tuple[int, int, int, bool, bool]:
    env_has_ddp = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if not env_has_ddp:
        return 0, 0, 1, True, False

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch detected (torchrun), but CUDA is not available.")

    local_rank = int(os.environ["LOCAL_RANK"])
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {num_gpus} GPU(s) available. "
            f"Please ensure --nproc_per_node <= {num_gpus} and CUDA_VISIBLE_DEVICES is set correctly."
        )

    torch.cuda.set_device(local_rank)
    init_kwargs = dict(
        backend="nccl",
        timeout=datetime.timedelta(minutes=30),
        init_method="env://",
    )
    try:
        dist.init_process_group(**init_kwargs, device_id=torch.device("cuda", local_rank))  # type: ignore[arg-type]
    except TypeError:
        dist.init_process_group(**init_kwargs)

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = (global_rank == 0)
    is_distributed = dist.is_available() and dist.is_initialized()
    return local_rank, global_rank, world_size, is_main_process, is_distributed


@hydra.main(config_path="configs", config_name="rwkv_mmdit", version_base="1.1")
def main(config):
    local_rank, global_rank, world_size, is_main_process, is_distributed = _init_distributed()

    with open_dict(config):
        config.training.world_size = world_size
        config.training.local_rank = local_rank
        config.training.global_rank = global_rank

    seed = config.training.seed + global_rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if hasattr(torch.backends.cudnn, 'allow_bf16_reduced_precision_reduction'):
        torch.backends.cudnn.allow_bf16_reduced_precision_reduction = True
    print("BF16 optimizations enabled")

    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass

    dtype = parse_dtype(config.training.dtype)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    print(f"Using device={device} and dtype={dtype}")

    tokenizer = None
    if config.training.resume is None:
        rwkv_model, tokenizer = load_rwkv_model(config, dtype=dtype, device=device)
        vocab_size = len(tokenizer)
        model = MultimodalRWKVDenoiser(
            config=config.model,
            rwkv_model=rwkv_model,
            vocab_size=vocab_size,
            latent_dim=config.model.get("latent_dim", 32),
            cluster_size=config.model.get("cluster_size", 0),
        ).to(device=device, dtype=dtype)

        text_noise_schedule = MaskedDiffusion(tokenizer)

        trainer = MultimodalDiffusionTrainer_new(
            model=model,
            tokenizer=tokenizer,
            text_noise_schedule=text_noise_schedule,
            dtype=dtype,
            config=config
        ).to(device=device)

        optimizer = get_optimizer(config, trainer)

        state = TrainingState(
            epoch=0,
            epoch_start_step=0,
            step=0,
        )
    else:
        (
            model,
            text_noise_schedule,
            tokenizer,
            old_config,
            trainer,
            optimizer,
            state,
        ) = load_checkpoint_for_training(config.training.resume, config=config, device=device, dtype=dtype)

    with main_process_first(local_rank):
        train_dl, test_dl = get_simple_dataloaders(config, tokenizer=tokenizer)

    # Fill missing scheduler fields after dataloader is known.
    with open_dict(config):
        if config.training.get("num_train_steps", None) is None:
            config.training.num_train_steps = int(config.training.num_epochs * len(train_dl))

    logger = Logger(is_main_process)
    os.environ.setdefault("WANDB_DIR", config.logging.get("wandb_dir", "./outputs/"))
    logger.init(
        name=config.logging.run_name,
        entity=config.logging.wandb_entity,
        project=config.logging.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
    )

    if is_main_process:
        pwd = Path(".").resolve()
        wandb.config.update({"pwd": pwd})
        print(f"Working directory: {pwd}")

    non_emb_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)

    if config.training.compile_model:
        try:
            opt_trainer = torch.compile(trainer)
        except RuntimeError as e:
            if "Python 3.13" in str(e):
                print("Warning: torch.compile not supported on Python 3.13+, skipping compilation")
                opt_trainer = trainer
            else:
                raise
    else:
        opt_trainer = trainer

    if is_distributed:
        find_unused = bool(config.training.get("ddp_find_unused_parameters", False))
        ddp_trainer = DDP(
            opt_trainer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )
    else:
        ddp_trainer = opt_trainer

    if is_main_process:
        non_emb_params_str = (
            f"{non_emb_params / 1e6:.1f}M" if non_emb_params < 500 * 1e6 else f"{non_emb_params / 1e9:.1f}B"
        )
        trainable_params_str = (
            f"{trainable_params / 1e6:.1f}M" if trainable_params < 500 * 1e6 else f"{trainable_params / 1e9:.1f}B"
        )
        print("*** Starting RWKV Denoiser Training ***")
        print(f"* World size: {world_size}")
        print(f"* Per-device batch size: {config.training.train_batch_size}")
        print(f"* Total batch size: {config.training.train_batch_size * world_size}")
        print(f"* Non-embedding parameters: {non_emb_params_str}")
        print(f"* Trainable parameters: {trainable_params_str}")
        print(f"* Model dtype: {next(iter(model.parameters())).dtype}")
        print(f"* Latent dimension: {config.model.get('latent_dim', 32)}")
        print("* Text diffusion: Masked Diffusion")
        print(
            f"* Latent diffusion: Continuous Diffusion "
            f"(beta={config.model.get('latent_beta_min', 0.0001)}-{config.model.get('latent_beta_max', 0.02)})"
        )
        print("*************************")

    if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
        train_dl.sampler.set_epoch(state.epoch)

    if state.start_time < 0:
        state.start_time = time.time()
        state.curr_time = state.start_time

    trained_time = 0 if config.training.resume is None else (state.start_time - state.curr_time)

    if config.training.resume is not None:
        print(f"Resuming from checkpoint: {config.training.resume}")

    log_buffer = []

    for epoch in range(state.epoch, config.training.num_epochs):
        if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(epoch)

        state.epoch = epoch
        state.epoch_start_step = state.step

        ddp_trainer.train()

        prev_time = time.time()

        for batch_idx, batch in enumerate(train_dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            loss, metrics = ddp_trainer(batch, step=state.step)

            curr_lr = get_lr(config, config.optimizer.lr, state.step)
            for param_group in optimizer.param_groups:
                if "lr" in param_group:
                    param_group["lr"] = curr_lr

            (loss * config.loss.loss_scale).backward()

            if config.optimizer.grad_clip_norm and config.optimizer.grad_clip_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.optimizer.grad_clip_norm)
            else:
                norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1e6)

            if torch.isnan(norm):
                print(f"Warning: NaN gradient detected at step {state.step}")
                for param in trainer.parameters():
                    if param.grad is not None:
                        param.grad.data.zero_()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            batch_tokens = (
                batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
                .sum()
                .item()
                * config.training.world_size
            )
            total_batch_size = batch["input_ids"].size(0) * config.training.world_size

            state.total_tokens += batch_tokens

            curr_time = time.time()
            step_time = curr_time - prev_time
            prev_time = curr_time

            log_buffer.append(
                {
                    "train/loss": float(loss.item()),
                    "train/lr": float(curr_lr),
                    "train/step": int(state.step + 1),
                    "train/grad_norm": float(norm.item()),
                    "train/epoch": float(state.epoch + (state.step - state.epoch_start_step) / len(train_dl)),
                    "train/total_tokens": float(state.total_tokens),
                    "train/tokens_per_sec": float(batch_tokens / step_time),
                    "train/samples_per_sec": float(total_batch_size / step_time),
                    "train/it_per_sec": float(1.0 / step_time),
                    "train/avg_it_per_sec": float((state.step + 1) / (curr_time - state.start_time)),
                    **{f"train/{k}": v for k, v in metrics.items()},
                }
            )

            if ((state.step + 1) % config.logging.log_freq) == 0:
                avg_metrics = {k: sum(d[k] for d in log_buffer) / len(log_buffer) for k in log_buffer[0]}
                logger.log(avg_metrics, step=state.step)
                logger.log({"trainer/global_step": state.step}, step=state.step)
                log_buffer = []

            state.step += 1

        if is_main_process and ((epoch + 1) % 1 == 0):
            save_path = Path(config.logging.save_dir) / f"epoch_{epoch + 1:03d}"
            save_checkpoint(save_path, trainer, optimizer, state)


if __name__ == "__main__":
    main()
