# DiffRwkv

This folder is a **self-contained diffusion RNN pipeline** built around RWKV weights. It bundles:
- RWKV loader + denoiser wrapper
- Diffusion schedules (masked text + continuous latent)
- Trainer, optimizer, data loader, and utilities
- Configs + evaluation checklist
- A dedicated training entrypoint (`train_rwkv.py`)

> Goal: train a diffusion-style RWKV denoiser on **text <-> latent** using **Qwen 32-dim latents**.

---

## Layout
```
rwkv_diffusion_rnn/
  configs/rwkv_mmdit.yaml
  train_rwkv.py
  pretrained_rwkv.py
  time_conditioning.py
  models/rwkv_denoiser.py
  diffusion_process.py
  continuous_diffusion.py
  improved_trainer.py
  data_simple.py
  optimizer.py
  utils.py
  checkpoints.py
  preprocess_tokenize_text.py
  eval_checklist.md
```

---

## 1) Prepare latent data (Qwen embeddings, 32-dim)

This pipeline assumes **Qwen embeddings already projected to 32-dim**.
If your current Qwen embeddings are 1024-dim, project them to 32-dim first.

Expected layout:
```
preprocessed_data/qwen_32d/
  latents/train/*.npy   # each latent is 32-dim
  train_data.json       # text + latent_path entries
```

---

## 2) Tokenize text with RWKV tokenizer

Use the provided tokenizer script to produce `.npz` files that match the JSON list:

```bash
python rwkv_diffusion_rnn/preprocess_tokenize_text.py \
  --json_path preprocessed_data/qwen_32d/train_data.json \
  --output_dir preprocessed_data/rwkv_tokens \
  --tokenizer BlinkDL/rwkv7-g1 \
  --max_length 512

Notes:
- If the tokenizer requires remote code, add `--trust-remote-code` (only in trusted envs).
- If you want an offline-only run, add `--local-files-only` and point `--cache-dir` to a populated HF cache.
```

Expected layout:
```
preprocessed_data/rwkv_tokens/
  train/*.npz           # input_ids + attention_mask
```

---

## 3) Configure paths

Edit `rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml`:
- `data.token_dir`: path to `preprocessed_data/rwkv_tokens/train`
- `data.latent_dir`: path to `preprocessed_data/qwen_32d/latents/train`
- `model.latent_dim`: should be `32`

---

## 4) Start training (smoke run)

```bash
HF_HOME=/tmp/hf_cache python rwkv_diffusion_rnn/train_rwkv.py \
  --config-name rwkv_mmdit \
  training.num_epochs=1 \
  training.train_batch_size=1 \
  training.eval_batch_size=1 \
  training.dtype=fp32 \
  data.num_workers=0 \
  data.max_samples=10 \
  logging.log_freq=1 \
  logging.eval_freq=100000 \
  tokenizer.local_files_only=false
```

Notes:
- `tokenizer.local_files_only=false` lets Transformers download weights if not cached.
- Use `HF_HOME` or `TRANSFORMERS_CACHE` to control cache location.
- On macOS CPU, this is **only for smoke testing** (very slow).

Hydra working directory:
- Hydra may run jobs from a per-run output directory depending on `version_base` and config.
- This module sets `hydra.job.chdir: false` in `rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml` so relative paths like `preprocessed_data/...` stay stable.

Security note (trust_remote_code):
- If you need `trust_remote_code=true`, prefer pinning `model.rwkv_revision` to a specific revision and only run in a trusted environment.

---

## 5) Full training

Increase:
- `training.num_epochs`
- `training.train_batch_size`
- `data.max_samples`

Keep `freeze_backbone: true` at first if you want to preserve pretrained RWKV behavior.

---

## Key RWKV settings

In `configs/rwkv_mmdit.yaml`:
- `model.rwkv_name: "BlinkDL/rwkv7-g1"`
- `model.freeze_backbone: true`
- `model.use_rwkv_lm_head: false` (train a fresh text head)
- `model.latent_dim: 32`
- `model.latent_prefix_len: 1`
- `model.time_conditioning: film`

---

## Troubleshooting

**Tokenizer errors**
- Add `tokenizer.local_files_only=false` or download weights into the cache.

**Weight loading errors**
- If the HF repo does not expose standard `pytorch_model.bin` / `model.safetensors`, you must provide local RWKV weights or add a custom loader.

**Dim mismatch**
- Ensure latent `.npy` vectors are 32-dim. Otherwise training will crash.

**Resume**
- Set `training.resume=/path/to/checkpoint_dir` (a directory created under `logging.save_dir/epoch_XXX`).

---

## Eval

See `rwkv_diffusion_rnn/eval_checklist.md` for a minimal regression checklist.
