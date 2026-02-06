# rwkv_diffusion_rnn

## OVERVIEW
Self-contained diffusion RNN pipeline built around RWKV weights (text<->latent with 32-d latents).

## STRUCTURE
- rwkv_diffusion_rnn/train_rwkv.py: Hydra entrypoint
- rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml: config
- rwkv_diffusion_rnn/pretrained_rwkv.py: RWKV loader
- rwkv_diffusion_rnn/data_simple.py: dataloader
- rwkv_diffusion_rnn/improved_trainer.py: trainer

## WHERE TO LOOK
Task | Location | Notes
train | rwkv_diffusion_rnn/train_rwkv.py | Hydra entrypoint; DDP init requires CUDA
config | rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml | points at token_dir + latent_dir
RWKV loader | rwkv_diffusion_rnn/pretrained_rwkv.py | `rwkv_trust_remote_code` + `local_files_only` handling
tokenize | rwkv_diffusion_rnn/preprocess_tokenize_text.py | writes `.npz` aligned with JSON rows
data loader | rwkv_diffusion_rnn/data_simple.py | near-duplicate of latentDLM_mmdit/data_simple.py
trainer | rwkv_diffusion_rnn/improved_trainer.py | header indicates copied from latentDLM_mmdit
checklist | rwkv_diffusion_rnn/eval_checklist.md | manual regression checklist

## CONVENTIONS
- Uses a 32-d latent space by default (`model.latent_dim: 32`).
- Config includes sequential schedule (unconditional -> l2t -> t2l).
- Config sets `hydra.job.chdir: false` to keep relative paths stable.

## ANTI-PATTERNS
- Do not enable DDP without CUDA; script raises when torchrun env is present.
- Avoid diverging copied trainer/loader code without documenting differences.

## COMMANDS
- Smoke run (see README): `python rwkv_diffusion_rnn/train_rwkv.py --config-name rwkv_mmdit data.max_samples=10 training.num_epochs=1 tokenizer.local_files_only=false`

## NOTES
- Resume: set `training.resume=/path/to/checkpoint_dir` (created under `logging.save_dir/epoch_XXX`).
