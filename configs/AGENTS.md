# rwkv_diffusion_rnn/configs

## OVERVIEW
Hydra configs for RWKV diffusion runs.

## STRUCTURE
- rwkv_mmdit.yaml: single-node defaults; points at preprocessed token/latent dirs

## WHERE TO LOOK
Task | Location | Notes
main config | rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml | `token_dir` + `latent_dir` point at preprocessed_data/

## CONVENTIONS
- `tokenizer.local_files_only` defaults true; set false for first-time downloads.
- RWKV loading uses `model.rwkv_trust_remote_code` (default false; enable only if required).

## ANTI-PATTERNS
- Ensure latents are projected to the expected dimension (32-d by default).

## NOTES
- First-time runs may require `tokenizer.local_files_only=false`.
