# Prompt 01 - Scope And Config Map

Goal
Create a clear, explicit configuration map for the text-latent RWKV pipeline, and decide which features are in scope for Stage I and Stage II.

Context
- We are implementing a text<->latent pipeline only.
- The denoiser is RWKV, not MMDiT.
- Partial fine-tuning of RWKV is acceptable.

Files
- rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml
- rwkv_diffusion_rnn/README.md

Tasks
1. Read current config and list all existing fields that affect model, training, and loss behavior.
2. Propose new config fields required for this implementation plan. Include names, default values, and brief descriptions.
3. Decide which of these should be grouped under `model`, `training`, `loss`, or `data`.
4. Update the README to document the new fields and where they apply.

Required new config fields
- model.latent_pred_from: text | prefix | both
- model.latent_encoder_type: none | mlp | rwkv_pool
- model.latent_prior_type: fixed_gaussian | learned_gaussian
- data.use_external_latents: true | false
- training.stage_schedule: list of stages with steps or epochs
- training.unfreeze_after_steps: integer or null
- training.unfreeze_patterns: list of parameter name substrings or regex
- loss.kl_weight: float
- loss.geom_weight: float

Acceptance criteria
- The config file contains all new fields with sane defaults that keep current behavior unchanged.
- README includes a short explanation of each new field.
- You do not alter any training logic in this step.
