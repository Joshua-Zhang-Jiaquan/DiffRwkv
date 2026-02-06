# Prompt 06 - Data Pipeline And Mode Switching

Goal
Support both external latents and learned latents without breaking existing data loading.

Context
- Current loader assumes latents are always available.
- Stage II should allow latents to be produced by q_phi instead.

Files
- rwkv_diffusion_rnn/data_simple.py
- rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml
- rwkv_diffusion_rnn/README.md

Tasks
1. Add `data.use_external_latents` and branch loader behavior.
   - If false, only load tokens and attention masks.
   - If true, keep current latent loading behavior.
2. Update collate and dataloader output to handle `latent` being optional.
3. Update README with data requirements for both modes.

Acceptance criteria
- Dataloader works when latents are missing and `use_external_latents=false`.
- Existing training runs unchanged when `use_external_latents=true`.
