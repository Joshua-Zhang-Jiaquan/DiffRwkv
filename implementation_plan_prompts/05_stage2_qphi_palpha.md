# Prompt 05 - Stage II q_phi And p_alpha

Goal
Implement the variational refinement components for text<->latent: q_phi(z|y) and p_alpha(z).

Context
- Stage II introduces a learned latent inference model and a prior.
- This stage should be optional via config flags.

Files
- rwkv_diffusion_rnn/models/latent_modules.py (new)
- rwkv_diffusion_rnn/improved_trainer.py
- rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml

Tasks
1. Create `latent_modules.py` with:
   - TextLatentEncoder: input text hidden states, output mean and logvar for z.
   - LatentPrior: returns mean and logvar for z given optional conditioning.
2. Add config fields to select encoder and prior types.
3. In trainer, add a Stage II mode that:
   - Computes q_phi(z|y) and samples z.
   - Computes KL(q_phi || p_alpha).
   - Uses z for conditional text diffusion loss.
4. Add loss weights for KL and optional geometry regularizer (no-op if disabled).

Acceptance criteria
- Stage II can run without external latent files if `data.use_external_latents=false`.
- KL is computed and logged.
- Stage II can be enabled or disabled without breaking Stage I.
