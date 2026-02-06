# Prompt 02 - Make T2L Text-Conditioned

Goal
Ensure the latent prediction depends on text tokens, not only prefix states, so T2L is truly text-conditioned.

Context
- Current `latent_pred` is computed from prefix hidden states only.
- With a causal RWKV, prefix states cannot attend to later tokens, so T2L becomes text-independent.

Files
- rwkv_diffusion_rnn/models/rwkv_denoiser.py
- rwkv_diffusion_rnn/eval_checklist.md

Tasks
1. Add config option `model.latent_pred_from` with allowed values: text, prefix, both.
2. Implement text pooling for latent prediction.
   - Use hidden states from text tokens only (exclude prefix tokens).
   - Start with mean pooling across the text sequence.
3. Implement combined pooling if `latent_pred_from=both`.
   - Compute prefix pooled vector and text pooled vector.
   - Fuse them with a small linear + sigmoid gate.
4. Use the chosen pooled representation as input to `latent_head`.
5. Add a short checklist item that confirms latent prediction changes when text tokens change (even if prefix is the same).

Acceptance criteria
- `latent_pred` changes when text input changes in T2L mode, even if latents are identical.
- All three modes (text, prefix, both) run without shape errors.
- No changes to training loop yet.
