# Prompt 07 - Sampling Utilities For L2T And T2L

Goal
Provide explicit sampling utilities for text from latents (L2T) and latents from text (T2L).

Context
- Training exists, but there is no clean inference API.
- We need reproducible, minimal sampling loops.

Files
- rwkv_diffusion_rnn/sampling.py (new)
- rwkv_diffusion_rnn/utils.py
- rwkv_diffusion_rnn/README.md

Tasks
1. Implement `sample_text_from_latent`.
   - Initialize all tokens as [MASK].
   - Iteratively denoise using the model logits and masked diffusion schedule.
2. Implement `sample_latent_from_text`.
   - Initialize latent as Gaussian noise.
   - Iteratively denoise using the latent diffusion schedule and model latent_pred.
3. Add a tiny CLI entrypoint or script usage example.
4. Document expected inputs and outputs.

Acceptance criteria
- Functions run with a loaded model and tokenizer.
- Sampling code is deterministic with a fixed seed.
- README includes usage examples.
