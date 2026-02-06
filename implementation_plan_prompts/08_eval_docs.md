# Prompt 08 - Evaluation And Docs

Goal
Add clear evaluation checks and documentation for the complete text-latent pipeline.

Context
- We need minimal regression checks for Stage I and Stage II.
- Users should know how to verify L2T and T2L.

Files
- rwkv_diffusion_rnn/eval_checklist.md
- rwkv_diffusion_rnn/README.md

Tasks
1. Expand eval checklist with:
   - L2T text generation sanity check.
   - T2L latent reconstruction MSE check.
   - Stage II KL stability check.
2. Add a short section in README on evaluation and expected logs.

Acceptance criteria
- Checklist covers Stage I and Stage II.
- README has clear next steps for evaluation.
