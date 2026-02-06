# Prompt 03 - Partial Fine-Tuning Policy

Goal
Implement partial fine-tuning controls without overriding `freeze_backbone` each step.

Context
- Current trainer resets all params to trainable, which defeats freeze_backbone.
- We need a configurable unfreeze policy for RWKV layers.

Files
- rwkv_diffusion_rnn/improved_trainer.py
- rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml

Tasks
1. Add config fields to control unfreezing.
   - training.unfreeze_after_steps
   - training.unfreeze_patterns (list of substrings or regex)
2. Implement a helper in the trainer to apply the policy once per step.
   - Do not set all params trainable by default.
   - If `freeze_backbone` is true, keep RWKV backbone frozen unless policy says otherwise.
3. Ensure L2T and T2L mode-specific freezing still works after the policy.
4. Log how many params are trainable after the policy.

Acceptance criteria
- With default config, behavior remains unchanged from current training.
- When `unfreeze_after_steps` is set, additional parameters become trainable at the correct step.
- L2T still freezes latent-specific params. T2L still freezes text-specific params.
