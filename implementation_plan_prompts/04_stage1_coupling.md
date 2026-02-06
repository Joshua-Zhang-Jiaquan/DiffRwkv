# Prompt 04 - Stage I Coupling Objective

Goal
Make Stage I an explicit, selectable training stage for joint text + latent diffusion.

Context
- Unconditional mode currently behaves like Stage I.
- We want explicit scheduling and clear logging for Stage I.

Files
- rwkv_diffusion_rnn/improved_trainer.py
- rwkv_diffusion_rnn/configs/rwkv_mmdit.yaml
- rwkv_diffusion_rnn/README.md

Tasks
1. Add a named stage `stage1_coupling` to the training schedule.
2. Update `get_training_mode` to map `stage1_coupling` to unconditional behavior.
3. Update logs and metrics to include the current stage name.
4. Add README documentation for Stage I.

Acceptance criteria
- Stage I can be scheduled by steps or epochs.
- Stage I runs with both text and latent diffusion losses enabled.
- Metrics clearly show Stage I in logs.
