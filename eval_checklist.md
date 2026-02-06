# RWKV Denoiser Eval Checklist

## Stage I: Coupling (Diffusion-based)

### Smoke test (forward + loss)
- Load a tiny batch and confirm shapes:
  - `text_logits`: [B, T, V]
  - `latent_pred`: [B, D]
- Ensure loss is finite for:
  - `loss_type=stage1_coupling` (unconditional)
  - `loss_type=l2t`
  - `loss_type=t2l`

Suggested command:
```bash
python rwkv_diffusion_rnn/train_rwkv.py \
  --config-name rwkv_mmdit \
  training.loss_type=stage1_coupling \
  training.num_epochs=1 \
  training.train_batch_size=1 \
  logging.log_freq=1 \
  logging.eval_freq=5
```

### Latent retention
- Track latent reconstruction MSE on a small held-out batch.
- Monitor latent norm and latent_pred norm for collapse (should be finite, non-zero).

### Text diffusion sanity
- Monitor masked diffusion loss for the first few hundred steps.
- Optionally compute validation loss by reusing the trainer on a short validation split.

### Conditional generation verification
- L2T: Given a latent, generate text and verify coherence
- T2L: Given text, generate latent and verify it's non-degenerate
- Check that `latent_pred_from` modes (text/prefix/both) produce different results

### Parameter freezing verification
- Verify parameter counts match expected for each mode:
  - `stage1_coupling`: all params trainable
  - `l2t`: latent params frozen, text params trainable
  - `t2l`: text params frozen, latent params trainable

## Stage II: Variational Inference

### KL divergence checks
- Monitor KL loss between q(z|x) and p(z)
- Ensure KL doesn't collapse to zero (posterior collapse)
- Verify KL weight scheduling if used
- Check that KL loss is in reasonable range (0.1-10.0 typical)

### Encoder/Prior verification
- TextLatentEncoder produces valid mu/logvar:
  - mu should be finite, roughly centered around 0
  - logvar should be finite, typically in range [-10, 10]
- LatentPrior parameters:
  - Fixed Gaussian: mu=0, logvar=0
  - Learned Gaussian: parameters should update during training

### Reparameterization check
- Sample from q(z|x) multiple times with same input
- Verify samples are different (stochastic)
- Verify samples follow approximate Gaussian distribution

### L2T with variational latents
- Encode text → sample latent → decode to text
- Verify reconstruction quality improves over training
- Check that different samples produce diverse outputs

### T2L with variational latents
- Given text, sample multiple latents from q(z|x)
- Verify latents are diverse but semantically similar
- Check latent space geometry (similar texts → similar latents)

### Stage transition verification
- Train Stage I checkpoint → load for Stage II
- Verify frozen parameters remain unchanged
- Check that `use_external_latents=false` works for Stage II
- Monitor that Stage II doesn't catastrophically forget Stage I

### Partial fine-tuning policy
- If using `unfreeze_after_steps`:
  - Verify parameters are frozen before threshold
  - Verify parameters unfreeze after threshold
  - Check that `unfreeze_patterns` correctly match parameter names
- Monitor active parameter ratio in logs

### Geometric regularization (if enabled)
- Monitor `geom_weight` impact on latent space structure
- Verify latent distances reflect semantic similarity
- Check that geometric loss doesn't dominate total loss

## (Optional) Bilingual alignment
- If EN/ZH paired data available: compute cosine similarity between paired latents.
- For Stage II: verify KL divergence is similar across languages
