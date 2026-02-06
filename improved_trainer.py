# File: latentDLM_mmdit/improved_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from rwkv_diffusion_rnn.continuous_diffusion import ContinuousDiffusion
from rwkv_diffusion_rnn.diffusion_process import MaskedDiffusion



class MultimodalDiffusionTrainer(nn.Module):
    def __init__(self, model, tokenizer, text_noise_schedule, 
                 dtype: torch.dtype, config):
        
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.text_noise_schedule = text_noise_schedule
        self.dtype = dtype
        self.config = config
        
        # Initialize improved continuous diffusion
        self.latent_diffusion = ContinuousDiffusion(
            num_timesteps=config.model.get("latent_timesteps", 1000),
            beta_schedule=config.model.get("latent_beta_schedule", "cosine"),
            parameterization=config.model.get("latent_parameterization", "epsilon"),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.mask_token_id = tokenizer.mask_token_id
        
        # Loss type configuration - fix config access
        self.loss_type = getattr(config.loss, "loss_type", "l2t")
        
        # For sequential training
        self.current_step = 0
        self.current_epoch = 0
        self.sequential_schedule = getattr(config.training, "sequential_schedule", [])
        
        # Store parameter groups for selective training
        self._setup_parameter_groups()
    
    def count_params_by_group(self, mode="unconditional"):
        """Count parameters by group and training status for a specific mode."""
        if mode == "unconditional":
            # All parameters trainable
            text_trainable = sum(p.numel() for p in self.text_params)
            latent_trainable = sum(p.numel() for p in self.latent_params)
            shared_trainable = sum(p.numel() for p in self.shared_params)
            total_trainable = text_trainable + latent_trainable + shared_trainable

        elif mode == "l2t":
            # Only text + shared trainable
            text_trainable = sum(p.numel() for p in self.text_params)
            latent_trainable = 0  # Latent params frozen
            shared_trainable = sum(p.numel() for p in self.shared_params)
            total_trainable = text_trainable + shared_trainable

        elif mode == "t2l":
            # Only latent + shared trainable
            text_trainable = 0  # Text params frozen
            latent_trainable = sum(p.numel() for p in self.latent_params)
            shared_trainable = sum(p.numel() for p in self.shared_params)
            total_trainable = latent_trainable + shared_trainable

        else:
            raise ValueError(f"Unknown mode: {mode}")

        total_params = sum(p.numel() for p in self.text_params) + \
                       sum(p.numel() for p in self.latent_params) + \
                       sum(p.numel() for p in self.shared_params)

        return {
            "text_trainable": text_trainable,
            "latent_trainable": latent_trainable,
            "shared_trainable": shared_trainable,
            "total_trainable": total_trainable,
            "total_params": total_params,
            "efficiency": total_trainable / total_params * 100
        }
    
    def _setup_parameter_groups(self):
        """Setup parameter groups for different training modes."""
        # Text-related parameters (text embedding, text head, etc.)
        self.text_params = set()
        # Latent-related parameters (latent projection, latent head, etc.) 
        self.latent_params = set()
        # Shared parameters (transformer blocks, etc.)
        self.shared_params = set()
        
        for name, param in self.model.named_parameters():
            if any(key in name.lower() for key in ['text_embed', 'text_head', 'vocab']):
                self.text_params.add(param)
            elif any(key in name.lower() for key in ['latent_proj', 'latent_head']):
                self.latent_params.add(param)
            else:
                self.shared_params.add(param)
                
    
    def verify_parameter_freezing(self, mode: str):
        """Verify that parameters are correctly frozen for the given mode."""
        print(f"\n{'='*60}")
        print(f"VERIFYING PARAMETER FREEZING FOR {mode.upper()}")
        print(f"{'='*60}")

        # Count trainable params in each group
        text_trainable = sum(1 for p in self.text_params if p.requires_grad)
        latent_trainable = sum(1 for p in self.latent_params if p.requires_grad)
        shared_trainable = sum(1 for p in self.shared_params if p.requires_grad)

        text_total = len(self.text_params)
        latent_total = len(self.latent_params)
        shared_total = len(self.shared_params)

        print(f"Text parameters:   {text_trainable}/{text_total} tensors trainable")
        print(f"Latent parameters: {latent_trainable}/{latent_total} tensors trainable")
        print(f"Shared parameters: {shared_trainable}/{shared_total} tensors trainable")

        # Verify based on mode
        if mode == "l2t":
            assert latent_trainable == 0, f"L2T mode should have 0 trainable latent params, got {latent_trainable}"
            assert text_trainable == text_total, f"L2T mode should have all text params trainable"
            print("L2T verification passed")

        elif mode == "t2l":
            assert text_trainable == 0, f"T2L mode should have 0 trainable text params, got {text_trainable}"
            assert latent_trainable == latent_total, f"T2L mode should have all latent params trainable"
            print("T2L verification passed")

        elif mode == "unconditional":
            assert text_trainable == text_total, f"Unconditional mode should have all text params trainable"
            assert latent_trainable == latent_total, f"Unconditional mode should have all latent params trainable"
            print("Unconditional verification passed")

        print(f"{'='*60}\n")
    
    def _freeze_unneeded_params(self, mode: str):
        """Freeze parameters not needed for current training mode with logging."""

        # Reset all parameters to trainable first
        for param in self.model.parameters():
            param.requires_grad = True

        # Count before freezing (all trainable)
        before_counts = self.count_params_by_group("unconditional")

        if mode == "l2t":
            # Only need text generation - freeze latent-specific params
            for param in self.latent_params:
                param.requires_grad = False
                if param.grad is not None:
                    param.grad = None

        elif mode == "t2l":
            # Only need latent generation - freeze text-specific params  
            for param in self.text_params:
                param.requires_grad = False
                if param.grad is not None:
                    param.grad = None

        # Count after freezing
        after_counts = self.count_params_by_group(mode)

        # Log parameter efficiency (every 100 steps to avoid spam)
        if self.current_step % 100 == 0:
            print(f"\n{'='*60}")
            print(f"PARAMETER FREEZING: {mode.upper()} MODE")
            print(f"{'='*60}")
            print(f"Before freezing: {before_counts['total_trainable']:,} / {before_counts['total_params']:,} params trainable")
            print(f"After freezing:  {after_counts['total_trainable']:,} / {after_counts['total_params']:,} params trainable")
            print(f"Parameter efficiency: {after_counts['efficiency']:.1f}%")
            print(f"{'='*60}")

            # Detailed breakdown
            if mode == "l2t":
                print(f"Text params:   {after_counts['text_trainable']:,} trainable")
                print(f"Latent params: {after_counts['latent_trainable']:,} trainable (frozen)")
                print(f"Shared params: {after_counts['shared_trainable']:,} trainable")
            elif mode == "t2l":
                print(f"Text params:   {after_counts['text_trainable']:,} trainable (frozen)")
                print(f"Latent params: {after_counts['latent_trainable']:,} trainable")
                print(f"Shared params: {after_counts['shared_trainable']:,} trainable")
            else:
                print(f"Text params:   {after_counts['text_trainable']:,} trainable")
                print(f"Latent params: {after_counts['latent_trainable']:,} trainable")
                print(f"Shared params: {after_counts['shared_trainable']:,} trainable")
            print(f"{'='*60}\n")
    
    def get_training_mode(self, batch_idx=None):
        """Determine which training mode to use based on config."""
        
        if self.loss_type == "sequential" and self.sequential_schedule:
            # Check if schedule uses steps or epochs
            if "steps" in self.sequential_schedule[0]:
                # Step-based scheduling
                if batch_idx is not None:
                    total_steps = sum(s["steps"] for s in self.sequential_schedule)
                    current_step = batch_idx % total_steps
                    
                    cumulative = 0
                    for schedule in self.sequential_schedule:
                        cumulative += schedule["steps"]
                        if current_step < cumulative:
                            return schedule["type"]
            else:
                # Epoch-based scheduling
                total_epochs = sum(s.get("epochs", 1) for s in self.sequential_schedule)
                current_epoch = self.current_epoch % total_epochs
                
                cumulative = 0
                for schedule in self.sequential_schedule:
                    cumulative += schedule.get("epochs", 1)
                    if current_epoch < cumulative:
                        return schedule["type"]
            
            # Fallback
            return "unconditional"
        elif self.loss_type == "random":
            # Random mode selection
            import numpy as np
            modes = ["unconditional", "l2t", "t2l"]
            weights = getattr(self.config.training, "mode_weights", [1.0, 1.0, 1.0])
            return np.random.choice(modes, p=np.array(weights)/sum(weights))
        else:
            # Fixed mode: "unconditional", "l2t", or "t2l"
            return self.loss_type
    
    def forward(self, batch, batch_idx=None, force_transitting: bool = False):
        # Get current training mode and freeze unneeded parameters
        mode = self.get_training_mode(batch_idx)
        self._freeze_unneeded_params(mode)
        
        # Extract data
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        latents = batch.get("latent", None)

        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize variables
        text_t = None
        latent_t = None
        noisy_input_ids = None
        noisy_latents = None
        text_target = None
        text_mask = None
        latent_target = None
        compute_text_loss = False
        compute_latent_loss = False
        
        # ===== MODE: Unconditional/Joint Generation =====
        if mode == "unconditional":
            # Both modalities get noise - train both diffusion processes
            text_t = torch.rand(batch_size, device=device)
            latent_t = self.latent_diffusion.sample_continuous_timesteps(batch_size, device=device) if latents is not None else None
            
            noisy_input_ids = self.text_noise_schedule.sample_zt(input_ids, text_t)
            text_target = input_ids
            text_mask = (noisy_input_ids == self.mask_token_id)
            
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noise = torch.randn_like(latents)
                noisy_latents, latent_target = self.latent_diffusion.add_noise(
                    latents, latent_t, noise
                )
            
            compute_text_loss = True
            compute_latent_loss = True
            
        # ===== MODE: Latent -> Text =====
        elif mode == "l2t":
            # Text: fully masked (generate from latents)
            text_t = torch.ones(batch_size, device=device)  # Full noise
            noisy_input_ids = torch.full_like(input_ids, self.mask_token_id)
            text_target = input_ids
            text_mask = torch.ones_like(text_target, dtype=torch.bool)
            
            # Latent: clean (conditioning) - NO DIFFUSION PROCESS
            latent_t = None  # Don't even sample timesteps
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noisy_latents = latents  # Just use clean latents directly
                latent_target = None
            
            compute_text_loss = True
            compute_latent_loss = False
            
        # ===== MODE: Text -> Latent =====
        elif mode == "t2l":
            # Text: clean (conditioning) - NO MASKING
            text_t = None  # Don't sample text timesteps
            noisy_input_ids = input_ids  # Clean text
            text_target = None
            text_mask = None
            
            # Latent: fully noisy (generate from text)
            latent_t = torch.ones(batch_size, device=device)  # Full noise
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noise = torch.randn_like(latents)
                # ONLY generate noisy latents here
                noisy_latents, latent_target = self.latent_diffusion.add_noise(
                    latents, latent_t, noise
                )
            
            compute_text_loss = False
            compute_latent_loss = True
        
        # ===== FORWARD PASS =====
        text_sigma = text_t.to(dtype=self.dtype) if text_t is not None else None
        if latent_t is not None:
            latent_t = latent_t.to(dtype=self.dtype)
        
        # Handle attention mask
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # Forward pass - only compute what's needed
        if compute_text_loss or compute_latent_loss:
            text_logits, latent_pred = self.model(
                text_tokens=noisy_input_ids,
                latents=noisy_latents.unsqueeze(1) if noisy_latents is not None else None,
                text_timesteps=text_sigma,
                latent_timesteps=latent_t,
                attention_mask=attention_mask,
            )
        else:
            # This shouldn't happen, but just in case
            raise ValueError(f"Invalid training mode: {mode}")
        
        # ===== LOSS CALCULATION =====
        total_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        
        # Text loss - only compute if needed
        if compute_text_loss and text_target is not None:
            vocab_size = text_logits.shape[-1]
            if text_mask is not None and text_mask.any():
                text_loss_unmasked = F.cross_entropy(
                    text_logits.view(-1, vocab_size),
                    text_target.view(-1),
                    ignore_index=-100,
                    reduction='none'
                )
                text_loss = (text_loss_unmasked * text_mask.view(-1)).sum() / text_mask.sum().clamp(min=1)
                total_loss = total_loss + text_loss
            else:
                text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        else:
            text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        
        # Latent loss - only compute if needed
        if compute_latent_loss and latent_pred is not None and latent_target is not None:
            # Ensure shapes match
            if latent_pred.dim() == 3 and latent_target.dim() == 2:
                latent_target = latent_target.unsqueeze(1)
            if latent_pred.dim() == 2 and latent_target.dim() == 3 and latent_target.shape[1] == 1:
                latent_target = latent_target.squeeze(1)
            
            # Handle shape mismatches
            if latent_pred.shape != latent_target.shape:
                if latent_pred.dim() == 3 and latent_pred.shape[1] > 1:
                    latent_pred = latent_pred.mean(dim=1)
                if latent_target.dim() == 3 and latent_target.shape[1] > 1:
                    latent_target = latent_target.mean(dim=1)
            
            latent_loss = F.mse_loss(latent_pred, latent_target)
            total_loss = total_loss + latent_loss
        else:
            latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        
        # ===== METRICS =====
        with torch.no_grad():
            text_accuracy = 0.0
            if compute_text_loss and text_logits is not None and text_target is not None:
                pred_tokens = torch.argmax(text_logits, dim=-1)
                if text_mask is not None and text_mask.any():
                    text_accuracy = (pred_tokens[text_mask] == text_target[text_mask]).float().mean().item()
            
            metrics = {
                "loss": float(total_loss.item()),
                "text_loss": float(text_loss.item()),
                "latent_loss": float(latent_loss.item()),
                "text_accuracy": float(text_accuracy),
                "mode": mode,  # Track which mode was used
            }
            
            if latents is not None:
                metrics["latent_norm"] = float(latents.norm(dim=-1).mean().item())
                if compute_latent_loss and latent_pred is not None:
                    metrics["latent_pred_norm"] = float(latent_pred.norm(dim=-1).mean().item())
        
        return total_loss, metrics
    


class MultimodalDiffusionTrainer_new(nn.Module):
    """
    Improved trainer for multimodal MMDiT with selective parameter training.
    
    Supports three training modes:
    - unconditional: Train both text and latent diffusion (both modalities noisy)
    - l2t (latent-to-text): Clean latents -> Generate text (only train text diffusion)
    - t2l (text-to-latent): Clean text -> Generate latents (only train latent diffusion)
    """
    
    def __init__(self, model, tokenizer, text_noise_schedule, 
                 dtype: torch.dtype, config):
        
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.text_noise_schedule = text_noise_schedule
        self.dtype = dtype
        self.config = config
        
        # Initialize continuous diffusion from model config.
        model_cfg = getattr(config, "model", None)
        self.latent_diffusion = ContinuousDiffusion(
            num_timesteps=getattr(model_cfg, "latent_timesteps", 1000),
            beta_schedule=getattr(model_cfg, "latent_beta_schedule", "cosine"),
            parameterization=getattr(model_cfg, "latent_parameterization", "epsilon"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        self.mask_token_id = tokenizer.mask_token_id
        
        # Training mode configuration
        self.loss_type = getattr(config.loss, "loss_type", "multimodal_mdlm")
        
        # For sequential/curriculum training
        self.current_step = 0
        self.current_epoch = 0
        
        # Setup parameter groups for selective training
        self._setup_parameter_groups()
        
        print("MultimodalDiffusionTrainer initialized")
        print(f"  Loss type: {self.loss_type}")
        print(f"  Text mask token ID: {self.mask_token_id}")
        print(f"  Model dtype: {self.dtype}")
        
    def count_params_by_group(self, mode="unconditional"):
        """Count parameters by group and training status for a specific mode."""
        if mode == "unconditional":
            # All parameters trainable
            text_trainable = sum(p.numel() for p in self.text_params)
            latent_trainable = sum(p.numel() for p in self.latent_params)
            shared_trainable = sum(p.numel() for p in self.shared_params)
            total_trainable = text_trainable + latent_trainable + shared_trainable

        elif mode == "l2t":
            # Only text + shared trainable
            text_trainable = sum(p.numel() for p in self.text_params)
            latent_trainable = 0  # Latent params frozen
            shared_trainable = sum(p.numel() for p in self.shared_params)
            total_trainable = text_trainable + shared_trainable

        elif mode == "t2l":
            # Only latent + shared trainable
            text_trainable = 0  # Text params frozen
            latent_trainable = sum(p.numel() for p in self.latent_params)
            shared_trainable = sum(p.numel() for p in self.shared_params)
            total_trainable = latent_trainable + shared_trainable

        else:
            raise ValueError(f"Unknown mode: {mode}")

        total_params = sum(p.numel() for p in self.text_params) + \
                       sum(p.numel() for p in self.latent_params) + \
                       sum(p.numel() for p in self.shared_params)

        return {
            "text_trainable": text_trainable,
            "latent_trainable": latent_trainable,
            "shared_trainable": shared_trainable,
            "total_trainable": total_trainable,
            "total_params": total_params,
            "efficiency": total_trainable / total_params * 100
        }
    
    def _setup_parameter_groups(self):
        """Setup parameter groups for different training modes."""
        self.text_params = set()
        self.latent_params = set()
        self.shared_params = set()
        
        # More comprehensive parameter detection
        for name, param in self.model.named_parameters():
            name_lower = name.lower()
            
            # Text-specific parameters
            if any(key in name_lower for key in [
                'text_embed', 'text_head', 'vocab', 'token_embed', 
                'text_proj', 'text_norm', 'text_out', 'text_encoder'
            ]):
                self.text_params.add(param)
                
            # Latent-specific parameters  
            elif any(key in name_lower for key in [
                'latent_proj', 'latent_head', 'latent_embed',
                'latent_norm', 'latent_out', 'latent_linear', 'latent_encoder'
            ]):
                self.latent_params.add(param)
                
            # Shared transformer parameters (backbone)
            else:
                self.shared_params.add(param)
        
        print(f"Parameter groups:")
        print(f"  Text-specific: {len(self.text_params)} parameters")
        print(f"  Latent-specific: {len(self.latent_params)} parameters")  
        print(f"  Shared: {len(self.shared_params)} parameters")
    
    def get_active_param_count(self):
        """Get count of parameters that are currently trainable."""
        active_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        return active_params, total_params
    
    def _apply_unfreeze_policy(self):
        unfreeze_after = getattr(self.config.training, "unfreeze_after_steps", None)
        unfreeze_patterns = getattr(self.config.training, "unfreeze_patterns", [])
        
        if unfreeze_after is None or self.current_step < unfreeze_after:
            return
        
        if not unfreeze_patterns:
            return
        
        import re
        for name, param in self.model.named_parameters():
            for pattern in unfreeze_patterns:
                try:
                    if re.search(pattern, name):
                        param.requires_grad = True
                        break
                except re.error:
                    if pattern in name:
                        param.requires_grad = True
                        break
    
    def verify_parameter_freezing(self, mode: str):
        """Verify that parameters are correctly frozen for the given mode."""
        print(f"\n{'='*60}")
        print(f"VERIFYING PARAMETER FREEZING FOR {mode.upper()}")
        print(f"{'='*60}")

        # Count trainable params in each group
        text_trainable = sum(1 for p in self.text_params if p.requires_grad)
        latent_trainable = sum(1 for p in self.latent_params if p.requires_grad)
        shared_trainable = sum(1 for p in self.shared_params if p.requires_grad)

        text_total = len(self.text_params)
        latent_total = len(self.latent_params)
        shared_total = len(self.shared_params)

        print(f"Text parameters:   {text_trainable}/{text_total} tensors trainable")
        print(f"Latent parameters: {latent_trainable}/{latent_total} tensors trainable")
        print(f"Shared parameters: {shared_trainable}/{shared_total} tensors trainable")

        # Verify based on mode
        if mode == "l2t":
            assert latent_trainable == 0, f"L2T mode should have 0 trainable latent params, got {latent_trainable}"
            assert text_trainable == text_total, f"L2T mode should have all text params trainable"
            print("L2T verification passed")

        elif mode == "t2l":
            assert text_trainable == 0, f"T2L mode should have 0 trainable text params, got {text_trainable}"
            assert latent_trainable == latent_total, f"T2L mode should have all latent params trainable"
            print("T2L verification passed")

        elif mode == "unconditional":
            assert text_trainable == text_total, f"Unconditional mode should have all text params trainable"
            assert latent_trainable == latent_total, f"Unconditional mode should have all latent params trainable"
            print("Unconditional verification passed")

        print(f"{'='*60}\n")
    
    def _freeze_unneeded_params(self, mode: str):
        """Freeze parameters not needed for current training mode with logging."""

        # Reset all parameters to trainable first
        for param in self.model.parameters():
            param.requires_grad = True
        
        self._apply_unfreeze_policy()

        # Count before freezing (all trainable)
        before_counts = self.count_params_by_group("unconditional")

        if mode == "l2t":
            # Only need text generation - freeze latent-specific params
            for param in self.latent_params:
                param.requires_grad = False
                if param.grad is not None:
                    param.grad = None

        elif mode == "t2l":
            # Only need latent generation - freeze text-specific params  
            for param in self.text_params:
                param.requires_grad = False
                if param.grad is not None:
                    param.grad = None

        # Count after freezing
        after_counts = self.count_params_by_group(mode)

        # Log parameter efficiency (every 100 steps to avoid spam)
        if self.current_step % 100 == 0:
            print(f"\n{'='*60}")
            print(f"PARAMETER FREEZING: {mode.upper()} MODE")
            print(f"{'='*60}")
            print(f"Before freezing: {before_counts['total_trainable']:,} / {before_counts['total_params']:,} params trainable")
            print(f"After freezing:  {after_counts['total_trainable']:,} / {after_counts['total_params']:,} params trainable")
            print(f"Parameter efficiency: {after_counts['efficiency']:.1f}%")
            print(f"{'='*60}")

            # Detailed breakdown
            if mode == "l2t":
                print(f"Text params:   {after_counts['text_trainable']:,} trainable")
                print(f"Latent params: {after_counts['latent_trainable']:,} trainable (frozen)")
                print(f"Shared params: {after_counts['shared_trainable']:,} trainable")
            elif mode == "t2l":
                print(f"Text params:   {after_counts['text_trainable']:,} trainable (frozen)")
                print(f"Latent params: {after_counts['latent_trainable']:,} trainable")
                print(f"Shared params: {after_counts['shared_trainable']:,} trainable")
            else:
                print(f"Text params:   {after_counts['text_trainable']:,} trainable")
                print(f"Latent params: {after_counts['latent_trainable']:,} trainable")
                print(f"Shared params: {after_counts['shared_trainable']:,} trainable")
            print(f"{'='*60}\n")
    
    def get_training_mode(self, step=None):
        """Determine which training mode to use based on config."""
        
        # Handle different loss_type configurations
        if self.loss_type in ["unconditional", "l2t", "t2l"]:
            return self.loss_type
            
        elif self.loss_type == "sequential":
            # Sequential training by step or epoch
            schedule = getattr(self.config.training, "sequential_schedule", [])
            if not schedule:
                return "unconditional"
            
            if step is not None:
                # Step-based scheduling
                total_steps = sum(s.get("steps", 1000) for s in schedule)
                current_step = step % total_steps
                
                cumulative = 0
                for schedule_item in schedule:
                    cumulative += schedule_item.get("steps", 1000)
                    if current_step < cumulative:
                        return schedule_item["type"]
            else:
                # Epoch-based scheduling (fallback)
                total_epochs = sum(s.get("epochs", 1) for s in schedule)
                current_epoch = self.current_epoch % total_epochs
                
                cumulative = 0
                for schedule_item in schedule:
                    cumulative += schedule_item.get("epochs", 1)
                    if current_epoch < cumulative:
                        return schedule_item["type"]
        
        elif self.loss_type == "random":
            # Random mode selection for each batch
            import numpy as np
            modes = ["unconditional", "l2t", "t2l"]
            weights = getattr(self.config.training, "mode_weights", [1.0, 1.0, 1.0])
            mode = np.random.choice(modes, p=np.array(weights)/sum(weights))
            return mode
        
        mode = "unconditional"
        
        if mode == "stage1_coupling":
            mode = "unconditional"
        
        return mode
    
    def forward(self, batch, step=None):
        if step is not None:
            self.current_step = int(step)
        
        mode = self.get_training_mode(step=step)
        original_mode = mode
        
        if mode == "stage1_coupling":
            mode = "unconditional"
        
        self._freeze_unneeded_params(mode)
        
        # Extract data
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        latents = batch.get("latent", None)

        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize variables
        text_t = None
        latent_t = None
        noisy_input_ids = None
        noisy_latents = None
        text_target = None
        text_mask = None
        latent_target = None
        compute_text_loss = False
        compute_latent_loss = False
        
        # ===== MODE: Unconditional/Joint Generation =====
        if mode == "unconditional":
            # Both modalities get noise - train both diffusion processes
            text_t = torch.rand(batch_size, device=device)
            latent_t = self.latent_diffusion.sample_continuous_timesteps(batch_size, device=device)
            
            noisy_input_ids = self.text_noise_schedule.sample_zt(input_ids, text_t)
            text_target = input_ids
            text_mask = (noisy_input_ids == self.mask_token_id)
            
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noise = torch.randn_like(latents)
                noisy_latents, latent_target = self.latent_diffusion.add_noise(
                    latents, latent_t, noise
                )
            
            compute_text_loss = True
            compute_latent_loss = True
            
        # ===== MODE: Latent -> Text =====
        elif mode == "l2t":
            # Text: fully masked (generate from latents)
            text_t = torch.ones(batch_size, device=device)  # Full noise
            noisy_input_ids = torch.full_like(input_ids, self.mask_token_id)
            text_target = input_ids
            text_mask = torch.ones_like(text_target, dtype=torch.bool)

            # Latent: clean (conditioning) - Use zeros for timestep
            latent_t = torch.zeros(batch_size, device=device)  # Use zeros, not None
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noisy_latents = latents  # Clean latents
                latent_target = None

            compute_text_loss = True
            compute_latent_loss = False

        # ===== MODE: Text -> Latent =====
        elif mode == "t2l":
            # Text: clean (conditioning) - Use zeros for timestep
            text_t = torch.zeros(batch_size, device=device)  # Use zeros, not None
            noisy_input_ids = input_ids  # Clean text
            text_target = None
            text_mask = None

            # Latent: fully noisy (generate from text)
            latent_t = torch.ones(batch_size, device=device)  # Full noise
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noise = torch.randn_like(latents)
                noisy_latents, latent_target = self.latent_diffusion.add_noise(
                    latents, latent_t, noise
                )

            compute_text_loss = False
            compute_latent_loss = True
        
        # ===== FORWARD PASS =====
        text_sigma = text_t.to(dtype=self.dtype) if text_t is not None else None
        if latent_t is not None:
            latent_t = latent_t.to(dtype=self.dtype)
        
        # Handle attention mask
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # Forward pass
        text_logits, latent_pred = self.model(
            text_tokens=noisy_input_ids,
            latents=noisy_latents.unsqueeze(1) if noisy_latents is not None else None,
            text_timesteps=text_sigma,
            latent_timesteps=latent_t,
            attention_mask=attention_mask,
        )
        
        # ===== LOSS CALCULATION =====
        total_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        
        # Text loss - only compute if needed
        if compute_text_loss and text_target is not None:
            vocab_size = text_logits.shape[-1]
            if text_mask is not None and text_mask.any():
                text_loss_unmasked = F.cross_entropy(
                    text_logits.view(-1, vocab_size),
                    text_target.view(-1),
                    ignore_index=-100,
                    reduction='none'
                )
                text_loss = (text_loss_unmasked * text_mask.view(-1)).sum() / text_mask.sum().clamp(min=1)
                total_loss = total_loss + text_loss
            else:
                text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        else:
            text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        
        # Latent loss - only compute if needed
        if compute_latent_loss and latent_pred is not None and latent_target is not None:
            # Ensure shapes match
            if latent_pred.dim() == 3 and latent_target.dim() == 2:
                latent_target = latent_target.unsqueeze(1)
            if latent_pred.dim() == 2 and latent_target.dim() == 3 and latent_target.shape[1] == 1:
                latent_target = latent_target.squeeze(1)
            
            # Handle shape mismatches
            if latent_pred.shape != latent_target.shape:
                if latent_pred.dim() == 3 and latent_pred.shape[1] > 1:
                    latent_pred = latent_pred.mean(dim=1)
                if latent_target.dim() == 3 and latent_target.shape[1] > 1:
                    latent_target = latent_target.mean(dim=1)
            
            latent_loss = F.mse_loss(latent_pred, latent_target)
            total_loss = total_loss + latent_loss
        else:
            latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        
        # ===== METRICS =====
        with torch.no_grad():
            text_accuracy = 0.0
            if compute_text_loss and text_logits is not None and text_target is not None:
                pred_tokens = torch.argmax(text_logits, dim=-1)
                if text_mask is not None and text_mask.any():
                    text_accuracy = (pred_tokens[text_mask] == text_target[text_mask]).float().mean().item()
            
            metrics = {
                "loss": float(total_loss.item()),
                "text_loss": float(text_loss.item()),
                "latent_loss": float(latent_loss.item()),
                "text_accuracy": float(text_accuracy),
                "training_mode": original_mode,
                "current_stage": original_mode,
            }
            
            if latents is not None:
                metrics["latent_norm"] = float(latents.norm(dim=-1).mean().item())
                if compute_latent_loss and latent_pred is not None:
                    metrics["latent_pred_norm"] = float(latent_pred.norm(dim=-1).mean().item())
            
            # Add parameter efficiency metrics occasionally
            if self.current_step % 100 == 0:
                active, total = self.get_active_param_count()
                metrics["active_params_ratio"] = float(active / total)
        
        return total_loss, metrics
