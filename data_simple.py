# # File: latentDLM_mmdit/data_simple.py (SIMPLIFIED WORKING VERSION)
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from pathlib import Path
# import random
# import os
# import time
# from typing import Optional, List, Tuple



# class DirectFileDataset(Dataset):
#     """Dataset that reads from tokenized .npz and latent .npy files."""
    
#     def __init__(self, token_dir: str, latent_dir: str, max_samples: Optional[int] = None):
#         self.token_dir = Path(token_dir)
#         self.latent_dir = Path(latent_dir)
        
#         # Get all token files
#         token_files = sorted(list(self.token_dir.glob("*.npz")))
        
#         if max_samples:
#             token_files = token_files[:max_samples]
        
#         # Create file pairs
#         self.file_pairs = []
#         for token_file in token_files:
#             latent_name = token_file.stem.replace('_tokens', '')
#             latent_file = self.latent_dir / f"{latent_name}.npy"
            
#             if latent_file.exists():
#                 self.file_pairs.append((token_file, latent_file))
        
#         print(f"Found {len(self.file_pairs)} valid file pairs")
    
#     def __len__(self):
#         return len(self.file_pairs)
    
#     def __getitem__(self, idx):
#         token_file, latent_file = self.file_pairs[idx]
        
#         # Load data
#         token_data = np.load(token_file)
#         input_ids = torch.from_numpy(token_data['input_ids'].astype(np.int32)).long()
#         attention_mask = torch.from_numpy(token_data['attention_mask'].astype(bool)).float()
        
#         # Load latent (always shape [latent_dim])
#         latent = torch.from_numpy(np.load(latent_file).astype(np.float32)).float()
        
#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'latent': latent  # Shape: [latent_dim]
#         }


# def get_direct_dataloaders(config, tokenizer=None):
#     """Get dataloaders using PyTorch's DistributedSampler."""
#     from torch.utils.data import DataLoader, random_split
    
#     # Get rank info
#     rank = 0
#     world_size = 1
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         rank = torch.distributed.get_rank()
#         world_size = torch.distributed.get_world_size()
    
#     data_config = config.data
#     seed = config.training.seed
    
#     # Training dataset (ALL data)
#     full_train_dataset = DirectFileDataset(
#         token_dir=data_config.token_dir,
#         latent_dir=data_config.latent_dir,
#         max_samples=data_config.get('max_samples', None)
#     )
    
#     # Validation dataset (ALL data or separate)
#     if hasattr(data_config, 'val_token_dir'):
#         val_dataset = DirectFileDataset(
#             token_dir=data_config.val_token_dir,
#             latent_dir=data_config.val_latent_dir,
#             max_samples=data_config.get('val_max_samples', None)
#         )
#         train_dataset = full_train_dataset
#     else:
#         # Split training data for validation
#         val_ratio = data_config.get('val_ratio', 0.05)
#         val_size = max(1, int(len(full_train_dataset) * val_ratio))
#         train_size = len(full_train_dataset) - val_size
        
#         train_dataset, val_dataset = random_split(
#             full_train_dataset,
#             [train_size, val_size],
#             generator=torch.Generator().manual_seed(seed)
#         )
    
#     print(f"Rank {rank}: Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
#     # Collate function
#     def collate_fn(batch):
#         input_ids = torch.stack([item['input_ids'] for item in batch])
#         attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
#         # Stack latents (each is [latent_dim])
#         latents = torch.stack([item['latent'] for item in batch])  # [B, latent_dim]
        
#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'latent': latents
#         }
    
#     # Create DistributedSamplers for DDP
#     if world_size > 1:
#         train_sampler = DistributedSampler(
#             train_dataset,
#             num_replicas=world_size,
#             rank=rank,
#             shuffle=True,
#             seed=seed,
#             drop_last=True
#         )
#         val_sampler = DistributedSampler(
#             val_dataset,
#             num_replicas=world_size,
#             rank=rank,
#             shuffle=False,
#             drop_last=False
#         )
#     else:
#         train_sampler = None
#         val_sampler = None
    
#     # Create DataLoaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.training.train_batch_size,
#         sampler=train_sampler,
#         shuffle=(train_sampler is None),
#         num_workers=min(4, data_config.get('num_workers', 2)),
#         collate_fn=collate_fn,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=False
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.training.eval_batch_size,
#         sampler=val_sampler,
#         shuffle=False,
#         num_workers=min(2, data_config.get('num_workers', 2)),
#         collate_fn=collate_fn,
#         pin_memory=True,
#         drop_last=False
#     )
    
#     return train_loader, val_loader


# def get_simple_dataloaders(config, tokenizer=None):
#     """Main entry point."""
#     if config.data.get('use_preprocessed', True):
#         return get_direct_dataloaders(config, tokenizer)
#     else:
#         # Legacy JSON loader
#         try:
#             from latentDLM_mmdit.data_simple_legacy import get_json_dataloaders
#             return get_json_dataloaders(config, tokenizer)
#         except ImportError:
#             raise ImportError("Legacy JSON data loader not found")

            
            
        
        
        
        
        

# File: latentDLM_mmdit/data_simple.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import json
import random
import os
from typing import Optional, Tuple, Union

class DirectFileDataset(Dataset):
    """Dataset that reads directly from tokenized .npz and latent .npy files."""
    
    def __init__(self, 
                 token_dir: str, 
                 latent_dir: str, 
                 max_samples: Optional[int] = None,
                 use_external_latents: bool = True,
                 verbose: bool = True):
        self.token_dir = Path(token_dir)
        self.latent_dir = Path(latent_dir) if latent_dir else None
        self.use_external_latents = use_external_latents
        
        self.token_files = sorted(list(self.token_dir.glob("*.npz")))
        
        if max_samples:
            self.token_files = self.token_files[:max_samples]
        
        valid_files = []
        missing_latents = 0
        
        if self.use_external_latents and self.latent_dir:
            for i, token_file in enumerate(self.token_files):
                latent_name = token_file.stem.replace('_tokens', '')
                latent_file = self.latent_dir / f"{latent_name}.npy"
                
                if latent_file.exists():
                    valid_files.append((token_file, latent_file))
                else:
                    missing_latents += 1
                    if verbose and missing_latents <= 5:
                        print(f"Warning: Missing latent for {token_file.name}")
        else:
            for token_file in self.token_files:
                valid_files.append((token_file, None))
        
        self.file_pairs = valid_files

        self._seq_len = 512
        self._latent_dim = 32
        if self.file_pairs and self.use_external_latents:
            try:
                token_f, latent_f = self.file_pairs[0]
                if latent_f:
                    token_data = np.load(token_f)
                    if "input_ids" in token_data:
                        ids = token_data["input_ids"]
                        if hasattr(ids, "shape") and len(ids.shape) >= 1:
                            self._seq_len = int(ids.shape[-1])
                    latent = np.load(latent_f)
                    if hasattr(latent, "shape") and len(latent.shape) >= 1:
                        self._latent_dim = int(latent.shape[-1])
            except Exception:
                pass
        
        if verbose:
            print(f"Found {len(self.token_files)} token files")
            print(f"Found {len(self.file_pairs)} valid file pairs")
            print(f"Missing latents: {missing_latents}")
            
            # Show some examples
            if len(self.file_pairs) > 0:
                print("Sample file mappings:")
                for i in range(min(3, len(self.file_pairs))):
                    token_f, latent_f = self.file_pairs[i]
                    print(f"  Token: {token_f.name}")
                    print(f"  Latent: {latent_f.name}")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        token_file, latent_file = self.file_pairs[idx]
        
        try:
            token_data = np.load(token_file)
            input_ids = token_data['input_ids'].astype(np.int32)
            attention_mask = token_data['attention_mask'].astype(bool)
            
            if input_ids.ndim == 0:
                input_ids = np.array([input_ids])
            if attention_mask.ndim == 0:
                attention_mask = np.array([attention_mask])
            
            input_ids_tensor = torch.from_numpy(input_ids).long()
            attention_mask_tensor = torch.from_numpy(attention_mask).float()
            
            if self.use_external_latents and latent_file is not None:
                latent = np.load(latent_file)
                latent_tensor = torch.from_numpy(latent).float()
                
                if latent_tensor.dim() == 1:
                    latent_tensor = latent_tensor.unsqueeze(0)
                elif latent_tensor.dim() == 2:
                    if latent_tensor.shape[0] != 1:
                        latent_tensor = latent_tensor[0:1]
            else:
                latent_tensor = None
            
            return {
                'input_ids': input_ids_tensor,
                'attention_mask': attention_mask_tensor,
                'latent': latent_tensor,
                'token_file': str(token_file.name),
                'latent_file': str(latent_file.name) if latent_file else 'none'
            }
            
        except Exception as e:
            print(f"Error loading files {token_file}, {latent_file}: {e}")
            return self._create_empty_sample()
    
    def _create_empty_sample(self):
        """Create an empty sample for error cases."""
        return {
            'input_ids': torch.zeros(self._seq_len, dtype=torch.long),
            'attention_mask': torch.zeros(self._seq_len, dtype=torch.float),
            'latent': torch.zeros(1, self._latent_dim, dtype=torch.float),
            'token_file': 'error',
            'latent_file': 'error'
        }


class PreTokenizedDataset(Dataset):
    """Dataset that reads pre-tokenized data from your specific directory structure."""
    
    def __init__(self, base_dir: Union[str, Path], split: str = "train", max_samples: Optional[int] = None):
        """
        Args:
            base_dir: Base directory (e.g., sonar-embeddings-1024d-interact)
            split: "train" or "val"
            max_samples: Maximum samples to load
        """
        self.base_dir = Path(base_dir)
        self.split = split
        
        # Directories
        self.token_dir = self.base_dir / "tokens" / split
        self.latent_dir = self.base_dir / "latents" / split
        self.text_dir = self.base_dir / "texts" / split
        
        print(f"Token dir: {self.token_dir}")
        print(f"Latent dir: {self.latent_dir}")
        
        # Get all token files
        token_files = sorted(list(self.token_dir.glob("*.npz")))
        
        if max_samples:
            token_files = token_files[:max_samples]
        
        # Create file pairs
        self.file_pairs = []
        for token_file in token_files:
            # Get corresponding latent file name
            latent_name = token_file.stem.replace('_tokens', '')
            latent_file = self.latent_dir / f"{latent_name}.npy"
            
            if latent_file.exists():
                self.file_pairs.append((token_file, latent_file))
        
        print(f"Loaded {len(self.file_pairs)} file pairs from {base_dir}")
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        token_file, latent_file = self.file_pairs[idx]
        
        # Load tokenized data
        token_data = np.load(token_file)
        
        # Extract arrays - handle both npz and direct numpy formats
        if 'input_ids' in token_data:
            input_ids = token_data['input_ids']
            attention_mask = token_data['attention_mask']
        else:
            # Assume it's a dictionary-like structure
            input_ids = token_data['input_ids']
            attention_mask = token_data['attention_mask']
        
        # Load latent
        latent = np.load(latent_file)
        
        # Convert to tensors
        input_ids = torch.from_numpy(input_ids.astype(np.int32)).long()
        attention_mask = torch.from_numpy(attention_mask.astype(bool)).float()
        latent = torch.from_numpy(latent).float()
        
        # Ensure latent has correct shape [1, latent_dim]
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'latent': latent
        }


def get_direct_dataloaders(config, tokenizer=None):
    """Get dataloaders that read directly from tokenized and latent files."""
    from torch.utils.data import DataLoader, random_split
    
    # Extract paths from config
    data_config = config.data
    
    # Option 1: Direct paths (for your current structure)
    if hasattr(data_config, 'token_dir') and hasattr(data_config, 'latent_dir'):
        token_dir = data_config.token_dir
        latent_dir = data_config.latent_dir
        
        print(f"Loading from token_dir: {token_dir}")
        print(f"Loading from latent_dir: {latent_dir}")
        
        # Create dataset
        full_dataset = DirectFileDataset(
            token_dir=token_dir,
            latent_dir=latent_dir,
            max_samples=data_config.get('max_samples', None),
            verbose=True
        )
    
    # Option 2: Base directory with standard structure
    elif hasattr(data_config, 'data_dir'):
        base_dir = Path(data_config.data_dir)
        split = data_config.get('split', 'train')
        
        print(f"Loading from base_dir: {base_dir}, split: {split}")
        
        # Create dataset
        full_dataset = PreTokenizedDataset(
            base_dir=base_dir,
            split=split,
            max_samples=data_config.get('max_samples', None)
        )
    
    else:
        raise ValueError("Either token_dir/latent_dir or data_dir must be specified in config")
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Split into train/val
    val_ratio = data_config.get('val_ratio', 0.05)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        use_external_latents = data_config.get('use_external_latents', True)
        
        if use_external_latents:
            latent_dim = config.model.get('latent_dim', 1024)
            
            latents = []
            for item in batch:
                latent = item['latent']
                if latent is not None:
                    if latent.dim() == 1:
                        latent = latent.unsqueeze(0)
                    elif latent.dim() == 2 and latent.shape[0] != 1:
                        latent = latent[0:1]
                    latents.append(latent)
                else:
                    latents.append(torch.zeros(1, latent_dim))
            
            latent_tensor = torch.cat(latents, dim=0)
        else:
            latent_tensor = None
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'latent': latent_tensor
        }
    
    # Create distributed samplers
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
            seed=config.training.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # DataLoader settings
    num_workers = data_config.get('num_workers', 16)
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=min(8, num_workers),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers
    )
    
    return train_loader, val_loader


# For backward compatibility
def get_simple_dataloaders(config, tokenizer=None):
    """Main entry point - automatically chooses the right data loader."""
    # Check which type of data we're using
    use_preprocessed = config.data.get('use_preprocessed', True)
    
    if use_preprocessed:
        print("Using preprocessed file-based data loader")
        return get_direct_dataloaders(config, tokenizer)
    else:
        print("Using JSON-based data loader (needs tokenizer)")
        raise ValueError(
            "JSON-based loader is not included in rwkv_diffusion_rnn. "
            "Set data.use_preprocessed=true and provide token_dir/latent_dir."
        )
























# # File: latentDLM_mmdit/data_simple.py (UPDATED VERSION)
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from pathlib import Path
# import json
# import random
# import os
# from typing import Optional, Tuple, List, Dict, Any

# class DirectFileDataset(Dataset):
#     """Dataset for large .npy files containing multiple samples."""
    
#     def __init__(self, 
#                  token_dir: str, 
#                  latent_dir: str, 
#                  max_samples: Optional[int] = None,
#                  verbose: bool = True):
#         """
#         Args:
#             token_dir: Directory containing .npy files with shape [num_samples, 2, seq_len]
#             latent_dir: Directory containing latent .npy files
#             max_samples: Maximum number of samples to load
#             verbose: Print debug information
#         """
#         self.token_dir = Path(token_dir)
#         self.latent_dir = Path(latent_dir)
        
#         # Get all token files
#         token_files = sorted(list(self.token_dir.glob("*.npy")))
        
#         if not token_files:
#             raise ValueError(f"No .npy files found in {token_dir}")
        
#         # Get all latent files
#         latent_files = sorted(list(self.latent_dir.glob("*.npy")))
#         if not latent_files:
#             raise ValueError(f"No latent files found in {latent_dir}")
        
#         # Ensure files match 1:1
#         if len(token_files) != len(latent_files):
#             print(f"Warning: {len(token_files)} token files vs {len(latent_files)} latent files")
        
#         # Create file mapping
#         self.file_pairs = []
#         for token_file, latent_file in zip(token_files, latent_files):
#             self.file_pairs.append((token_file, latent_file))
        
#         # Compute total samples
#         self.total_samples = 0
#         self.file_info = []  # per-file metadata
        
#         for token_file, latent_file in self.file_pairs:
#             # Check token file shape
#             token_data = np.load(token_file, mmap_mode='r')
            
#             if token_data.ndim == 3:
#                 num_samples = token_data.shape[0]
#             else:
#                 num_samples = 1
#                 print(f"Warning: {token_file.name} has unexpected shape {token_data.shape}")
            
#             self.file_info.append({
#                 'token_file': token_file,
#                 'latent_file': latent_file,
#                 'start_idx': self.total_samples,
#                 'num_samples': num_samples,
#                 'total_samples': self.total_samples + num_samples
#             })
            
#             self.total_samples += num_samples
        
#         # Cap max samples
#         if max_samples and max_samples < self.total_samples:
#             self.total_samples = max_samples
        
#         if verbose:
#             print("\nDataset stats:")
#             for info in self.file_info:
#                 print(f"  file: {info['token_file'].name}")
#                 print(f"    offset: {info['start_idx']:,}, samples: {info['num_samples']:,}")
#             print(f"\nTotal samples: {self.total_samples:,}")
    
#     def __len__(self):
#         return self.total_samples
    
#     def __getitem__(self, idx):
#         # Find the file containing this sample
#         file_idx = -1
#         for i, info in enumerate(self.file_info):
#             if idx < info['total_samples']:
#                 file_idx = i
#                 break
        
#         if file_idx == -1:
#             # Out of range; wrap around
#             idx = idx % self.total_samples
#             for i, info in enumerate(self.file_info):
#                 if idx < info['total_samples']:
#                     file_idx = i
#                     break
        
#         info = self.file_info[file_idx]
#         local_idx = idx - info['start_idx']
        
#         # Ensure local index in range
#         if local_idx >= info['num_samples']:
#             local_idx = local_idx % info['num_samples']
        
#         try:
#             # Load token data
#             token_array = np.load(info['token_file'], mmap_mode='r')
            
#             if token_array.ndim == 3:
#                 # 3D array: [num_samples, 2, seq_len]
#                 input_ids = token_array[local_idx, 0, :].astype(np.int32)
#                 attention_mask = token_array[local_idx, 1, :].astype(bool)
#             else:
#                 raise ValueError(f"Unexpected token array shape: {token_array.shape}")
            
#             # Load latent data
#             latent_array = np.load(info['latent_file'], mmap_mode='r')
            
#             # Assume latent file is 3D or 2D
#             if latent_array.ndim == 3:
#                 # [num_samples, 1, latent_dim] or [num_samples, latent_dim]
#                 if local_idx >= latent_array.shape[0]:
#                     latent_idx = local_idx % latent_array.shape[0]
#                 else:
#                     latent_idx = local_idx
                
#                 if latent_array.shape[1] == 1:
#                     latent = latent_array[latent_idx, 0, :]
#                 else:
#                     latent = latent_array[latent_idx, :]
#             elif latent_array.ndim == 2:
#                 # [num_samples, latent_dim]
#                 if local_idx >= latent_array.shape[0]:
#                     latent_idx = local_idx % latent_array.shape[0]
#                 else:
#                     latent_idx = local_idx
#                 latent = latent_array[latent_idx, :]
#             elif latent_array.ndim == 1:
#                 # [latent_dim] - single sample
#                 latent = latent_array
#             else:
#                 raise ValueError(f"Unexpected latent array shape: {latent_array.shape}")
            
#             # Convert to tensors
#             input_ids_tensor = torch.from_numpy(input_ids).long()
#             attention_mask_tensor = torch.from_numpy(attention_mask).float()
#             latent_tensor = torch.from_numpy(latent).float()
            
#             # Ensure latent shape [1, latent_dim]
#             if latent_tensor.dim() == 1:
#                 latent_tensor = latent_tensor.unsqueeze(0)
            
#             return {
#                 'input_ids': input_ids_tensor,
#                 'attention_mask': attention_mask_tensor,
#                 'latent': latent_tensor,
#                 'file_idx': file_idx,
#                 'sample_idx': local_idx,
#                 'global_idx': idx
#             }
            
#         except Exception as e:
#             print(f"Error loading sample {idx}: {e}")
#             # Return an empty sample
#             return {
#                 'input_ids': torch.zeros(512, dtype=torch.long),
#                 'attention_mask': torch.ones(512, dtype=torch.float),  # uses ones, not zeros
#                 'latent': torch.zeros(1, 1024, dtype=torch.float),
#                 'file_idx': -1,
#                 'sample_idx': -1,
#                 'global_idx': idx
#             }

# def get_direct_dataloaders(config, tokenizer=None):
#     """Get dataloaders that read directly from tokenized and latent files."""
#     from torch.utils.data import DataLoader, random_split
    
#     # Extract paths from config
#     data_config = config.data
    
#     # Get paths from config
#     if hasattr(data_config, 'token_dir'):
#         token_dir = data_config.token_dir
#     else:
#         # Fallback to old config structure
#         token_dir = "/inspire/ssd/project/future-reading/public/jiaquan/latent/MM-LDLM/preprocessed_data/e5_embedding/output_bert_tokenized/tokens"
    
#     if hasattr(data_config, 'latent_dir'):
#         latent_dir = data_config.latent_dir
#     else:
#         # Fallback to old config structure
#         latent_dir = "/inspire/ssd/project/future-reading/public/jiaquan/latent/MM-LDLM/preprocessed_data/e5_embedding/output_sonar_1024/embeddings"
    
#     print(f"Loading from token_dir: {token_dir}")
#     print(f"Loading from latent_dir: {latent_dir}")
    
#     # Check if directories exist
#     token_path = Path(token_dir)
#     latent_path = Path(latent_dir)
    
#     if not token_path.exists():
#         raise ValueError(f"Token directory does not exist: {token_dir}")
#     if not latent_path.exists():
#         raise ValueError(f"Latent directory does not exist: {latent_dir}")
    
#     # List files to verify
#     token_files = list(token_path.glob("*.npy")) + list(token_path.glob("*.npz"))
#     latent_files = list(latent_path.glob("*.npy"))
    
#     print(f"Found {len(token_files)} token files")
#     print(f"Found {len(latent_files)} latent files")
    
#     # Show some sample file names
#     if token_files:
#         print("Sample token files:")
#         for f in token_files[:3]:
#             print(f"  {f.name}")
    
#     if latent_files:
#         print("Sample latent files:")
#         for f in latent_files[:3]:
#             print(f"  {f.name}")
    
#     # Create dataset
#     full_dataset = DirectFileDataset(
#         token_dir=token_dir,
#         latent_dir=latent_dir,
#         max_samples=data_config.get('max_samples', None),
#         verbose=True
#     )
    
#     if len(full_dataset) == 0:
#         raise ValueError(f"No valid file pairs found between {token_dir} and {latent_dir}")
    
#     print(f"Total samples: {len(full_dataset)}")
    
#     # Split into train/val
#     val_ratio = data_config.get('val_ratio', 0.05)
#     val_size = int(len(full_dataset) * val_ratio)
#     train_size = len(full_dataset) - val_size
    
#     train_dataset, val_dataset = random_split(
#         full_dataset, 
#         [train_size, val_size],
#         generator=torch.Generator().manual_seed(config.training.seed)
#     )
    
#     print(f"Train samples: {len(train_dataset)}")
#     print(f"Val samples: {len(val_dataset)}")
    
#     # Collate function
#     def collate_fn(batch):
#         input_ids = torch.stack([item['input_ids'] for item in batch])
#         attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
#         # Get latent dimension from config or first sample
#         latent_dim = config.model.get('latent_dim', 1024)
        
#         # Process latents
#         latents = []
#         for item in batch:
#             latent = item['latent']
#             # Ensure shape [1, latent_dim]
#             if latent.dim() == 1:
#                 latent = latent.unsqueeze(0)
#             elif latent.dim() == 2 and latent.shape[0] != 1:
#                 latent = latent[0:1]
#             latents.append(latent)
        
#         latent_tensor = torch.cat(latents, dim=0)  # [batch_size, 1, latent_dim]
        
#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'latent': latent_tensor
#         }
    
#     # Create distributed samplers
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         train_sampler = DistributedSampler(
#             train_dataset,
#             num_replicas=torch.distributed.get_world_size(),
#             rank=torch.distributed.get_rank(),
#             shuffle=True,
#             seed=config.training.seed
#         )
#         val_sampler = DistributedSampler(
#             val_dataset,
#             num_replicas=torch.distributed.get_world_size(),
#             rank=torch.distributed.get_rank(),
#             shuffle=False
#         )
#     else:
#         train_sampler = None
#         val_sampler = None
    
#     # DataLoader settings
#     num_workers = data_config.get('num_workers', 16)
#     persistent_workers = num_workers > 0
#     prefetch_factor = 2 if num_workers > 0 else None
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.training.train_batch_size,
#         sampler=train_sampler,
#         shuffle=(train_sampler is None),
#         num_workers=num_workers,
#         collate_fn=collate_fn,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=persistent_workers,
#         prefetch_factor=prefetch_factor
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.training.eval_batch_size,
#         sampler=val_sampler,
#         shuffle=False,
#         num_workers=min(8, num_workers),
#         collate_fn=collate_fn,
#         pin_memory=True,
#         drop_last=False,
#         persistent_workers=persistent_workers
#     )
    
#     return train_loader, val_loader


# # Main entry point for backward compatibility
# def get_simple_dataloaders(config, tokenizer=None):
#     """Main entry point - automatically chooses the right data loader."""
#     use_preprocessed = config.data.get('use_preprocessed', True)
    
#     if use_preprocessed:
#         print("Using preprocessed file-based data loader")
#         return get_direct_dataloaders(config, tokenizer)
#     else:
#         print("Using JSON-based data loader (needs tokenizer)")
#         # Fallback to old JSON-based loader
#         return get_json_dataloaders(config, tokenizer)
