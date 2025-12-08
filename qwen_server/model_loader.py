"""
Model loader for Qwen2-VL.
Handles downloading, caching, and initialization of the vision-language model.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Manages loading and initialization of Qwen2-VL models.
    Supports automatic downloading from HuggingFace Hub.
    """
    
    def __init__(self, config: dict):
        """
        Initialize model loader with configuration.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.model_name = config['model']['name']
        self.device = config['model']['device']
        self.dtype = config['model']['dtype']
        self.cache_dir = os.path.expanduser(config['model']['cache_dir'])
        self.max_length = config['model'].get('max_length', 512)
        
        # Multi-GPU configuration
        self.use_data_parallel = config['model'].get('use_data_parallel', False)
        self.data_parallel_devices = config['model'].get('data_parallel_devices', None)
        
        self.model = None
        self.processor = None
        
        # Convert dtype string to torch dtype
        self.torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
    
    def load_model(self) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
        """
        Load the Qwen2-VL model and processor.
        Downloads from HuggingFace if not cached.
        
        Returns:
            Tuple of (model, processor)
        """
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dtype: {self.dtype}")
        
        # Check CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
            self.torch_dtype = torch.float32
        
        # Log GPU information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        try:
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Load model with appropriate parallelism strategy
            logger.info("Loading model (this may take a few minutes on first run)...")
            
            if self.use_data_parallel and self.device == "cuda" and torch.cuda.device_count() > 1:
                # Data Parallel mode: Load on single GPU, then wrap with DataParallel
                logger.info(f"Using DataParallel mode across {torch.cuda.device_count()} GPUs")
                
                # Load model on primary GPU (cuda:0) without device_map
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True
                )
                
                # Move to primary GPU and ensure it's in eval mode before wrapping
                self.model = self.model.to('cuda:0')
                self.model.eval()
                
                # Wrap with DataParallel
                if self.data_parallel_devices:
                    device_ids = self.data_parallel_devices
                    logger.info(f"Using specified GPU devices: {device_ids}")
                else:
                    device_ids = list(range(torch.cuda.device_count()))
                    logger.info(f"Using all available GPU devices: {device_ids}")
                
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
                logger.info(f"Model wrapped in DataParallel across GPUs: {device_ids}")
                
                # Force model replication to all GPUs by doing a dummy forward pass
                logger.info("Warming up model on all GPUs...")
                try:
                    with torch.no_grad():
                        # Create dummy input on primary device
                        dummy_input = torch.zeros(len(device_ids), 10, device='cuda:0', dtype=self.torch_dtype)
                        # This will trigger replication to all GPUs
                        _ = self.model.module.model.embed_tokens(dummy_input.long())
                    logger.info("Model successfully replicated across all GPUs")
                except Exception as e:
                    logger.warning(f"Could not warm up model: {e}. Model will replicate on first real batch.")
                
            else:
                # Standard mode: Use device_map auto for model sharding or single GPU
                if self.use_data_parallel:
                    logger.warning("DataParallel requested but only 1 GPU available or not using CUDA")
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Move to device if not using device_map
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            # Enable GPU optimizations
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN auto-tuner for optimized kernels")
            
            # Configure generation to avoid warnings
            if hasattr(self.model, 'generation_config'):
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
                self.model.generation_config.top_k = None
                self.model.generation_config.do_sample = False
            
            logger.info("Model loaded successfully!")
            
            # Log model size
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model parameters: {param_count / 1e9:.2f}B")
            
            return self.model, self.processor
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def verify_installation(self) -> bool:
        """
        Verify that the model can be loaded successfully.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.load_model()
            logger.info("Model verification successful!")
            return True
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        info = {
            "model_name": self.model_name,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
        
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
            info["gpu_memory_cached"] = torch.cuda.memory_reserved() / 1e9
        
        return info

def download_model(model_name: str, cache_dir: str) -> bool:
    """
    Pre-download model to cache directory.
    Useful for setup scripts.
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Cache directory path
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"Downloading model: {model_name}")
        
        # Download processor
        AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Download model
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        logger.info("Model downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False
