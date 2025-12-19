"""
Model loader for Qwen VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL).
Handles downloading, caching, and initialization of vision-language models.
Supports automatic detection of model family from model name.
Supports quantization: int8, int4 (nf4/fp4) via bitsandbytes.
"""

import os
import re
import torch
from pathlib import Path
from typing import Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)

# Quantization support detection
BITSANDBYTES_AVAILABLE = False
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
    logger.info("bitsandbytes is available for quantization")
except ImportError:
    logger.warning("bitsandbytes not installed. INT8/INT4 quantization will not be available.")

# Model family constants
MODEL_FAMILY_QWEN2_VL = "qwen2_vl"
MODEL_FAMILY_QWEN2_5_VL = "qwen2_5_vl"
MODEL_FAMILY_QWEN3_VL = "qwen3_vl"
MODEL_FAMILY_AUTO = "auto"

# Supported model families and their patterns
# Order matters: check more specific patterns first (qwen3 before qwen2.5 before qwen2)
MODEL_FAMILY_PATTERNS = {
    MODEL_FAMILY_QWEN3_VL: [r"qwen3-vl", r"qwen3_vl"],
    MODEL_FAMILY_QWEN2_5_VL: [r"qwen2\.5-vl", r"qwen2_5-vl", r"qwen2\.5_vl"],
    MODEL_FAMILY_QWEN2_VL: [r"qwen2-vl", r"qwen2_vl"],
}


def detect_model_family(model_name: str) -> str:
    """
    Detect model family from model name.
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-8B-Instruct")
        
    Returns:
        Model family string (qwen2_vl, qwen2_5_vl, or qwen3_vl)
    """
    model_name_lower = model_name.lower()
    
    # Check patterns in order (qwen3 before qwen2.5 before qwen2 to avoid false matches)
    for family, patterns in MODEL_FAMILY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, model_name_lower):
                logger.info(f"Detected model family: {family} (from pattern: {pattern})")
                return family
    
    # Default to qwen3_vl as it's the newest
    logger.warning(f"Could not detect model family from '{model_name}', defaulting to {MODEL_FAMILY_QWEN3_VL}")
    return MODEL_FAMILY_QWEN3_VL


def get_model_class(model_family: str):
    """
    Get the appropriate model class for the specified family.
    
    Args:
        model_family: Model family string
        
    Returns:
        Model class from transformers
    """
    if model_family == MODEL_FAMILY_QWEN2_VL:
        from transformers import Qwen2VLForConditionalGeneration
        logger.info("Using Qwen2VLForConditionalGeneration")
        return Qwen2VLForConditionalGeneration
    elif model_family == MODEL_FAMILY_QWEN2_5_VL:
        # Qwen2.5-VL uses Qwen2_5_VLForConditionalGeneration in transformers >= 4.45
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            logger.info("Using Qwen2_5_VLForConditionalGeneration")
            return Qwen2_5_VLForConditionalGeneration
        except ImportError:
            # Fallback for older transformers versions
            logger.warning("Qwen2_5_VLForConditionalGeneration not found, trying AutoModelForVision2Seq")
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq
    elif model_family == MODEL_FAMILY_QWEN3_VL:
        # Qwen3-VL uses Qwen3VLForConditionalGeneration in transformers >= 4.51
        try:
            from transformers import Qwen3VLForConditionalGeneration
            logger.info("Using Qwen3VLForConditionalGeneration")
            return Qwen3VLForConditionalGeneration
        except ImportError:
            # Fallback: try AutoModelForVision2Seq with trust_remote_code
            logger.warning("Qwen3VLForConditionalGeneration not found, trying AutoModelForVision2Seq")
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq
    else:
        raise ValueError(f"Unknown model family: {model_family}")


def get_processor_class(model_family: str):
    """
    Get the appropriate processor class for the specified family.
    
    Args:
        model_family: Model family string
        
    Returns:
        Processor class from transformers
    """
    # AutoProcessor works for all Qwen VL models
    from transformers import AutoProcessor
    return AutoProcessor


def check_gpu_compute_capability(min_capability: float = 7.0) -> Tuple[bool, float]:
    """
    Check if GPU supports the minimum compute capability.
    
    Args:
        min_capability: Minimum required compute capability
        
    Returns:
        Tuple of (is_supported, actual_capability)
    """
    if not torch.cuda.is_available():
        return False, 0.0
    
    # Get compute capability of primary GPU
    major, minor = torch.cuda.get_device_capability(0)
    capability = float(f"{major}.{minor}")
    
    return capability >= min_capability, capability


class ModelLoader:
    """
    Manages loading and initialization of Qwen VL models.
    Supports automatic downloading from HuggingFace Hub.
    Handles Qwen2-VL, Qwen2.5-VL, and Qwen3-VL model families.
    Supports INT8 and INT4 quantization via bitsandbytes.
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
        
        # Quantization configuration
        self.quantization_config = config['model'].get('quantization', {})
        self.quant_type = self.quantization_config.get('quant_type', 'nf4')
        self.double_quant = self.quantization_config.get('double_quant', True)
        self.compute_dtype_str = self.quantization_config.get('compute_dtype', 'float16')
        
        # Model family detection
        configured_family = config['model'].get('model_family', MODEL_FAMILY_AUTO)
        if configured_family == MODEL_FAMILY_AUTO:
            self.model_family = detect_model_family(self.model_name)
        else:
            self.model_family = configured_family
            logger.info(f"Using configured model family: {self.model_family}")
        
        # Multi-GPU configuration
        self.use_data_parallel = config['model'].get('use_data_parallel', False)
        self.data_parallel_devices = config['model'].get('data_parallel_devices', None)
        
        self.model = None
        self.processor = None
        
        # Determine if using quantization
        self.use_quantization = self.dtype in ['int8', 'int4']
        
        # Convert dtype string to torch dtype
        self._setup_dtype()
    
    def _setup_dtype(self):
        """Set up torch dtype based on configuration."""
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'int8': torch.float16,  # Compute dtype for int8
            'int4': torch.float16,  # Compute dtype for int4
        }
        
        if self.dtype in ['int8', 'int4']:
            # For quantized models, use the compute dtype
            compute_dtype_map = {
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            self.torch_dtype = compute_dtype_map.get(self.compute_dtype_str, torch.float16)
            logger.info(f"Quantization enabled: {self.dtype} with compute dtype {self.compute_dtype_str}")
        else:
            self.torch_dtype = dtype_map.get(self.dtype, torch.float16)
    
    def _get_quantization_config(self):
        """
        Create BitsAndBytesConfig for quantization.
        
        Returns:
            BitsAndBytesConfig or None if quantization not enabled/available
        """
        if not self.use_quantization:
            return None
        
        if not BITSANDBYTES_AVAILABLE:
            logger.error("bitsandbytes not installed. Cannot use INT8/INT4 quantization.")
            logger.error("Install with: pip install bitsandbytes>=0.41.0")
            raise ImportError("bitsandbytes is required for INT8/INT4 quantization")
        
        # Check GPU compute capability
        is_supported, capability = check_gpu_compute_capability(7.0)
        if not is_supported:
            raise RuntimeError(
                f"GPU compute capability {capability} is too low for quantization. "
                f"Minimum required: 7.0 (Volta or newer)"
            )
        
        logger.info(f"GPU compute capability: {capability}")
        
        # Check for FP8 (not supported on compute < 8.9)
        if self.dtype == 'fp8':
            if capability < 8.9:
                raise RuntimeError(
                    f"FP8 requires compute capability 8.9+ (Ada Lovelace/Hopper). "
                    f"Your GPU has {capability}. Use int8 or int4 instead."
                )
        
        from transformers import BitsAndBytesConfig
        
        if self.dtype == 'int8':
            logger.info("Creating INT8 quantization config (LLM.int8())")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,  # Default threshold for outlier detection
            )
        elif self.dtype == 'int4':
            logger.info(f"Creating INT4 quantization config (quant_type={self.quant_type}, double_quant={self.double_quant})")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quant_type,  # 'nf4' or 'fp4'
                bnb_4bit_use_double_quant=self.double_quant,
                bnb_4bit_compute_dtype=self.torch_dtype,
            )
        
        return None
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load the Qwen VL model and processor.
        Downloads from HuggingFace if not cached.
        
        Returns:
            Tuple of (model, processor)
        """
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Model family: {self.model_family}")
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
            # Get appropriate classes for this model family
            ModelClass = get_model_class(self.model_family)
            ProcessorClass = get_processor_class(self.model_family)
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = ProcessorClass.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Get quantization config if enabled
            quantization_config = self._get_quantization_config()
            
            # Load model with appropriate parallelism strategy
            logger.info("Loading model (this may take a few minutes on first run)...")
            
            # Note: Quantized models cannot be used with DataParallel
            if self.use_quantization:
                if self.use_data_parallel:
                    logger.warning("DataParallel is not compatible with quantized models. Using device_map='auto' instead.")
                
                logger.info(f"Loading quantized model ({self.dtype})...")
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info(f"Model loaded with {self.dtype} quantization")
                
            elif self.use_data_parallel and self.device == "cuda" and torch.cuda.device_count() > 1:
                # Data Parallel mode: Load on single GPU, then wrap with DataParallel
                logger.info(f"Using DataParallel mode across {torch.cuda.device_count()} GPUs")
                
                # Load model on primary GPU (cuda:0) without device_map
                self.model = ModelClass.from_pretrained(
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
                
                self.model = ModelClass.from_pretrained(
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
            model_for_config = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            if hasattr(model_for_config, 'generation_config'):
                model_for_config.generation_config.temperature = None
                model_for_config.generation_config.top_p = None
                model_for_config.generation_config.top_k = None
                model_for_config.generation_config.do_sample = False
            
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
            "model_family": self.model_family,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "config_dtype": self.dtype,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "quantization_enabled": self.use_quantization,
        }
        
        # Add quantization details if enabled
        if self.use_quantization:
            info["quantization_bits"] = 8 if self.dtype == 'int8' else 4
            info["quantization_type"] = self.quant_type if self.dtype == 'int4' else 'llm_int8'
            info["double_quantization"] = self.double_quant if self.dtype == 'int4' else False
            info["compute_dtype"] = self.compute_dtype_str
        
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
            info["gpu_memory_cached"] = torch.cuda.memory_reserved() / 1e9
            
            # Get compute capability
            major, minor = torch.cuda.get_device_capability(0)
            info["gpu_compute_capability"] = f"{major}.{minor}"
        
        return info


def download_model(model_name: str, cache_dir: str, model_family: str = MODEL_FAMILY_AUTO) -> bool:
    """
    Pre-download model to cache directory.
    Useful for setup scripts.
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Cache directory path
        model_family: Model family (auto-detected if not specified)
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"Downloading model: {model_name}")
        
        # Detect family if auto
        if model_family == MODEL_FAMILY_AUTO:
            model_family = detect_model_family(model_name)
        
        # Get appropriate classes
        ModelClass = get_model_class(model_family)
        ProcessorClass = get_processor_class(model_family)
        
        # Download processor
        logger.info("Downloading processor...")
        ProcessorClass.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Download model
        logger.info("Downloading model weights...")
        ModelClass.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        logger.info("Model downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False
