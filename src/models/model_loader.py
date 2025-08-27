"""
Model loading and management utilities.
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import Tuple, Optional, Dict, Any
import gc
from pathlib import Path

from ..config.settings import settings
from ..utils.logging import LoggerMixin, log_model_info, log_error_with_context


class ModelLoader(LoggerMixin):
    """Handle loading and management of models."""
    
    def __init__(self):
        self.loaded_models = {}
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if settings.system.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return settings.system.device
    
    def load_trainer_model(self, force_reload: bool = False) -> Tuple[Any, Any]:
        """
        Load the gpt-oss trainer model.
        
        Args:
            force_reload: Force reload even if already cached
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = settings.model.trainer_model_name
        cache_key = f"trainer_{model_name}"
        
        if not force_reload and cache_key in self.loaded_models:
            self.logger.info(f"Using cached trainer model: {model_name}")
            return self.loaded_models[cache_key]
        
        self.logger.info(f"Loading trainer model: {model_name}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configure model loading based on available memory
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True
            }
            
            # Use 4-bit quantization for large models on GPU
            if self.device == "cuda" and "120b" in model_name.lower():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                model_kwargs["quantization_config"] = quantization_config
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Cache the loaded model
            self.loaded_models[cache_key] = (model, tokenizer)
            
            # Log model info
            num_params = sum(p.numel() for p in model.parameters())
            log_model_info(model_name, num_params, str(model.device))
            
            return model, tokenizer
            
        except Exception as e:
            log_error_with_context(e, "loading trainer model")
            # Fallback to smaller model
            if "120b" in model_name:
                self.logger.warning("Falling back to gpt-oss-20b")
                settings.model.trainer_model_name = "openai/gpt-oss-20b"
                return self.load_trainer_model(force_reload=True)
            raise
    
    def load_target_model(self, force_reload: bool = False) -> Tuple[Any, Any]:
        """
        Load the target model to be trained.
        
        Args:
            force_reload: Force reload even if already cached
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = settings.model.target_model_name
        cache_key = f"target_{model_name}"
        
        if not force_reload and cache_key in self.loaded_models:
            self.logger.info(f"Using cached target model: {model_name}")
            return self.loaded_models[cache_key]
        
        self.logger.info(f"Loading target model: {model_name}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model (smaller model, so simpler config)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                model = model.to(self.device)
            
            # Enable gradient computation for training
            for param in model.parameters():
                param.requires_grad = True
            
            # Cache the loaded model
            self.loaded_models[cache_key] = (model, tokenizer)
            
            # Log model info
            num_params = sum(p.numel() for p in model.parameters())
            log_model_info(model_name, num_params, str(model.device if hasattr(model, 'device') else self.device))
            
            return model, tokenizer
            
        except Exception as e:
            log_error_with_context(e, "loading target model")
            raise
    
    def unload_model(self, model_type: str):
        """
        Unload a model to free memory.
        
        Args:
            model_type: 'trainer' or 'target'
        """
        keys_to_remove = [key for key in self.loaded_models.keys() if key.startswith(model_type)]
        
        for key in keys_to_remove:
            if key in self.loaded_models:
                model, tokenizer = self.loaded_models[key]
                del model, tokenizer
                del self.loaded_models[key]
                self.logger.info(f"Unloaded model: {key}")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self, model) -> Dict[str, Any]:
        """Get information about a model."""
        try:
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                "total_parameters": num_params,
                "trainable_parameters": trainable_params,
                "device": str(model.device if hasattr(model, 'device') else 'unknown'),
                "dtype": str(next(model.parameters()).dtype),
                "model_size_mb": num_params * 4 / (1024 * 1024)  # Rough estimate for float32
            }
        except Exception as e:
            log_error_with_context(e, "getting model info")
            return {}
    
    def save_target_model(self, model, tokenizer, save_path: str):
        """
        Save the target model and tokenizer.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save  
            save_path: Path to save to
        """
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            self.logger.info(f"Saved target model to: {save_path}")
            
        except Exception as e:
            log_error_with_context(e, f"saving model to {save_path}")
            raise
    
    def load_saved_target_model(self, load_path: str) -> Tuple[Any, Any]:
        """
        Load a previously saved target model.
        
        Args:
            load_path: Path to load from
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            load_path = Path(load_path)
            
            if not load_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {load_path}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(load_path), trust_remote_code=True)
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                str(load_path),
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Move to device if needed
            if self.device != "cuda":
                model = model.to(self.device)
            
            self.logger.info(f"Loaded saved target model from: {load_path}")
            
            return model, tokenizer
            
        except Exception as e:
            log_error_with_context(e, f"loading saved model from {load_path}")
            raise
    
    def cleanup(self):
        """Clean up all loaded models and free memory."""
        self.logger.info("Cleaning up loaded models...")
        
        for key in list(self.loaded_models.keys()):
            model, tokenizer = self.loaded_models[key]
            del model, tokenizer
            
        self.loaded_models.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Model cleanup completed")


# Singleton instance for easy access
model_loader = ModelLoader()