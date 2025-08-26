"""
Configuration settings for model-run training system.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for models."""
    # Trainer model (gpt-oss)
    trainer_model_name: str = "openai/gpt-oss-20b"  # or gpt-oss-120b
    trainer_model_path: Optional[str] = None
    
    # Target model (model being trained)
    target_model_name: str = "microsoft/DialoGPT-small"
    target_model_path: Optional[str] = None
    
    # Model parameters
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass  
class TrainingConfig:
    """Configuration for training parameters."""
    max_iterations: int = 10
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 1
    warmup_steps: int = 100
    eval_steps: int = 50
    save_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    test_batch_size: int = 8
    num_test_questions: int = 20
    evaluation_metrics: List[str] = None
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["accuracy", "reasoning_quality", "confidence"]


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    log_level: str = "INFO"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Data paths
    data_dir: str = "./data"
    results_dir: str = "./data/results" 
    checkpoints_dir: str = "./checkpoints"
    
    # Analysis settings
    analysis_max_examples: int = 10
    generation_max_examples: int = 5


class Settings:
    """Main settings class combining all configurations."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig() 
        self.system = SystemConfig()
        
        # Create directories
        self._create_directories()
        
        # Load environment overrides
        self._load_env_overrides()
    
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            self.system.output_dir,
            self.system.cache_dir,
            self.system.data_dir,
            self.system.results_dir,
            self.system.checkpoints_dir
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        # Model overrides
        if os.getenv("TRAINER_MODEL"):
            self.model.trainer_model_name = os.getenv("TRAINER_MODEL")
        if os.getenv("TARGET_MODEL"):
            self.model.target_model_name = os.getenv("TARGET_MODEL")
            
        # Training overrides
        if os.getenv("MAX_ITERATIONS"):
            self.training.max_iterations = int(os.getenv("MAX_ITERATIONS"))
        if os.getenv("LEARNING_RATE"):
            self.training.learning_rate = float(os.getenv("LEARNING_RATE"))
        if os.getenv("BATCH_SIZE"):
            self.training.batch_size = int(os.getenv("BATCH_SIZE"))
            
        # System overrides
        if os.getenv("OUTPUT_DIR"):
            self.system.output_dir = os.getenv("OUTPUT_DIR")
        if os.getenv("LOG_LEVEL"):
            self.system.log_level = os.getenv("LOG_LEVEL")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "system": self.system.__dict__
        }


# Global settings instance
settings = Settings()