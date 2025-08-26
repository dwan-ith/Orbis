"""
Model-Run Model Training

A revolutionary approach to AI training where gpt-oss intelligently 
trains smaller models through real-time analysis and adaptive data generation.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.evaluator import ModelEvaluator
from .core.analyzer import PerformanceAnalyzer
from .core.trainer import AdaptiveTrainer
from .core.data_generator import TrainingDataGenerator
from .models.model_loader import ModelLoader

__all__ = [
    "ModelEvaluator",
    "PerformanceAnalyzer", 
    "AdaptiveTrainer",
    "TrainingDataGenerator",
    "ModelLoader"
]