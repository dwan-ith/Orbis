"""
Basic demonstration of model-run training.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config.settings import settings
from utils.logging import logger
from models.target_model import TargetModel
from core.evaluator import ModelEvaluator
from core.analyzer import PerformanceAnalyzer
from core.data_generator import TrainingDataGenerator
from core.trainer import AdaptiveTrainer


def demonstrate_analysis():
    """Demonstrate the analysis capability."""
    print("\n🔍 DEMONSTRATION: Performance Analysis")
    print("=" * 50)
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    if not analyzer.load_trainer_model():
        print("❌ Failed to load trainer model for analysis")
        return
    
    # Example of a poor response
    question = "If Tom is taller than Jerry, and Jerry is taller than Spike, who is shortest?"
    poor_response = "Tom is shortest"
    expected_answer = "Spike"
    
    print(f"Question: {question}")
    print(f"Poor Response: {poor_response}")
    print(f"Expected: {expected_answer}")
    print("\nAnalyzing response...")
    
    # Analyze the response
    analysis = analyzer.analyze_response(question, poor_response, expected_answer)
    
    print(f"\n📊 Analysis Results:")
    print(f"Weaknesses: {analysis.get('weaknesses', [])}")
    print(f"Recommendations: {analysis.get('recommendations', [])}")
    print(f"Priority: {analysis.get('priority', 'unknown')}")
    
    analyzer.cleanup()


def demonstrate_data_generation():
    """Demonstrate training data generation."""
    print("\n🏭 DEMONSTRATION: Training Data Generation