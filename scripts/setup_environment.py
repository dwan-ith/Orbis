"""
Environment setup script.
"""

import os
import sys
import subprocess
from pathlib import Path
import json


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version OK: {version.major}.{version.minor}.{version.micro}")
    return True


def check_gpu_availability():
    """Check GPU availability and CUDA installation."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"✅ GPU available: {gpu_name} ({gpu_count} device(s))")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU detected - training will use CPU (much slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet - GPU check will be done after installation")
        return False


def install_requirements():
    """Install required packages."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("📦 Installing required packages...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True, check=True)
        
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print(f"   Error output: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories."""
    base_path = Path(__file__).parent.parent
    directories = [
        "data/benchmarks",
        "data/results", 
        "models/checkpoints",
        "models/pretrained",
        "output",
        "cache"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created")
    return True


def create_test_data():
    """Create test data files."""
    base_path = Path(__file__).parent.parent
    
    # Create test questions
    test_questions = [
        {
            "question": "What is 2 + 2?",
            "expected_answer": "4",
            "category": "arithmetic",
            "difficulty": "easy"
        },
        {
            "question": "If Tom is taller than Jerry, and Jerry is taller than Spike, who is shortest?",
            "expected_answer": "Spike",
            "category": "logic",
            "difficulty": "medium"
        },
        {
            "question": "What comes after Monday?",
            "expected_answer": "Tuesday", 
            "category": "knowledge",
            "difficulty": "easy"
        },
        {
            "question": "Is a cat a mammal?",
            "expected_answer": "Yes",
            "category": "knowledge",
            "difficulty": "easy"
        },
        {
            "question": "If it's raining, should I bring an umbrella?",
            "expected_answer": "Yes",
            "category": "reasoning",
            "difficulty": "easy"
        }
    ]
    
    test_file = base_path / "data" / "benchmarks" / "test_questions.json"
    
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_questions, f, indent=2)
        
        print("✅ Test data created")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return False


def check_memory():
    """Check available memory."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total // (1024**3)
        
        if memory_gb < 8:
            print(f"⚠️  Low memory: {memory_gb}GB available")
            print("   Consider using gpt-oss-20b instead of gpt-oss-120b")
        else:
            print(f"✅ Memory OK: {memory_gb}GB available")
        
        return True
        
    except ImportError:
        print("⚠️  Could not check memory (psutil not installed)")
        return True


def create_env_file():
    """Create example .env file."""
    base_path = Path(__file__).parent.parent
    env_file = base_path / ".env.example"
    
    env_content = """# Model Run Training Environment Variables

# Model Selection
TRAINER_MODEL=openai/gpt-oss-20b
TARGET_MODEL=microsoft/DialoGPT-small

# Training Parameters
MAX_ITERATIONS=10
LEARNING_RATE=5e-5
BATCH_SIZE=4

# System Settings
OUTPUT_DIR=./output
LOG_LEVEL=INFO

# Hardware Settings (auto-detected if not set)
# DEVICE=cuda
# DEVICE=cpu
"""
    
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print("✅ Created .env.example file")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def run_basic_tests():
    """Run basic functionality tests."""
    try:
        print("🧪 Running basic tests...")
        
        # Test imports
        print("   Testing imports...")
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from config.settings import settings
        from utils.logging import logger
        from models.model_loader import ModelLoader
        
        print("   ✅ Core imports successful")
        
        # Test configuration
        print("   Testing configuration...")
        config_dict = settings.to_dict()
        assert "model" in config_dict
        assert "training" in config_dict
        print("   ✅ Configuration OK")
        
        # Test logging
        print("   Testing logging...")
        logger.info("Test log message")
        print("   ✅ Logging OK")
        
        print("✅ Basic tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup function."""
    print("🚀 Model-Run Training Environment Setup")
    print("=" * 45)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    # Create test data
    if not create_test_data():
        success = False
    
    # Create env file
    if not create_env_file():
        success = False
    
    # Check system capabilities
    check_gpu_availability()
    check_memory()
    
    # Run basic tests
    if not run_basic_tests():
        success = False
    
    print("\n" + "=" * 45)
    
    if success:
        print("✅ Environment setup completed successfully!")
        print("\nNext steps:")
        print("  1. Download models: python scripts/download_models.py")
        print("  2. Run demo: python examples/basic_demo.py")
        print("  3. Start training: python src/main.py")
        return 0
    else:
        print("❌ Environment setup encountered some issues")
        print("   Please check the errors above and resolve them")
        return 1


if __name__ == "__main__":
    sys.exit(main())