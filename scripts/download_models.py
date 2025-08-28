"""
Script to download required models.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging import logger


def download_model(model_name: str, cache_dir: str = None) -> bool:
    """
    Download a model from Hugging Face.
    
    Args:
        model_name: Name of the model to download
        cache_dir: Directory to cache the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {model_name}...")
        
        # Check if model exists
        if cache_dir and Path(cache_dir).exists():
            logger.info(f"Model {model_name} already exists in cache")
            return True
        
        # Download model
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        
        logger.info(f"Successfully downloaded {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False


def download_gpt_oss_models(size: str = "20b") -> bool:
    """
    Download gpt-oss models.
    
    Args:
        size: Model size ("20b" or "120b")
        
    Returns:
        True if successful
    """
    model_name = f"openai/gpt-oss-{size}"
    cache_dir = f"./models/gpt-oss-{size}"
    
    return download_model(model_name, cache_dir)


def download_target_model(model_name: str = "microsoft/DialoGPT-small") -> bool:
    """
    Download target model for training.
    
    Args:
        model_name: Name of the target model
        
    Returns:
        True if successful
    """
    cache_dir = f"./models/{model_name.replace('/', '_')}"
    return download_model(model_name, cache_dir)


def check_disk_space() -> bool:
    """Check if there's enough disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        # gpt-oss-20b is ~40GB, gpt-oss-120b is ~240GB
        required_gb = 50  # Conservative estimate for 20b model + overhead
        free_gb = free // (1024**3)
        
        if free_gb < required_gb:
            logger.warning(f"Low disk space: {free_gb}GB available, {required_gb}GB recommended")
            return False
        
        logger.info(f"Disk space check passed: {free_gb}GB available")
        return True
        
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Proceed anyway


def main():
    """Main download script."""
    parser = argparse.ArgumentParser(description="Download models for model-run training")
    parser.add_argument("--gpt-oss-size", choices=["20b", "120b"], default="20b",
                       help="Size of gpt-oss model to download (default: 20b)")
    parser.add_argument("--target-model", default="microsoft/DialoGPT-small",
                       help="Target model to download (default: microsoft/DialoGPT-small)")
    parser.add_argument("--skip-space-check", action="store_true",
                       help="Skip disk space check")
    
    args = parser.parse_args()
    
    logger.info("Model Download Script")
    logger.info("=" * 30)
    
    # Check disk space
    if not args.skip_space_check:
        if not check_disk_space():
            response = input("Continue anyway? [y/N]: ")
            if not response.lower().startswith('y'):
                logger.info("Download cancelled")
                return 1
    
    success = True
    
    # Download gpt-oss model
    logger.info(f"Downloading gpt-oss-{args.gpt_oss_size} model...")
    if not download_gpt_oss_models(args.gpt_oss_size):
        success = False
    
    # Download target model
    logger.info(f"Downloading target model: {args.target_model}")
    if not download_target_model(args.target_model):
        success = False
    
    if success:
        logger.info("✅ All models downloaded successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Run basic demo: python examples/basic_demo.py")
        logger.info("  2. Run full training: python src/main.py")
        return 0
    else:
        logger.error("❌ Some downloads failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
