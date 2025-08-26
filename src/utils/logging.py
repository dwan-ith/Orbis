"""
Logging utilities for the model-run training system.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Get level from settings if not provided
    if level is None:
        level = settings.system.log_level
        
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("model_run_training")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "model_run_training") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


# Setup default logger
_default_log_file = Path(settings.system.output_dir) / "training.log"
logger = setup_logging(log_file=str(_default_log_file))


class LoggerMixin:
    """Mixin class to add logging capability to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        class_name = self.__class__.__name__
        return logging.getLogger(f"model_run_training.{class_name}")


def log_training_step(iteration: int, step: str, details: str = ""):
    """Log a training step with consistent formatting."""
    logger.info(f"[Iteration {iteration:02d}] {step}" + (f" - {details}" if details else ""))


def log_performance_metrics(metrics: dict, prefix: str = ""):
    """Log performance metrics in a structured way."""
    prefix_str = f"{prefix} " if prefix else ""
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{prefix_str}{key}: {value:.4f}")
        else:
            logger.info(f"{prefix_str}{key}: {value}")


def log_model_info(model_name: str, num_parameters: int, device: str):
    """Log model information."""
    logger.info(f"Loaded {model_name}")
    logger.info(f"Parameters: {num_parameters:,}")
    logger.info(f"Device: {device}")


def log_error_with_context(error: Exception, context: str = ""):
    """Log error with additional context."""
    context_str = f" in {context}" if context else ""
    logger.error(f"Error{context_str}: {type(error).__name__}: {str(error)}")


# Export main logger for easy import
__all__ = [
    "setup_logging",
    "get_logger", 
    "logger",
    "LoggerMixin",
    "log_training_step",
    "log_performance_metrics", 
    "log_model_info",
    "log_error_with_context"
]