"""Logging configuration for the SmartDocQ application."""

import logging
import sys
import time
from pathlib import Path
from functools import wraps

from .config import settings


def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set log levels for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    # Hide watchfiles change detection messages
    logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
    
    # Create OCR-specific logger with its own file handler
    ocr_logger = logging.getLogger("smartdocq.ocr")
    ocr_log_file = Path(log_dir) / "ocr_performance.log"
    ocr_file_handler = logging.FileHandler(ocr_log_file)
    ocr_file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    ocr_logger.addHandler(ocr_file_handler)
    ocr_logger.setLevel(log_level)
    
    # Return logger for the application
    return logging.getLogger("smartdocq")


# Create logger instance
logger = setup_logging()


def get_logger(name):
    """Get a logger with the given name.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(f"smartdocq.{name}")


def log_execution_time(logger_name="smartdocq", level=logging.INFO):
    """Decorator to log the execution time of a function.
    
    Args:
        logger_name (str): Name of the logger to use
        level (int): Logging level
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the logger
            log = logging.getLogger(logger_name)
            
            # Get function name and arguments for logging
            func_name = func.__name__
            
            # Log start of function execution
            log.log(level, f"Starting {func_name}")
            
            # Record start time
            start_time = time.time()
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log execution time
            log.log(level, f"Completed {func_name} in {execution_time:.2f} seconds")
            
            return result
        return wrapper
    return decorator


def log_step_execution_time(logger_name="smartdocq", level=logging.INFO):
    """Context manager to log the execution time of a code block.
    
    Args:
        step_name (str): Name of the step being timed
        logger_name (str): Name of the logger to use
        level (int): Logging level
        
    Returns:
        Context manager
    """
    class StepTimer:
        def __init__(self, step_name):
            self.step_name = step_name
            self.logger = logging.getLogger(logger_name)
            self.level = level
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            self.logger.log(self.level, f"Starting step: {self.step_name}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            execution_time = time.time() - self.start_time
            self.logger.log(self.level, f"Completed step: {self.step_name} in {execution_time:.2f} seconds")
            
    return StepTimer