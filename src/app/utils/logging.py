 
import logging
import sys
from src.app.utils.config import settings

def setup_logger(name: str = "biomed-rag") -> logging.Logger:
    """Configures and returns a standardized system logger."""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is initialized multiple times
    if not logger.handlers:
        logger.setLevel(settings.LOG_LEVEL.upper())
        
        # Standardize format: Timestamp | Level | Module | Message
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Stream directly to standard output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

# Single instance initialization for utility tracking
logger = setup_logger()
