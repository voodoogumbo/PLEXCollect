"""Logging utilities for PLEXCollect."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from utils.config import get_config

def setup_logging(config_override: Optional[dict] = None) -> None:
    """Setup application logging."""
    
    try:
        config = get_config()
        log_config = config.logging
    except Exception:
        # Fallback configuration if config loading fails
        log_config = type('LogConfig', (), {
            'level': 'INFO',
            'file_enabled': True,
            'file_path': 'data/plexcollect.log',
            'max_file_size_mb': 10,
            'backup_count': 5
        })()
    
    # Override with provided config
    if config_override:
        for key, value in config_override.items():
            if hasattr(log_config, key):
                setattr(log_config, key, value)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config.level.upper(), logging.INFO))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_config.file_enabled:
        try:
            # Ensure log directory exists
            log_file = Path(log_config.file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_config.file_path,
                maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count
            )
            
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            # If file logging fails, just log to console
            console_handler.addFilter(lambda record: record.levelno >= logging.WARNING)
            logging.warning(f"Failed to setup file logging: {e}")
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('plexapi').setLevel(logging.INFO)
    logging.getLogger('openai').setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)