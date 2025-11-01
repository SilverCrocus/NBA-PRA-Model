"""
Centralized Logging Configuration

Provides consistent logging setup across all production modules.

Author: NBA PRA Prediction System
Date: 2025-11-01
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_production_logging(
    name: str,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Setup production logger with console and file handlers.

    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to PROJECT_ROOT/logs)
        console: Enable console logging
        file: Enable file logging

    Returns:
        Configured logger instance

    Examples:
        >>> logger = setup_production_logging('predictor')
        >>> logger.info("Starting predictions")

        >>> logger = setup_production_logging('model_trainer', level='DEBUG')
        >>> logger.debug("Detailed debug info")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file:
        # Default to PROJECT_ROOT/logs if not specified
        if log_dir is None:
            from production.config import PROJECT_ROOT
            log_dir = PROJECT_ROOT / 'logs'

        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get or create a logger with production configuration.

    Convenience wrapper around setup_production_logging() with sensible defaults.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    return setup_production_logging(name, level=level)
