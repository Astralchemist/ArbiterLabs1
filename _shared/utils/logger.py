"""
Logging Utilities

Standardized logging setup for strategies.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_file: str = None,
    level: str = "INFO",
    log_to_console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_strategy_logger(strategy_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Get logger for a strategy with automatic file naming.

    Args:
        strategy_name: Name of the strategy
        log_dir: Directory for log files

    Returns:
        Configured logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{strategy_name}_{timestamp}.log"

    return setup_logger(strategy_name, log_file)
