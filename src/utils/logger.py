import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

# Define log levels mapping
LOG_LEVELS: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_log_level() -> int:
    """
    Get log level from environment variable or use default.

    Returns:
        int: Logging level
    """
    env_level = os.environ.get("PYSCREW_LOG_LEVEL", "INFO")
    return LOG_LEVELS.get(env_level.upper(), logging.INFO)


def get_logs_dir() -> Path:
    """
    Get or create logs directory.

    Returns:
        Path: Path to logs directory
    """
    # Define logs directory relative to this file
    # Assuming this file is in src/utils/
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    logs_dir = base_dir / "src" / "utils" / "logs"

    # Create directory if it doesn't exist
    logs_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir


def setup_console_handler() -> logging.Handler:
    """
    Configure and return a console log handler.

    Returns:
        logging.Handler: Configured console handler
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    )
    return console_handler


def setup_file_handler(log_file: Optional[Union[str, Path]] = None) -> logging.Handler:
    """
    Configure and return a file log handler.

    Args:
        log_file: Optional custom path to log file

    Returns:
        logging.Handler: Configured file handler
    """
    if log_file is None:
        # Generate default log filename based on date
        logs_dir = get_logs_dir()
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = logs_dir / f"pyscrew_{timestamp}.log"
    else:
        log_file = Path(log_file)
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    )
    return file_handler


def get_logger(
    name: str,
    level: Optional[Union[str, int]] = None,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Get a logger configured for use in PyScrew.

    This function provides a consistent logging interface across the library:
    - Creates a logger with the specified name
    - Sets appropriate log level from arguments or environment
    - Configures console output by default
    - Sets up file logging in utils/logs by default

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Optional log level (default: from environment or INFO)
        log_file: Optional custom path to log file
        console: Whether to enable console output (default: True)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process...")
        2024-01-29 10:30:45 - INFO - my_module - Starting process...
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Only configure if the logger hasn't been set up
    if not logger.handlers:
        # Determine log level
        if level is None:
            level = get_log_level()
        elif isinstance(level, str):
            level = LOG_LEVELS.get(level.upper(), logging.INFO)

        logger.setLevel(level)

        # Add console handler if enabled
        if console:
            logger.addHandler(setup_console_handler())

        # Add file handler
        logger.addHandler(setup_file_handler(log_file))

        # Prevent propagation to root logger
        logger.propagate = False

    return logger
