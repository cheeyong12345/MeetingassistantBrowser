"""
Structured Logging Configuration for Meeting Assistant.

This module provides centralized logging configuration with:
- Environment-based log levels
- Rotating file handlers
- Structured formatting with timestamps
- Module-specific loggers
"""

import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


# ANSI color codes for terminal output
class LogColors:
    """ANSI color codes for colorized terminal output."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels in terminal output."""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.CYAN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: f"{LogColors.BOLD}{LogColors.RED}",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for terminal output.

        Args:
            record: The log record to format

        Returns:
            Formatted and colorized log message
        """
        # Add color to level name
        levelname = record.levelname
        if record.levelno in self.LEVEL_COLORS:
            colored_levelname = (
                f"{self.LEVEL_COLORS[record.levelno]}"
                f"{levelname:8}"
                f"{LogColors.RESET}"
            )
            record.levelname = colored_levelname

        # Format the message
        result = super().format(record)

        # Reset levelname to original for file handlers
        record.levelname = levelname

        return result


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: str = "logs",
    log_file: str = "meeting_assistant.log",
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_file: bool = True,
) -> None:
    """Configure the root logger with console and file handlers.

    This function sets up structured logging with:
    - Colored console output (if terminal supports it)
    - Rotating file handler to prevent log files from growing too large
    - Environment-based log level configuration
    - Consistent formatting across all handlers

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            If None, reads from LOG_LEVEL environment variable (default: INFO)
        log_dir: Directory where log files will be stored
        log_file: Name of the log file
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        enable_console: Whether to enable console logging (default: True)
        enable_file: Whether to enable file logging (default: True)

    Example:
        >>> setup_logging(log_level='DEBUG', log_dir='logs')
        >>> logger = get_logger(__name__)
        >>> logger.info('Application started')
    """
    # Determine log level from parameter or environment
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logs directory if it doesn't exist
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        full_log_path = log_path / log_file

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Console formatter (with colors)
    console_format = (
        '%(asctime)s | %(levelname)s | %(name)s | '
        '%(funcName)s:%(lineno)d | %(message)s'
    )
    console_formatter = ColoredFormatter(
        console_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File formatter (without colors)
    file_format = (
        '%(asctime)s | %(levelname)-8s | %(name)-20s | '
        '%(funcName)-15s:%(lineno)-4d | %(message)s'
    )
    file_formatter = logging.Formatter(
        file_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Rotating file handler
    if enable_file:
        file_handler = RotatingFileHandler(
            full_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Log the configuration
    root_logger.info(
        f"Logging initialized: level={log_level}, "
        f"console={enable_console}, file={enable_file}"
    )
    if enable_file:
        root_logger.info(f"Log file: {full_log_path}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified module.

    This function returns a logger configured with the application's
    logging settings. Each module should get its own logger using
    `get_logger(__name__)`.

    Args:
        name: Name of the logger (typically __name__ of the module)

    Returns:
        A configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info('Processing started')
        >>> logger.error('Failed to process', exc_info=True)
    """
    return logging.getLogger(name)


# Convenience function to disable logging for specific modules
def silence_logger(logger_name: str, level: int = logging.WARNING) -> None:
    """Silence a specific logger by setting its level higher.

    Useful for silencing verbose third-party libraries.

    Args:
        logger_name: Name of the logger to silence
        level: Minimum level to log (default: WARNING)

    Example:
        >>> silence_logger('transformers', logging.ERROR)
        >>> silence_logger('urllib3.connectionpool')
    """
    logging.getLogger(logger_name).setLevel(level)


# Initialize logging on module import if not already configured
if not logging.getLogger().handlers:
    setup_logging()
