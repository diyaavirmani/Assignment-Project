"""
Structured logging configuration for the 3GPP RAG Assistant
"""
import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "3gpp_rag",
    level: str = "INFO",
    log_file: str = None,
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (use __name__ in modules for hierarchy)
        level: Logging level: DEBUG, INFO, WARNING, ERROR
        log_file: Optional path to write logs to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured, avoid duplicate handlers

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the root 3gpp_rag hierarchy.

    Use this in every module:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(f"3gpp_rag.{name}")
