"""
logger.py

Lightweight logging utilities for the project.
This keeps logging configuration centralized and consistent across modules.
"""

from __future__ import annotations

import logging


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """
    Create or retrieve a configured project logger.

    Parameters
    ----------
    name : str
        Logger name, usually `__name__`.
    level : int, optional
        Logging level.
    log_format : str, optional
        Formatter string.
    date_format : str, optional
        Datetime format for log messages.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_project_log_level(level: int) -> None:
    """Set the root logging level for the current Python process."""
    logging.getLogger().setLevel(level)


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Logger initialized successfully.")