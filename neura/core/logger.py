"""
neura.core.logger

Logging system for the Neura AI Platform.
"""

from __future__ import annotations

import logging
import sys


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


def create_logger(
    name: str = "neura",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a Neura logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            DEFAULT_LOG_FORMAT
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(
    name: str = "neura",
) -> logging.Logger:
    """
    Return a configured Neura logger.
    """

    return create_logger(name)


def set_log_level(
    logger: logging.Logger,
    level: int,
) -> None:
    """
    Change the logger level dynamically.
    """

    logger.setLevel(level)


def log_system_info(
    logger: logging.Logger,
    message: str,
) -> None:
    """
    Log system information.
    """

    logger.info(message)


if __name__ == "__main__":
    logger = get_logger()

    logger.info(
        "Neura Logger initialized successfully."
    )
