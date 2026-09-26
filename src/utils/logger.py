"""Structured logging setup for the IoT anomaly detection system."""

from __future__ import annotations

import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a logger with a structured console handler.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Minimum severity level to emit.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger
