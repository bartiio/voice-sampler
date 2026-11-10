"""
backend/app/utils/logger.py
Standardized logging utilities for Voice Sampler backend.
"""

import logging
import os
import sys
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: Optional[str] = "voice_sampler",
    level: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a standardized logger instance."""
    if level is None:
        level = os.getenv("VOICE_SAMPLER_LOG_LEVEL", "INFO").upper()
    else:
        level = level.upper()

    numeric_level = getattr(logging, level, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger child under voice_sampler root namespace."""
    full_name = f"voice_sampler.{name}" if not name.startswith("voice_sampler") else name
    return logging.getLogger(full_name)
