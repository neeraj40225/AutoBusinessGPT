"""Structured logging, one configured logger per module."""

from __future__ import annotations

import logging
import sys
from typing import Final

from core.config import settings

_FORMAT: Final = "%(asctime)s | %(levelname)-8s | %(name)-26s | %(funcName)-20s | %(message)s"
_DATEFMT: Final = "%Y-%m-%d %H:%M:%S"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger("autobusiness")
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(stream)

    try:
        file_handler = logging.FileHandler(settings.paths.logs / "autobusiness.log")
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(file_handler)
    except OSError:  # pragma: no cover - read-only fs fallback
        pass

    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger."""
    _configure_root()
    short = name.split(".")[-1]
    return logging.getLogger(f"autobusiness.{short}")


__all__ = ["get_logger"]
