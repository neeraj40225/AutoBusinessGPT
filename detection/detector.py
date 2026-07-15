"""Unified schema detection: Gemini-first, heuristic fallback.

This is the only detection entry point the rest of the app calls. It encodes
the decision made at design time — Gemini drives, heuristics catch it — so that
a missing key or a failed API call degrades to a working offline path instead
of a crash.

The returned :class:`SchemaMapping` carries a ``source`` field ("gemini" or
"heuristic") so the UI can tell the user which ran and why.
"""

from __future__ import annotations

import pandas as pd

from core.config import settings
from detection import gemini_detector, heuristic_detector
from detection.schema import SchemaMapping
from utils import llm
from utils.helpers import LLMError
from utils.logger import get_logger

logger = get_logger(__name__)


def detect_schema(frame: pd.DataFrame) -> SchemaMapping:
    """Detect the dataset schema, preferring Gemini and falling back cleanly.

    Strategy is controlled by ``settings.detection.strategy``:
      * ``gemini_first`` (default): try Gemini, fall back to heuristics on any
        failure or missing key.
      * ``heuristic_only``: never call the LLM.
      * ``gemini_only``: raise if Gemini is unavailable (no fallback).

    Args:
        frame: The uploaded dataset (already parsed to a DataFrame).

    Returns:
        A populated :class:`SchemaMapping`.
    """
    strategy = settings.detection.strategy

    if strategy == "heuristic_only":
        return heuristic_detector.detect(frame)

    if strategy == "gemini_only":
        # explicit no-fallback mode
        return gemini_detector.detect(frame)

    # default: gemini_first
    if llm.is_available():
        try:
            mapping = gemini_detector.detect(frame)
            logger.info("Detection via Gemini succeeded.")
            return mapping
        except LLMError as exc:
            logger.warning("Gemini detection failed (%s); falling back to heuristics.", exc)
        except Exception as exc:  # noqa: BLE001 - never let detection crash the app
            logger.exception("Unexpected detection error; falling back to heuristics.")
    else:
        logger.info("No Gemini key; using heuristic detection.")

    mapping = heuristic_detector.detect(frame)
    return mapping


__all__ = ["detect_schema"]
