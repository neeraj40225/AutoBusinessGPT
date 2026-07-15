"""Shared exceptions, formatters, and small dataframe utilities."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class AutoBusinessError(Exception):
    """Base class for all application errors."""


class DataNotFoundError(AutoBusinessError):
    """Expected data was missing."""


class DetectionError(AutoBusinessError):
    """Schema detection failed irrecoverably."""


class ModelNotTrainedError(AutoBusinessError):
    """A model artifact was requested before training."""


class UnsafeSQLError(AutoBusinessError):
    """LLM-generated SQL failed the safety boundary."""


class LLMError(AutoBusinessError):
    """The language model call failed."""


class UnsupportedFileError(AutoBusinessError):
    """Uploaded file could not be parsed."""


# --------------------------------------------------------------------------- #
# Formatters
# --------------------------------------------------------------------------- #
def format_currency(value: float, symbol: str = "$", decimals: int = 0) -> str:
    try:
        return f"{symbol}{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return f"{symbol}0"


def format_compact(value: float, symbol: str = "$") -> str:
    """Human-readable large numbers: 1.2K, 3.4M, 5.6B."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{symbol}0"
    sign = "-" if v < 0 else ""
    v = abs(v)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= threshold:
            return f"{sign}{symbol}{v / threshold:.1f}{suffix}"
    return f"{sign}{symbol}{v:,.0f}"


def format_percent(value: float, decimals: int = 1) -> str:
    try:
        return f"{value:.{decimals}f}%"
    except (TypeError, ValueError):
        return "0%"


def format_number(value: float) -> str:
    try:
        return f"{value:,.0f}"
    except (TypeError, ValueError):
        return "0"


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def pct_change(current: float, previous: float) -> float:
    if not previous:
        return 0.0
    return (current - previous) / abs(previous) * 100.0


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion that strips currency symbols and commas."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = (
        series.astype(str)
        .str.replace(r"[,$₹€£%\s]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def is_probably_datetime(series: pd.Series, sample: int = 50) -> bool:
    """Heuristic: does a sample of this column parse as dates?"""
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample_vals = non_null.head(sample)
    parsed = pd.to_datetime(sample_vals, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.8


def ensure_columns(frame: pd.DataFrame, required: list[str], context: str = "") -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataNotFoundError(
            f"Missing required columns {missing}"
            + (f" for {context}" if context else "")
        )


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator logging wall-clock time of a call at DEBUG."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.debug("%s took %.3fs", func.__name__, time.perf_counter() - start)

    return wrapper


__all__ = [
    "AutoBusinessError", "DataNotFoundError", "DetectionError",
    "ModelNotTrainedError", "UnsafeSQLError", "LLMError", "UnsupportedFileError",
    "format_currency", "format_compact", "format_percent", "format_number",
    "safe_divide", "pct_change", "coerce_numeric", "is_probably_datetime",
    "ensure_columns", "timed",
]
