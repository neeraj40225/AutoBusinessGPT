"""Dataset loading — turn an uploaded CSV/Excel file into a DataFrame.

Handles the file-format zoo (csv, tsv, xlsx, xls) and the encoding/delimiter
guesswork that real-world business exports require, then does light structural
validation before the schema detector ever sees the data.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import settings
from utils.helpers import UnsupportedFileError
from utils.logger import get_logger

logger = get_logger(__name__)

_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


def _read_csv_robust(buffer: bytes) -> pd.DataFrame:
    """Try several encodings and let pandas sniff the delimiter."""
    last_err: Exception | None = None
    for enc in _CSV_ENCODINGS:
        try:
            return pd.read_csv(
                io.BytesIO(buffer),
                encoding=enc,
                sep=None,           # sniff , ; \t |
                engine="python",
            )
        except Exception as exc:  # noqa: BLE001 - trying the next encoding
            last_err = exc
    raise UnsupportedFileError(f"Could not parse CSV: {last_err}")


def load_dataframe(file: Any, filename: str | None = None) -> pd.DataFrame:
    """Load an uploaded file object or path into a DataFrame.

    Args:
        file: A Streamlit UploadedFile, a bytes buffer, or a filesystem path.
        filename: Name hint used to pick the parser when ``file`` is raw bytes.

    Returns:
        A DataFrame with stripped column names and blank columns/rows dropped.

    Raises:
        UnsupportedFileError: If the format is unrecognised or parsing fails.
    """
    name = (filename or getattr(file, "name", "") or "").lower()

    if hasattr(file, "read"):
        raw = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
    elif isinstance(file, (bytes, bytearray)):
        raw = bytes(file)
    else:  # a path
        raw = Path(file).read_bytes()
        name = name or str(file).lower()

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        try:
            frame = pd.read_excel(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedFileError(f"Could not parse Excel file: {exc}") from exc
    elif name.endswith((".csv", ".tsv", ".txt")) or not name:
        frame = _read_csv_robust(raw)
    else:
        raise UnsupportedFileError(
            f"Unsupported file type: {name!r}. Upload a CSV or Excel file."
        )

    frame = _tidy(frame)
    _validate_structure(frame)
    logger.info("Loaded dataset: %d rows x %d cols from %s", len(frame), frame.shape[1], name or "buffer")
    return frame


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip header whitespace, drop fully-empty rows and columns."""
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    # drop pandas' unnamed index columns from Excel/CSV round-trips
    junk = [c for c in frame.columns if c.lower().startswith("unnamed:")]
    if junk:
        frame = frame.drop(columns=junk)
    frame = frame.reset_index(drop=True)
    return frame


def _validate_structure(frame: pd.DataFrame) -> None:
    """Reject datasets that are structurally unusable for analysis."""
    if frame.empty:
        raise UnsupportedFileError("The uploaded file has no data rows.")
    if frame.shape[1] < 2:
        raise UnsupportedFileError(
            "The dataset has only one column — nothing to analyze. "
            "Check the delimiter or file format."
        )
    if len(frame) < 5:
        raise UnsupportedFileError(
            f"Only {len(frame)} rows found. At least a handful of rows are "
            "needed for meaningful analysis."
        )
    # duplicate column names break SQL and role mapping
    dupes = frame.columns[frame.columns.duplicated()].tolist()
    if dupes:
        raise UnsupportedFileError(f"Duplicate column names found: {dupes}. Rename them and re-upload.")


def save_upload(raw: bytes, filename: str) -> Path:
    """Persist an uploaded file to the uploads dir; return the path."""
    safe = Path(filename).name
    dest = settings.paths.uploads / safe
    dest.write_bytes(raw)
    return dest


__all__ = ["load_dataframe", "save_upload"]
