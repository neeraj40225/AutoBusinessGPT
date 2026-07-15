"""Automatic data cleaning — role-aware, with a change log.

Cleaning is deterministic and reversible in spirit: every transformation is
recorded in a :class:`CleaningLog` so the UI can show the user exactly what was
done rather than silently mutating their data. Role information drives the
type-correct fixes (coerce the revenue column to numeric, parse the date column
as dates) that a schema-blind cleaner cannot do safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.config import Role
from detection.schema import SchemaMapping
from utils.helpers import coerce_numeric
from utils.logger import get_logger

logger = get_logger(__name__)

# roles whose columns should be numeric
_NUMERIC_ROLES = {
    Role.REVENUE, Role.QUANTITY, Role.UNIT_PRICE, Role.COST,
    Role.PROFIT, Role.DISCOUNT, Role.STOCK,
}
# roles whose columns are categorical text worth standardising
_TEXT_ROLES = {
    Role.CATEGORY, Role.SUB_CATEGORY, Role.REGION, Role.COUNTRY,
    Role.STATE, Role.CITY, Role.STORE, Role.SUPPLIER, Role.PRODUCT,
}


@dataclass
class CleaningLog:
    """Record of every cleaning action taken."""

    actions: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0

    def add(self, action: str) -> None:
        self.actions.append(action)
        logger.info("clean: %s", action)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    def summary(self) -> dict[str, Any]:
        return {
            "actions": len(self.actions),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": self.rows_removed,
        }


def clean(frame: pd.DataFrame, mapping: SchemaMapping) -> tuple[pd.DataFrame, CleaningLog]:
    """Return a cleaned copy of the frame and a log of what changed.

    Steps: drop duplicate rows, coerce numeric-role columns to numbers, parse
    the date column, standardise categorical text, fill missing values by a
    type-appropriate rule, and remove impossible values (negative revenue/qty).
    """
    df = frame.copy()
    log = CleaningLog(rows_before=len(df))

    # 1. duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        log.add(f"Removed {before - len(df)} duplicate rows")

    # 2. numeric coercion for numeric roles
    for cm in mapping.columns:
        if cm.role in _NUMERIC_ROLES and cm.column in df.columns:
            original_non_null = df[cm.column].notna().sum()
            df[cm.column] = coerce_numeric(df[cm.column])
            new_non_null = df[cm.column].notna().sum()
            lost = original_non_null - new_non_null
            if lost > 0:
                log.add(f"Coerced '{cm.column}' to numeric ({lost} values became NaN)")
            elif not pd.api.types.is_numeric_dtype(frame[cm.column]):
                log.add(f"Coerced '{cm.column}' to numeric")

    # 3. date parsing
    date_col = mapping.role_to_column(Role.ORDER_DATE)
    if date_col and date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        if parsed.notna().any():
            df[date_col] = parsed
            log.add(f"Parsed '{date_col}' as datetime")

    # 4. standardise categorical text
    for cm in mapping.columns:
        if cm.role in _TEXT_ROLES and cm.column in df.columns:
            if df[cm.column].dtype == object:
                df[cm.column] = (
                    df[cm.column].astype(str).str.strip()
                    .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                )
                log.add(f"Standardised text in '{cm.column}'")

    # 5. remove impossible values on key numeric roles
    for role, label in ((Role.REVENUE, "revenue"), (Role.QUANTITY, "quantity"),
                        (Role.UNIT_PRICE, "price")):
        col = mapping.role_to_column(role)
        if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            neg = int((df[col] < 0).sum())
            if neg:
                df = df[df[col] >= 0].reset_index(drop=True)
                log.add(f"Removed {neg} rows with negative {label}")

    # 6. fill remaining missing values, type-appropriately
    _fill_missing(df, mapping, log)

    log.rows_after = len(df)
    logger.info("Cleaning complete: %d -> %d rows, %d actions",
                log.rows_before, log.rows_after, len(log.actions))
    return df, log


def _fill_missing(df: pd.DataFrame, mapping: SchemaMapping, log: CleaningLog) -> None:
    """Fill NaNs: median for numeric, mode/'Unknown' for categorical."""
    numeric_cols = {cm.column for cm in mapping.columns if cm.role in _NUMERIC_ROLES}
    filled_numeric, filled_text = 0, 0

    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if col in numeric_cols and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            filled_numeric += 1
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            filled_numeric += 1
        elif not pd.api.types.is_datetime64_any_dtype(df[col]):
            mode = df[col].mode()
            fill_val = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            filled_text += 1

    if filled_numeric:
        log.add(f"Filled missing values in {filled_numeric} numeric column(s) with median")
    if filled_text:
        log.add(f"Filled missing values in {filled_text} categorical column(s) with mode")


__all__ = ["clean", "CleaningLog"]
