"""Data quality analysis — a role-aware quality report.

Generic quality checks (missing values, duplicates, type errors) run on any
dataset. Role-aware checks (negative revenue, negative quantity, future dates,
discounts outside 0–1) only fire once the schema tells us which column plays
which role — that mapping is what turns a generic profiler into a business
validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from core.config import Role
from detection.schema import SchemaMapping
from utils.helpers import coerce_numeric
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityIssue:
    """One detected data-quality problem."""

    column: str
    kind: str          # "missing", "duplicate", "negative", "future_date", ...
    severity: str      # "low" | "medium" | "high"
    count: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "issue": self.kind,
            "severity": self.severity,
            "rows_affected": self.count,
            "detail": self.detail,
        }


@dataclass
class QualityReport:
    """The full quality assessment for a dataset."""

    issues: list[QualityIssue] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    n_duplicate_rows: int = 0
    total_missing_cells: int = 0
    score: float = 100.0

    def by_severity(self, severity: str) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == severity]

    @property
    def grade(self) -> str:
        if self.score >= 90:
            return "Excellent"
        if self.score >= 75:
            return "Good"
        if self.score >= 60:
            return "Fair"
        return "Poor"

    def summary(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "grade": self.grade,
            "issues": len(self.issues),
            "high": len(self.by_severity("high")),
            "duplicate_rows": self.n_duplicate_rows,
            "missing_cells": self.total_missing_cells,
        }


def analyze(frame: pd.DataFrame, mapping: SchemaMapping) -> QualityReport:
    """Produce a quality report, using role mappings for business checks."""
    report = QualityReport(n_rows=len(frame), n_cols=frame.shape[1])
    n = len(frame)

    # -- missing values (generic) -------------------------------------------
    missing = frame.isna().sum()
    report.total_missing_cells = int(missing.sum())
    for col, miss in missing.items():
        if miss == 0:
            continue
        frac = miss / n
        sev = "high" if frac > 0.4 else "medium" if frac > 0.1 else "low"
        report.issues.append(QualityIssue(
            column=str(col), kind="missing", severity=sev, count=int(miss),
            detail=f"{frac:.1%} of values missing",
        ))

    # -- duplicate rows (generic) -------------------------------------------
    dup_mask = frame.duplicated()
    report.n_duplicate_rows = int(dup_mask.sum())
    if report.n_duplicate_rows:
        report.issues.append(QualityIssue(
            column="(all)", kind="duplicate", severity="medium",
            count=report.n_duplicate_rows,
            detail=f"{report.n_duplicate_rows} fully duplicated rows",
        ))

    # -- role-aware checks ---------------------------------------------------
    _check_non_negative(frame, mapping, report, Role.REVENUE, "revenue")
    _check_non_negative(frame, mapping, report, Role.QUANTITY, "quantity")
    _check_non_negative(frame, mapping, report, Role.UNIT_PRICE, "unit price")
    _check_non_negative(frame, mapping, report, Role.STOCK, "stock")
    _check_future_dates(frame, mapping, report)
    _check_discount_range(frame, mapping, report)
    _check_outliers(frame, mapping, report)

    report.score = _score(report, n)
    logger.info("Quality: score=%.1f grade=%s issues=%d", report.score, report.grade, len(report.issues))
    return report


def _check_non_negative(frame, mapping, report, role, label) -> None:
    col = mapping.role_to_column(role)
    if not col or col not in frame.columns:
        return
    numeric = coerce_numeric(frame[col])
    neg = int((numeric < 0).sum())
    if neg:
        report.issues.append(QualityIssue(
            column=col, kind="negative", severity="high", count=neg,
            detail=f"{neg} rows with negative {label} (usually invalid)",
        ))


def _check_future_dates(frame, mapping, report) -> None:
    col = mapping.role_to_column(Role.ORDER_DATE)
    if not col or col not in frame.columns:
        return
    dates = pd.to_datetime(frame[col], errors="coerce", format="mixed")
    future = int((dates > datetime.now()).sum())
    if future:
        report.issues.append(QualityIssue(
            column=col, kind="future_date", severity="medium", count=future,
            detail=f"{future} rows dated in the future",
        ))
    unparseable = int(dates.isna().sum() - frame[col].isna().sum())
    if unparseable > 0:
        report.issues.append(QualityIssue(
            column=col, kind="bad_date", severity="medium", count=unparseable,
            detail=f"{unparseable} values could not be parsed as dates",
        ))


def _check_discount_range(frame, mapping, report) -> None:
    col = mapping.role_to_column(Role.DISCOUNT)
    if not col or col not in frame.columns:
        return
    numeric = coerce_numeric(frame[col]).dropna()
    if numeric.empty:
        return
    # a fraction discount should be 0–1; an amount could be larger, so only
    # flag clearly impossible values (negative or absurd fractions)
    bad = int(((numeric < 0) | ((numeric > 1) & (numeric <= 100) & (numeric.max() <= 1.5))).sum())
    if (numeric < 0).any():
        report.issues.append(QualityIssue(
            column=col, kind="bad_discount", severity="medium",
            count=int((numeric < 0).sum()),
            detail="negative discount values",
        ))


def _check_outliers(frame, mapping, report) -> None:
    """Flag extreme outliers in the revenue column via IQR."""
    col = mapping.role_to_column(Role.REVENUE)
    if not col or col not in frame.columns:
        return
    numeric = coerce_numeric(frame[col]).dropna()
    if len(numeric) < 20:
        return
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return
    fence = q3 + 3 * iqr
    extreme = int((numeric > fence).sum())
    if extreme and extreme < len(numeric) * 0.05:  # only if genuinely rare
        report.issues.append(QualityIssue(
            column=col, kind="outlier", severity="low", count=extreme,
            detail=f"{extreme} extreme high outliers (>3×IQR above Q3)",
        ))


def _score(report: QualityReport, n_rows: int) -> float:
    """Deduct from 100 by weighted issue severity, capped per category."""
    penalty = 0.0
    weights = {"high": 12.0, "medium": 5.0, "low": 1.5}
    for issue in report.issues:
        # scale by how much of the data is affected, but cap each issue
        frac = min(1.0, issue.count / n_rows) if n_rows else 0.0
        penalty += weights.get(issue.severity, 1.0) * (0.3 + 0.7 * frac)
    return max(0.0, 100.0 - penalty)


__all__ = ["QualityReport", "QualityIssue", "analyze"]
