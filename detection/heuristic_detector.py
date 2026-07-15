"""Heuristic schema detector — no LLM required.

This is the offline fallback for Gemini-first detection, and it must be good
enough to stand alone. It combines two signal sources per column:

1. **Name patterns** — regex against the column header (e.g. ``/sales|revenue|
   amount/`` -> revenue). Fast but fooled by ambiguous names.
2. **Value signals** — what the data actually looks like: does it parse as
   dates, is it numeric, how unique is it (an ID is ~100% unique; a category is
   not), does it look like an email/phone.

Neither alone is reliable; the combination is. Confidence reflects how many
independent signals agreed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import BusinessType, Role, settings
from detection.schema import ColumnMapping, SchemaMapping
from utils.helpers import coerce_numeric, is_probably_datetime
from utils.logger import get_logger

logger = get_logger(__name__)


# Name-pattern rules: (role, compiled regex, base confidence when name matches).
# Order matters — more specific patterns first so "customer_id" beats "customer".
_NAME_RULES: list[tuple[str, re.Pattern[str], float]] = [
    (Role.CUSTOMER_ID, re.compile(r"\b(customer|client|cust|member|user|account|guest|subscriber|patient)[\s_-]*(id|no|number|code|key)\b|^cust[\s_-]*id$", re.I), 0.85),
    (Role.ORDER_ID, re.compile(r"\b(order|invoice|transaction|txn|receipt|bill)[\s_-]*(id|no|number|code)\b|\b(order|invoice)id\b", re.I), 0.85),
    (Role.PRODUCT, re.compile(r"(product|item|article)[\s_-]*name|^(product|item)$|\bsku\b", re.I), 0.8),
    (Role.PRODUCT, re.compile(r"(product|item|article|goods)[\s_-]*(id|code|no)", re.I), 0.55),
    (Role.SUB_CATEGORY, re.compile(r"\bsub[\s_-]*categ", re.I), 0.9),
    (Role.CATEGORY, re.compile(r"\bcateg|department|dept|product[\s_-]*type|segment[\s_-]*type\b", re.I), 0.75),
    (Role.CUSTOMER_NAME, re.compile(r"(customer|client|account|member)[\s_-]*name", re.I), 0.8),
    (Role.ORDER_DATE, re.compile(r"(order|invoice|transaction|txn|purchase|sale|booking|admission)[\s_-]*date|date|timestamp|datetime", re.I), 0.65),
    (Role.REVENUE, re.compile(r"revenue|sales|amount|turnover|gross|net[\s_-]*sales|grand[\s_-]*total|line[\s_-]*total|\btotal\b|\bmrr\b|\barr\b", re.I), 0.7),
    (Role.UNIT_PRICE, re.compile(r"unit[\s_-]*price|\bprice\b|\brate\b|\bmrp\b|list[\s_-]*price|selling[\s_-]*price", re.I), 0.7),
    (Role.QUANTITY, re.compile(r"quantity|\bqty\b|\bunits\b|\bcount\b|no[\s_-]*of[\s_-]*items|pieces", re.I), 0.75),
    (Role.COST, re.compile(r"\bcost\b|\bcogs\b|purchase[\s_-]*price|buy[\s_-]*price|expense", re.I), 0.7),
    (Role.PROFIT, re.compile(r"\b(profit|margin|earnings|gain)\b", re.I), 0.75),
    (Role.DISCOUNT, re.compile(r"\b(discount|markdown|rebate|off)\b", re.I), 0.75),
    (Role.STOCK, re.compile(r"\b(stock|inventory|on[\s_-]*hand|units[\s_-]*available|balance[\s_-]*qty)\b", re.I), 0.75),
    (Role.SUPPLIER, re.compile(r"\b(supplier|vendor|manufacturer|distributor)\b", re.I), 0.8),
    (Role.EMPLOYEE, re.compile(r"\b(employee|salesperson|sales[\s_-]*rep|agent|staff|cashier)\b", re.I), 0.75),
    (Role.STORE, re.compile(r"\b(store|branch|outlet|shop|location|warehouse)\b", re.I), 0.7),
    (Role.REGION, re.compile(r"\b(region|zone|territory|area)\b", re.I), 0.8),
    (Role.COUNTRY, re.compile(r"\bcountry\b", re.I), 0.9),
    (Role.STATE, re.compile(r"\b(state|province|county)\b", re.I), 0.8),
    (Role.CITY, re.compile(r"\b(city|town|municipality)\b", re.I), 0.8),
    (Role.EMAIL, re.compile(r"e[\s_-]*mail|\bmail\b", re.I), 0.7),
    (Role.PHONE, re.compile(r"phone|mobile|\bcontact\b|\btel\b|\bcell\b", re.I), 0.7),
    (Role.TARGET, re.compile(r"target|label|churn|approved|default|outcome|\bclass\b|is[\s_-]*\w+|flag", re.I), 0.45),
]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^[\d\s()+.-]{7,}$")


@dataclass
class _ColumnStats:
    """Value-based signals computed once per column."""

    dtype: str
    n_unique: int
    n_total: int
    null_frac: float
    unique_frac: float
    is_numeric: bool
    is_datetime: bool
    numeric_min: float
    numeric_mean: float
    frac_email: float
    frac_phone: float
    frac_between_0_1: float


def _column_stats(series: pd.Series) -> _ColumnStats:
    n_total = len(series)
    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    numeric = coerce_numeric(series)
    is_numeric = numeric.notna().mean() >= 0.8 if n_total else False
    is_dt = is_probably_datetime(series)

    num_valid = numeric.dropna()
    frac_0_1 = float(((num_valid >= 0) & (num_valid <= 1)).mean()) if len(num_valid) else 0.0

    str_sample = non_null.astype(str).head(200)
    frac_email = float(str_sample.apply(lambda s: bool(_EMAIL_RE.match(s))).mean()) if len(str_sample) else 0.0
    frac_phone = float(str_sample.apply(lambda s: bool(_PHONE_RE.match(s)) and any(ch.isdigit() for ch in s)).mean()) if len(str_sample) else 0.0

    return _ColumnStats(
        dtype=str(series.dtype),
        n_unique=n_unique,
        n_total=n_total,
        null_frac=float(series.isna().mean()),
        unique_frac=float(n_unique / n_total) if n_total else 0.0,
        is_numeric=is_numeric,
        is_datetime=is_dt,
        numeric_min=float(num_valid.min()) if len(num_valid) else 0.0,
        numeric_mean=float(num_valid.mean()) if len(num_valid) else 0.0,
        frac_email=frac_email,
        frac_phone=frac_phone,
        frac_between_0_1=frac_0_1,
    )


def _value_adjust(role: str, stats: _ColumnStats) -> float:
    """Return a confidence delta from value signals for a proposed role.

    Positive when the data supports the name-based guess, negative when it
    contradicts it. This is what stops "Amount" (a text status column) from
    being mapped to revenue.
    """
    delta = 0.0

    if role in (Role.CUSTOMER_ID, Role.ORDER_ID):
        delta += 0.10 if stats.unique_frac > 0.5 else -0.15
    if role in (Role.REVENUE, Role.UNIT_PRICE, Role.COST, Role.PROFIT, Role.QUANTITY, Role.STOCK):
        delta += 0.15 if stats.is_numeric else -0.35
    if role == Role.ORDER_DATE:
        delta += 0.25 if stats.is_datetime else -0.4
    if role == Role.DISCOUNT:
        delta += 0.15 if stats.frac_between_0_1 > 0.8 else 0.0
    if role in (Role.CATEGORY, Role.SUB_CATEGORY, Role.REGION, Role.STATE, Role.COUNTRY, Role.CITY):
        # low-cardinality relative to rows
        delta += 0.1 if stats.unique_frac < 0.5 else -0.1

    if role == Role.EMAIL:
        delta += 0.3 if stats.frac_email > 0.5 else -0.4
    if role == Role.PHONE:
        delta += 0.2 if stats.frac_phone > 0.5 else -0.2
    if role == Role.TARGET:
        # a target is usually low-cardinality
        delta += 0.15 if stats.n_unique <= 10 else -0.2

    return delta


def _detect_business_type(mapped: set[str], columns: list[str]) -> tuple[str, float, str]:
    """Very rough business-type guess from which roles are present.

    Contextual only — never gates ML — so a coarse heuristic is acceptable.
    """
    joined = " ".join(columns).lower()
    signals: list[tuple[str, float, str]] = []

    if re.search(r"\b(patient|admission|diagnos|hospital|ward|doctor)\b", joined):
        signals.append((BusinessType.HOSPITAL, 0.7, "clinical column names"))
    if re.search(r"\b(room|guest|check[\s_-]*in|check[\s_-]*out|booking|reservation)\b", joined):
        signals.append((BusinessType.HOTEL, 0.65, "booking/room column names"))
    if re.search(r"\b(account|balance|loan|credit|debit|transaction|branch|ifsc)\b", joined):
        signals.append((BusinessType.BANK, 0.55, "banking column names"))
    if re.search(r"\b(table|menu|dish|cuisine|order[\s_-]*type|dine)\b", joined):
        signals.append((BusinessType.RESTAURANT, 0.6, "menu/dining column names"))
    if {Role.STOCK}.issubset(mapped) and Role.REVENUE not in mapped:
        signals.append((BusinessType.INVENTORY, 0.6, "stock without revenue"))
    if {Role.PRODUCT, Role.REVENUE, Role.CUSTOMER_ID}.issubset(mapped):
        signals.append((BusinessType.RETAIL, 0.6, "product + revenue + customer"))
    if {Role.REVENUE, Role.ORDER_DATE}.issubset(mapped) and Role.PRODUCT not in mapped:
        signals.append((BusinessType.SALES, 0.5, "revenue over time"))

    if not signals:
        return BusinessType.GENERIC, 0.4, "no strong type signal"
    signals.sort(key=lambda s: s[1], reverse=True)
    return signals[0]


def detect(frame: pd.DataFrame) -> SchemaMapping:
    """Detect a schema mapping using name + value heuristics only."""
    columns: list[ColumnMapping] = []
    taken: set[str] = set()  # roles already assigned (one column per role)

    # Score every (column, matching-rule) pair, then greedily assign best first.
    candidates: list[tuple[float, str, str, str, _ColumnStats]] = []
    stats_cache: dict[str, _ColumnStats] = {}

    for col in frame.columns:
        stats = _column_stats(frame[col])
        stats_cache[col] = stats
        for role, pattern, base in _NAME_RULES:
            if pattern.search(str(col)):
                conf = min(0.98, max(0.05, base + _value_adjust(role, stats)))
                reason = f"name matches ‘{role}’ pattern"
                if stats.is_datetime and role == Role.ORDER_DATE:
                    reason = "values parse as dates"
                elif stats.is_numeric and role in (Role.REVENUE, Role.QUANTITY):
                    reason = f"numeric values, name matches ‘{role}’"
                candidates.append((conf, col, role, reason, stats))

    candidates.sort(key=lambda c: c[0], reverse=True)
    assigned_cols: dict[str, tuple[str, float, str]] = {}
    for conf, col, role, reason, _stats in candidates:
        if col in assigned_cols or role in taken:
            continue
        assigned_cols[col] = (role, conf, reason)
        taken.add(role)

    # Build ColumnMapping for every column (mapped or not).
    for col in frame.columns:
        stats = stats_cache[col]
        sample = frame[col].dropna().head(3).tolist()
        if col in assigned_cols:
            role, conf, reason = assigned_cols[col]
            columns.append(ColumnMapping(
                column=col, role=role, confidence=conf, reason=reason,
                dtype=stats.dtype, sample_values=sample,
            ))
        else:
            columns.append(ColumnMapping(
                column=col, role=None, confidence=0.0,
                reason="no role pattern matched", dtype=stats.dtype,
                sample_values=sample,
            ))

    biz, biz_conf, biz_reason = _detect_business_type(taken, list(frame.columns))
    logger.info(
        "Heuristic detection: %d/%d columns mapped, type=%s",
        len(taken), len(frame.columns), biz,
    )
    return SchemaMapping(
        columns=columns,
        business_type=biz,
        business_type_confidence=biz_conf,
        business_type_reason=biz_reason,
        source="heuristic",
        n_rows=len(frame),
    )


__all__ = ["detect"]
