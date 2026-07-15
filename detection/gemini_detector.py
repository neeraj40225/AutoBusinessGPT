"""Gemini-powered schema detector — the primary detection path.

An LLM reads column names *and* sample values and maps them to semantic roles,
handling vocabulary the regex heuristics cannot (``TransactionAmount`` ->
revenue, ``CustEmail`` -> email, domain-specific headers). It returns the exact
same :class:`SchemaMapping` the heuristic detector does, so the rest of the app
is agnostic to which ran.

If the call fails or no key is set, the caller (``detection.detector``) falls
back to heuristics. This module never silently returns a wrong answer — it
raises, and the fallback owns the recovery.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from core.config import BusinessType, Role, settings
from detection.schema import ColumnMapping, SchemaMapping
from utils import llm
from utils.helpers import LLMError
from utils.logger import get_logger

logger = get_logger(__name__)


_SYSTEM = (
    "You are a data schema analyst. You map spreadsheet columns to a fixed set "
    "of semantic roles for a business-analytics pipeline. You are precise and "
    "conservative: if a column does not clearly fit a role, you leave it "
    "unmapped rather than guessing. You only ever respond with valid JSON."
)


def _build_prompt(frame: pd.DataFrame) -> str:
    """Construct the detection prompt from headers + a small value sample."""
    roles_block = "\n".join(
        f"- {role}: {Role.DESCRIPTIONS[role]}" for role in Role.ALL
    )

    sample_n = min(settings.detection.sample_rows, len(frame))
    sample = frame.head(sample_n)

    col_block_lines: list[str] = []
    for col in frame.columns:
        values = sample[col].dropna().astype(str).head(5).tolist()
        dtype = str(frame[col].dtype)
        col_block_lines.append(
            f'  "{col}" (dtype={dtype}) sample={json.dumps(values, default=str)}'
        )
    col_block = "\n".join(col_block_lines)

    types_block = ", ".join(BusinessType.ALL)

    return f"""Analyze this dataset and return a JSON object mapping each column to a semantic role.

AVAILABLE ROLES (use these exact strings, or null if no role fits):
{roles_block}

BUSINESS TYPES (choose the single best fit): {types_block}

DATASET ({len(frame)} rows, {len(frame.columns)} columns):
{col_block}

Return ONLY this JSON structure, nothing else:
{{
  "business_type": "<one of the business types above>",
  "business_type_confidence": <0.0-1.0>,
  "business_type_reason": "<one short sentence>",
  "columns": [
    {{"column": "<exact column name>", "role": "<role string or null>", "confidence": <0.0-1.0>, "reason": "<short phrase>"}}
  ]
}}

Rules:
- Map at most ONE column to each role. If two columns could be revenue, pick the better one and leave the other null or map it to a different role (e.g. unit_price).
- Prefer a customer NAME column for customer_name and a customer ID column for customer_id; do not map both to the same role.
- Use null for the role of any column that does not clearly match (IDs you can't place, flags, free text, ship dates, postal codes).
- confidence reflects how certain you are; be honest about uncertainty."""


def _coerce_role(raw: Any) -> str | None:
    """Validate a role string from the model against the known vocabulary."""
    if raw in (None, "null", "", "(unmapped)"):
        return None
    if isinstance(raw, str) and raw in Role.ALL:
        return raw
    logger.warning("Model returned unknown role %r; treating as unmapped", raw)
    return None


def _coerce_business_type(raw: Any) -> str:
    if isinstance(raw, str) and raw in BusinessType.ALL:
        return raw
    return BusinessType.GENERIC


def detect(frame: pd.DataFrame) -> SchemaMapping:
    """Detect schema via Gemini. Raises LLMError on failure (caller falls back)."""
    if not llm.is_available():
        raise LLMError("Gemini not available for detection.")

    if len(frame.columns) > settings.detection.max_columns:
        raise LLMError(
            f"Dataset has {len(frame.columns)} columns, over the detection cap "
            f"of {settings.detection.max_columns}."
        )

    prompt = _build_prompt(frame)
    payload = llm.generate_json(prompt, system=_SYSTEM)

    if not isinstance(payload, dict) or "columns" not in payload:
        raise LLMError("Detection response missing 'columns'.")

    # index model output by column name for robust lookup
    by_col: dict[str, dict[str, Any]] = {}
    for entry in payload.get("columns", []):
        if isinstance(entry, dict) and "column" in entry:
            by_col[str(entry["column"])] = entry

    columns: list[ColumnMapping] = []
    seen_roles: set[str] = set()
    for col in frame.columns:
        entry = by_col.get(str(col), {})
        role = _coerce_role(entry.get("role"))
        # enforce one-column-per-role even if the model slipped
        if role in seen_roles:
            logger.warning("Duplicate role %s from model; unmapping %s", role, col)
            role = None
        if role:
            seen_roles.add(role)
        try:
            conf = float(entry.get("confidence", 0.6))
        except (TypeError, ValueError):
            conf = 0.6
        columns.append(ColumnMapping(
            column=str(col),
            role=role,
            confidence=max(0.0, min(1.0, conf)),
            reason=str(entry.get("reason", "")) or "Gemini mapping",
            dtype=str(frame[col].dtype),
            sample_values=frame[col].dropna().head(3).tolist(),
        ))

    biz = _coerce_business_type(payload.get("business_type"))
    try:
        biz_conf = float(payload.get("business_type_confidence", 0.6))
    except (TypeError, ValueError):
        biz_conf = 0.6

    logger.info(
        "Gemini detection: %d/%d columns mapped, type=%s",
        len(seen_roles), len(frame.columns), biz,
    )
    return SchemaMapping(
        columns=columns,
        business_type=biz,
        business_type_confidence=max(0.0, min(1.0, biz_conf)),
        business_type_reason=str(payload.get("business_type_reason", "")),
        source="gemini",
        n_rows=len(frame),
    )


__all__ = ["detect"]
