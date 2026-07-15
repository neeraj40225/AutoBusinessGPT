"""Business Copilot — natural-language questions answered against the database.

Gemini writes SQL grounded in the actual table schema; the query runs through
the read-only engine with the authorizer callback (see pipeline.database), so a
prompt-injected mutation cannot execute. Results are summarised back in plain
language. The column-name map lets the copilot show users friendly names even
though the table uses sanitised identifiers.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import text

from core.config import settings
from pipeline.database import TABLE_NAME, get_readonly_engine, get_schema_ddl
from utils import llm
from utils.helpers import LLMError, UnsafeSQLError
from utils.logger import get_logger

logger = get_logger(__name__)

# Defence in depth: even before the authorizer, reject obviously non-SELECT text.
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|"
                        r"truncate|attach|detach|pragma|vacuum)\b", re.I)


@dataclass
class CopilotResponse:
    question: str
    sql: str
    frame: pd.DataFrame
    answer: str
    row_count: int
    elapsed_ms: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _schema_hint(column_map: dict[str, str]) -> str:
    """Build a friendly-name -> sql-name reference for the prompt."""
    ddl = get_schema_ddl()
    lines = [f"Table: {TABLE_NAME}", ddl, "", "Column meanings (original -> sql name):"]
    for original, ident in column_map.items():
        lines.append(f"- {original!r} -> {ident}")
    return "\n".join(lines)


def _generate_sql(question: str, column_map: dict[str, str]) -> str:
    schema = _schema_hint(column_map)
    prompt = (
        f"You write SQLite SELECT queries for one table.\n\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"Rules:\n"
        f"- Return ONLY a single SELECT query, no explanation, no markdown.\n"
        f"- Use the sql column names (right side of the arrows), not the originals.\n"
        f"- Always add LIMIT {settings.sql.max_rows} unless the query is an aggregate returning few rows.\n"
        f"- Never write anything other than SELECT."
    )
    raw = llm.generate(prompt, temperature=0.0)
    return _clean_sql(raw)


def _clean_sql(raw: str) -> str:
    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    # keep only the first statement
    sql = sql.split(";")[0].strip()
    return sql


def _validate(sql: str) -> None:
    if not sql.lower().lstrip().startswith(("select", "with")):
        raise UnsafeSQLError("Only SELECT queries are allowed.")
    if _FORBIDDEN.search(sql):
        raise UnsafeSQLError("Query contains a forbidden keyword.")


def _execute(sql: str) -> tuple[pd.DataFrame, float]:
    _validate(sql)
    engine = get_readonly_engine()  # authorizer is the real boundary
    start = time.perf_counter()
    with engine.connect() as conn:
        frame = pd.read_sql(text(sql), conn)
    elapsed = (time.perf_counter() - start) * 1000
    if len(frame) > settings.sql.max_rows:
        frame = frame.head(settings.sql.max_rows)
    return frame, elapsed


def _summarise(question: str, sql: str, frame: pd.DataFrame) -> str:
    if frame.empty:
        return "The query ran but returned no rows."
    if not llm.is_available():
        return _template_summary(frame)
    preview = frame.head(20).to_markdown(index=False)
    prompt = (
        f"Question: {question}\n\nQuery result (first rows):\n{preview}\n\n"
        f"Answer the question in 1-3 sentences using these results. "
        f"Cite specific numbers. Do not invent data."
    )
    try:
        return llm.generate(prompt, temperature=0.3)
    except LLMError:
        return _template_summary(frame)


def _template_summary(frame: pd.DataFrame) -> str:
    return f"Returned {len(frame)} rows across {frame.shape[1]} columns. See the table below."


def ask(question: str, column_map: dict[str, str]) -> CopilotResponse:
    """Answer a natural-language question against the database."""
    if not llm.is_available():
        return CopilotResponse(
            question=question, sql="", frame=pd.DataFrame(), answer="",
            row_count=0, elapsed_ms=0.0,
            error="The Copilot needs a Gemini API key. Add it in Settings.",
        )
    try:
        sql = _generate_sql(question, column_map)
        frame, elapsed = _execute(sql)
        answer = _summarise(question, sql, frame)
        logger.info("Copilot answered (%d rows, %.0fms)", len(frame), elapsed)
        return CopilotResponse(question, sql, frame, answer, len(frame), elapsed)
    except (UnsafeSQLError, LLMError) as exc:
        return CopilotResponse(question, "", pd.DataFrame(), "", 0, 0.0, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Copilot failed")
        return CopilotResponse(question, "", pd.DataFrame(), "", 0, 0.0,
                               error=f"Query failed: {exc}")


__all__ = ["ask", "CopilotResponse"]
