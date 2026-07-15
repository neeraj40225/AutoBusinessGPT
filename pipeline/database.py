"""Dynamic SQLite database builder.

Unlike a fixed-schema app, the table is generated from whatever columns the
cleaned dataset has. Column names are sanitised to safe SQL identifiers and a
mapping back to the originals is kept so the copilot can translate between the
two.

The security model is lifted from hard-won practice: the copilot never touches
the read-write engine. It gets an engine opened with SQLite's ``mode=ro`` URI
flag *and* a ``sqlite3`` authorizer callback that rejects every non-read opcode.
A prompt-injected ``DROP TABLE`` fails at compile time, not because we grepped
for the word "drop".
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, event, text

from core.config import settings
from detection.schema import SchemaMapping
from utils.helpers import AutoBusinessError
from utils.logger import get_logger

logger = get_logger(__name__)

TABLE_NAME = "business_data"

_engine: Engine | None = None
_readonly_engine: Engine | None = None


def sanitize_identifier(name: str) -> str:
    """Turn an arbitrary column header into a safe SQL identifier."""
    ident = re.sub(r"\W+", "_", str(name).strip().lower()).strip("_")
    if not ident:
        ident = "col"
    if ident[0].isdigit():
        ident = f"c_{ident}"
    return ident


def build_column_map(frame: pd.DataFrame) -> dict[str, str]:
    """Map original column names -> unique sanitised identifiers."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in frame.columns:
        base = sanitize_identifier(col)
        ident = base
        i = 2
        while ident in used:
            ident = f"{base}_{i}"
            i += 1
        used.add(ident)
        mapping[col] = ident
    return mapping


def get_engine() -> Engine:
    """Process-wide read-write engine (lazy)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.paths.sqlite_url,
            future=True,
            connect_args={"check_same_thread": False},
        )
        logger.info("Read-write engine at %s", settings.paths.sqlite_file)
    return _engine


def _authorizer(action: int, arg1: str | None, arg2: str | None,
                dbname: str | None, source: str | None) -> int:
    """Permit reads, deny everything else. The real boundary for LLM SQL."""
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    if action in allowed:
        return sqlite3.SQLITE_OK
    logger.warning("authorizer denied action=%s arg1=%s", action, arg1)
    return sqlite3.SQLITE_DENY


def get_readonly_engine() -> Engine:
    """Engine that physically cannot write. Used only by the copilot."""
    global _readonly_engine
    if _readonly_engine is None:
        if not settings.paths.sqlite_file.exists():
            raise AutoBusinessError("Database not built yet.")
        _readonly_engine = create_engine(
            settings.paths.sqlite_url_readonly,
            future=True,
            connect_args={"check_same_thread": False, "uri": True},
        )

        @event.listens_for(_readonly_engine, "connect")
        def _harden(dbapi_conn: Any, _record: Any) -> None:
            dbapi_conn.set_authorizer(_authorizer)

        logger.info("Read-only engine initialised (authorizer active)")
    return _readonly_engine


def reset_engines() -> None:
    """Dispose cached engines. Call before rebuilding the DB file."""
    global _engine, _readonly_engine
    for eng in (_engine, _readonly_engine):
        if eng is not None:
            eng.dispose()
    _engine = None
    _readonly_engine = None


def build_database(frame: pd.DataFrame, mapping: SchemaMapping) -> dict[str, str]:
    """Write the cleaned dataset into SQLite as a single analysable table.

    Args:
        frame: The cleaned DataFrame.
        mapping: The confirmed schema (used to index role-bearing columns).

    Returns:
        The original->sanitised column name map, for the copilot's use.
    """
    reset_engines()
    if settings.paths.sqlite_file.exists():
        settings.paths.sqlite_file.unlink()

    col_map = build_column_map(frame)
    renamed = frame.rename(columns=col_map)

    engine = get_engine()
    renamed.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

    # index the columns that back common queries (dates, categories, ids)
    _create_indexes(engine, mapping, col_map)

    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()
    logger.info("Built %s: %d rows, %d columns", TABLE_NAME, n, len(col_map))
    return col_map


def _create_indexes(engine: Engine, mapping: SchemaMapping, col_map: dict[str, str]) -> None:
    """Add indexes on role-bearing columns that are commonly filtered/grouped."""
    from core.config import Role

    index_roles = (
        Role.ORDER_DATE, Role.CUSTOMER_ID, Role.PRODUCT, Role.CATEGORY,
        Role.REGION, Role.ORDER_ID,
    )
    with engine.begin() as conn:
        for role in index_roles:
            original = mapping.role_to_column(role)
            if not original or original not in col_map:
                continue
            ident = col_map[original]
            try:
                conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS idx_{ident} ON {TABLE_NAME}({ident})"
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Index on %s failed: %s", ident, exc)


def get_schema_ddl() -> str:
    """Return the CREATE TABLE statement, to ground the copilot's prompt."""
    with get_engine().connect() as conn:
        row = conn.execute(text(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=:t"
        ), {"t": TABLE_NAME}).fetchone()
    return row[0] if row else ""


def database_exists() -> bool:
    if not settings.paths.sqlite_file.exists():
        return False
    try:
        with get_engine().connect() as conn:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()
        return bool(n)
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "build_database", "get_engine", "get_readonly_engine", "reset_engines",
    "get_schema_ddl", "database_exists", "sanitize_identifier", "TABLE_NAME",
]
