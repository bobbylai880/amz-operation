"""Read/write helpers for Doris using SQLAlchemy."""
from __future__ import annotations

import logging
import warnings
from itertools import islice
from typing import Iterable, Mapping, Sequence

import pandas as pd
from sqlalchemy import Column, MetaData, Table, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SAWarning
from sqlalchemy.sql.sqltypes import NullType


logger = logging.getLogger(__name__)

# Filter out Doris-specific schema reflection warnings emitted by SQLAlchemy's
# MySQL dialect when encountering Doris extensions such as DISTRIBUTED BY or
# PROPERTIES clauses. These warnings are expected and do not impact runtime
# behaviour, so we silence them globally to keep logs clean.
warnings.filterwarnings(
    "ignore",
    message="Unknown schema content",
    category=SAWarning,
)


def fetch_dataframe(engine: Engine, sql: str, params: Mapping[str, object] | None = None) -> pd.DataFrame:
    """Execute ``sql`` and return the result as a DataFrame."""

    stmt = text(sql)
    with engine.connect() as conn:
        result = conn.execute(stmt, params or {})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=result.keys())
        df = pd.DataFrame(rows, columns=result.keys())
    return df


def _normalise_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Convert SQL friendly structures (e.g., ``Timestamp``) to Python types."""

    normalised: list[dict[str, object]] = []
    for row in records:
        converted: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                converted[key] = value.to_pydatetime()
            elif pd.isna(value):  # type: ignore[arg-type]
                converted[key] = None
            else:
                converted[key] = value
        normalised.append(converted)
    return normalised


def _chunks(seq: Sequence[Mapping[str, object]], size: int) -> Iterable[list[Mapping[str, object]]]:
    """Yield ``size`` sized chunks from ``seq``."""

    it = iter(seq)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def replace_into(engine: Engine, table: str, df: pd.DataFrame, *, chunk_size: int = 500) -> int:
    """Execute a Doris ``UPSERT INTO`` using the provided ``DataFrame``."""

    if df.empty:
        return 0

    records = _normalise_records(df.to_dict(orient="records"))

    metadata = MetaData()
    try:
        table_obj = Table(table, metadata, autoload_with=engine)
    except Exception as exc:  # pragma: no cover - reflection fallback
        logger.debug(
            "replace_into falling back to synthetic metadata for %s due to %s",
            table,
            exc,
        )
        metadata = MetaData()
        columns = [Column(name, NullType()) for name in df.columns]
        table_obj = Table(table, metadata, *columns)

    return _execute_doris_upsert(engine, table_obj, df, records, chunk_size)


def _execute_doris_upsert(
    engine: Engine,
    table_obj: Table,
    df: pd.DataFrame,
    records: Sequence[Mapping[str, object]],
    chunk_size: int,
) -> int:
    preparer = engine.dialect.identifier_preparer
    column_tokens = [_format_identifier(preparer, name) for name in df.columns]
    value_tokens = [f":{name}" for name in df.columns]

    with engine.begin() as conn:
        version = _get_doris_version(conn)
        verb = "UPSERT INTO" if version and "doris" in version.lower() else "REPLACE INTO"
        if verb == "UPSERT INTO":
            logger.info(
                "replace_into using Doris UPSERT for %s (version=%s)",
                table_obj.fullname,
                version,
            )
        else:
            logger.info(
                "replace_into falling back to REPLACE INTO for %s (version=%s)",
                table_obj.fullname,
                version or "unknown",
            )
        statement = text(
            f"UPSERT INTO {_format_table(preparer, table_obj)} ({', '.join(column_tokens)}) VALUES ({', '.join(value_tokens)})"
        )
        affected = 0
        for chunk in _chunks(records, chunk_size):
            conn.execute(statement, chunk)
            affected += len(chunk)
    return affected


def _format_table(preparer, table_obj: Table) -> str:
    try:
        return preparer.format_table(table_obj)
    except Exception:  # pragma: no cover - fallback for custom dialects
        return table_obj.fullname


def _format_identifier(preparer, identifier: str) -> str:
    try:
        return preparer.quote(identifier)
    except AttributeError:  # pragma: no cover - compatibility shim
        return preparer.quote_identifier(identifier)
def _get_doris_version(conn: Connection) -> str | None:
    """Return Doris version, tolerant of MySQL-compatible layers."""

    try:
        comment_result = conn.execute(text("SELECT version_comment()"))
    except Exception:  # pragma: no cover - permissions or compatibility issues
        comment_result = None
    else:
        try:
            comment = comment_result.scalar()
        except Exception:  # pragma: no cover - scalar not supported
            comment = None
        if isinstance(comment, str):
            cleaned = comment.strip()
            if cleaned and "doris" in cleaned.lower():
                return cleaned

    try:
        version_result = conn.execute(text("SELECT version()"))
    except Exception as exc:  # pragma: no cover - network failures or permissions
        logger.warning("replace_into could not determine Doris version: %s", exc)
        return None
    try:
        version = version_result.scalar()
    except Exception:  # pragma: no cover - scalar not supported
        return None
    if isinstance(version, str):
        cleaned = version.strip()
        if cleaned.startswith("5.7.99"):
            return "Doris 2.x (mysql-compatible)"
        return cleaned
    return None


__all__ = ["fetch_dataframe", "replace_into"]
