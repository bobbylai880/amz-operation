"""Read/write helpers for Doris using SQLAlchemy."""
from __future__ import annotations

import logging
from itertools import islice
from typing import Iterable, Mapping, Sequence

import pandas as pd
from sqlalchemy import Column, MetaData, Table, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.sqltypes import NullType


logger = logging.getLogger(__name__)


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
        if not version or not version.startswith("2."):
            raise RuntimeError(
                f"Doris version {version or 'unknown'} not supported, require 2.x+"
            )
        logger.info(
            "replace_into using Doris UPSERT for %s (version=%s)",
            table_obj.fullname,
            version,
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
    try:
        result = conn.execute(text("SELECT version()"))
    except Exception as exc:  # pragma: no cover - network failures or permissions
        logger.warning("replace_into could not determine Doris version: %s", exc)
        return None
    try:
        version = result.scalar()
    except Exception:  # pragma: no cover - scalar not supported
        return None
    if isinstance(version, str):
        return version.strip()
    return None


__all__ = ["fetch_dataframe", "replace_into"]
