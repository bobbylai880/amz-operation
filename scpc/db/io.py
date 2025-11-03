"""Read/write helpers for Doris/MySQL using SQLAlchemy."""
from __future__ import annotations

import logging
from itertools import islice
from typing import Iterable, Mapping, Sequence

import pandas as pd
from sqlalchemy import Column, MetaData, Table, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.sqltypes import NullType


logger = logging.getLogger(__name__)


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
    """Execute an UPSERT using Doris preferred semantics with MySQL compatibility."""

    if df.empty:
        return 0

    records = _normalise_records(df.to_dict(orient="records"))
    columns = list(df.columns)
    col_list = ", ".join(f"`{col}`" for col in columns)
    placeholders = ", ".join(f"%({col})s" for col in columns)

    update_columns = [col for col in columns if col not in {"create_time"}]
    if not update_columns:
        update_columns = columns
    update_clause = ", ".join(f"`{col}`=VALUES(`{col}`)" for col in update_columns)

    insert_sql = (
        "INSERT INTO {table} ({cols}) VALUES ({values}) ON DUPLICATE KEY UPDATE {updates}".format(
            table=table,
            cols=col_list,
            values=placeholders,
            updates=update_clause,
        )
    )

    affected = 0
    with engine.begin() as conn:
        for chunk in _chunks(records, chunk_size):
            try:
                conn.exec_driver_sql(insert_sql, chunk)
                affected += len(chunk)
            except Exception as exc:
                message = str(exc)
                if "Encountered: ON" in message or "Expected: COMMA" in message:
                    replace_sql = f"REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
                    logger.info(
                        "replace_into falling back to Doris REPLACE INTO for %s due to %s",
                        table,
                        message,
                    )
                    conn.exec_driver_sql(replace_sql, chunk)
                    affected += len(chunk)
                else:
                    raise
    return affected


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
        verb = "UPSERT INTO" if version and version.startswith("2.") else "REPLACE INTO"
        logger.info(
            "replace_into using Doris %s for %s (version=%s)",
            "UPSERT" if verb.startswith("UPSERT") else "REPLACE",
            table_obj.fullname,
            version or "unknown",
        )
        statement = text(
            f"{verb} {_format_table(preparer, table_obj)} ({', '.join(column_tokens)}) VALUES ({', '.join(value_tokens)})"
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


def _is_doris_engine(engine: Engine) -> bool:
    url = engine.url
    candidates = [
        getattr(url, "host", None),
        getattr(url, "database", None),
        getattr(url, "drivername", None),
    ]
    for candidate in candidates:
        if candidate and "doris" in candidate.lower():
            return True
    return False


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
