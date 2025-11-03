"""Read/write helpers for Doris using SQLAlchemy."""
from __future__ import annotations

from itertools import islice
from typing import Iterable, Mapping, Sequence

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.mysql import insert as mysql_insert


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
    """Execute an UPSERT using MySQL ``ON DUPLICATE KEY UPDATE`` semantics."""

    if df.empty:
        return 0

    records = _normalise_records(df.to_dict(orient="records"))

    metadata = MetaData()
    table_obj = Table(table, metadata, autoload_with=engine)

    insert_stmt = mysql_insert(table_obj)
    update_targets = {
        column.name: getattr(insert_stmt.inserted, column.name)
        for column in table_obj.columns
        if not column.primary_key
    }
    if not update_targets:
        primary_columns = list(table_obj.primary_key.columns)
        if not primary_columns:
            raise ValueError(f"Table '{table}' has no primary key defined; cannot build UPSERT")
        update_targets = {
            column.name: getattr(insert_stmt.inserted, column.name) for column in primary_columns
        }

    upsert_stmt = insert_stmt.on_duplicate_key_update(**update_targets)

    affected = 0
    with engine.begin() as conn:
        for chunk in _chunks(records, chunk_size):
            conn.execute(upsert_stmt, chunk)
            affected += len(chunk)
    return affected


__all__ = ["fetch_dataframe", "replace_into"]
