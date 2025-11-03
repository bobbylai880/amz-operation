"""Read/write helpers for Doris using SQLAlchemy."""
from __future__ import annotations

from itertools import islice
from typing import Iterable, Mapping, Sequence

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


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
    """Execute an UPSERT using ``INSERT ... ON DUPLICATE KEY UPDATE`` semantics."""

    if df.empty:
        return 0

    data = df.to_dict(orient="records")
    records = _normalise_records(data)
    columns = list(df.columns)
    col_clause = ", ".join(columns)
    values_clause = ", ".join(f":{col}" for col in columns)
    update_clause = ", ".join(f"{col} = VALUES({col})" for col in columns)
    sql = (
        f"INSERT INTO {table} ({col_clause}) VALUES ({values_clause})"
        + (f" ON DUPLICATE KEY UPDATE {update_clause}" if update_clause else "")
    )
    stmt = text(sql)

    affected = 0
    with engine.begin() as conn:
        for chunk in _chunks(records, chunk_size):
            conn.execute(stmt, chunk)
            affected += len(chunk)
    return affected


__all__ = ["fetch_dataframe", "replace_into"]
