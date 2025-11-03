from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine

pytest.importorskip("pandas")
import pandas as pd

from scpc.db.io import replace_into


class _MockResult:
    def __init__(self, scalar_value: Any = None) -> None:
        self._scalar_value = scalar_value

    def scalar(self) -> Any:
        return self._scalar_value


def _build_mock_engine(url: str, collector: list[str], *, side_effect=None):
    def executor(sql, *multiparams, **params):  # type: ignore[override]
        statement = str(sql)
        collector.append(statement)
        if side_effect is not None:
            result = side_effect(statement)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, _MockResult):
                return result
        return _MockResult()

    return create_engine(url, strategy="mock", executor=executor)


def test_replace_into_uses_doris_upsert() -> None:
    statements: list[str] = []

    def side_effect(statement: str):
        if "SELECT version()" in statement:
            return _MockResult("2.1.0")
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
        side_effect=side_effect,
    )

    df = pd.DataFrame([{"scene": "A", "metric": 1}, {"scene": "B", "metric": 2}])

    inserted = replace_into(engine, "bi_table", df, chunk_size=2)

    assert inserted == 2
    assert any(stmt.startswith("SELECT version()") for stmt in statements)
    assert any("UPSERT INTO" in stmt for stmt in statements)


def test_replace_into_requires_doris_v2() -> None:
    statements: list[str] = []

    def side_effect(statement: str):
        if "SELECT version()" in statement:
            return _MockResult("1.2.3")
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
        side_effect=side_effect,
    )

    df = pd.DataFrame([{"scene": "A", "metric": 1}])

    with pytest.raises(RuntimeError, match=r"require 2.x\+"):
        replace_into(engine, "bi_table", df)

    assert any(stmt.startswith("SELECT version()") for stmt in statements)
    assert not any("UPSERT INTO" in stmt for stmt in statements)
