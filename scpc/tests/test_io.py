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


def test_replace_into_uses_force_replace_mode_without_version_probe() -> None:
    statements: list[str] = []

    def side_effect(statement: str):
        if "version" in statement.lower():
            raise AssertionError("replace_into should not query Doris version")
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
        side_effect=side_effect,
    )

    df = pd.DataFrame([{"scene": "A", "metric": 1}, {"scene": "B", "metric": 2}])

    inserted = replace_into(engine, "bi_table", df, chunk_size=2)

    assert inserted == 2
    assert any("REPLACE INTO" in stmt for stmt in statements)
    assert not any("SELECT version" in stmt for stmt in statements)


def test_replace_into_batches_rows_using_replace_into() -> None:
    statements: list[str] = []

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
    )

    df = pd.DataFrame(
        [
            {"scene": "A", "metric": 1},
            {"scene": "B", "metric": 2},
            {"scene": "C", "metric": 3},
        ]
    )

    inserted = replace_into(engine, "bi_table", df, chunk_size=2)

    assert inserted == 3
    assert sum(1 for stmt in statements if "REPLACE INTO" in stmt) == 2
