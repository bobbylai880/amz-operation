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
        return _MockResult()

    return create_engine(url, strategy="mock", executor=executor)


def test_replace_into_mysql_uses_on_duplicate() -> None:
    statements: list[str] = []
    engine = _build_mock_engine("mysql+pymysql://user:pass@localhost/test", statements)

    df = pd.DataFrame([{"id": 1, "value": 10}, {"id": 2, "value": 20}])

    inserted = replace_into(engine, "sample_table", df, chunk_size=100)

    assert inserted == 2
    assert any("ON DUPLICATE KEY UPDATE" in stmt for stmt in statements)


def test_replace_into_doris_falls_back_to_replace() -> None:
    statements: list[str] = []
    triggered = {"raised": False}

    def side_effect(statement: str):
        if "ON DUPLICATE KEY UPDATE" in statement and not triggered["raised"]:
            triggered["raised"] = True
            return RuntimeError("Encountered: ON DUPLICATE KEY UPDATE Expected: COMMA")
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
        side_effect=side_effect,
    )

    df = pd.DataFrame([{"scene": "A", "metric": 1}])

    inserted = replace_into(engine, "bi_table", df)

    assert inserted == 1
    assert any(stmt.startswith("INSERT INTO") for stmt in statements)
    assert any(stmt.startswith("REPLACE INTO") for stmt in statements)
