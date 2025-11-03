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


def _build_mock_engine(url: str, collector: list[str]):
    def executor(sql, *multiparams, **params):  # type: ignore[override]
        collector.append(str(sql))
        return _MockResult()

    return create_engine(url, strategy="mock", executor=executor)


def test_replace_into_mysql_uses_on_duplicate() -> None:
    statements: list[str] = []
    engine = _build_mock_engine("mysql+pymysql://user:pass@localhost/test", statements)

    df = pd.DataFrame([{"id": 1, "value": 10}, {"id": 2, "value": 20}])

    inserted = replace_into(engine, "sample_table", df, chunk_size=100)

    assert inserted == 2
    assert any("ON DUPLICATE KEY UPDATE" in stmt for stmt in statements)


def test_replace_into_doris_replaces_when_no_version(monkeypatch: pytest.MonkeyPatch) -> None:
    statements: list[str] = []
    engine = _build_mock_engine("mysql+pymysql://user:pass@doris-host/test", statements)

    monkeypatch.setattr("scpc.db.io._get_doris_version", lambda conn: None)

    df = pd.DataFrame([{"scene": "A", "metric": 1}])

    inserted = replace_into(engine, "bi_table", df)

    assert inserted == 1
    assert any(stmt.startswith("REPLACE INTO") for stmt in statements)


def test_replace_into_doris_uses_upsert_for_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    statements: list[str] = []
    engine = _build_mock_engine("mysql+pymysql://user:pass@doris-cluster/test", statements)

    monkeypatch.setattr("scpc.db.io._get_doris_version", lambda conn: "2.0.1")

    df = pd.DataFrame([{"scene": "B", "metric": 5}])

    inserted = replace_into(engine, "bi_table", df)

    assert inserted == 1
    assert any(stmt.startswith("UPSERT INTO") for stmt in statements)
