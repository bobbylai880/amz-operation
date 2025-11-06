from __future__ import annotations

import json
import logging
from typing import Any, Callable

import pytest

pytest.importorskip("sqlalchemy")

pytest.importorskip("pandas")
import pandas as pd

np = pytest.importorskip("numpy")

from scpc.db.io import _get_doris_version, _normalise_records, replace_into


class _MockResult:
    def __init__(self, scalar_value: Any = None) -> None:
        self._scalar_value = scalar_value

    def scalar(self) -> Any:
        return self._scalar_value


SideEffect = Callable[[str], Any]


class _IdentifierPreparer:
    def quote(self, identifier: str) -> str:
        return f"`{identifier}`"

    def quote_identifier(self, identifier: str) -> str:
        return self.quote(identifier)

    def format_table(self, table_obj: Any) -> str:
        return table_obj.fullname


def _build_mock_engine(
    url: str,
    collector: list[str],
    *,
    side_effect: SideEffect | None = None,
) -> Any:
    del url
    def executor(sql, params=None):
        statement = str(sql)
        collector.append(statement)
        if side_effect is not None:
            result = side_effect(statement)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, _MockResult):
                return result
            if result is not None:
                return result
        return _MockResult()

    class _MockConnection:
        def execute(self, sql, params=None):
            return executor(sql, params)

        def close(self) -> None:  # pragma: no cover - nothing to release
            pass

    class _BeginContext:
        def __enter__(self):
            return _MockConnection()

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class _EngineWrapper:
        def __init__(self) -> None:
            self.dialect = type("_Dialect", (), {"identifier_preparer": _IdentifierPreparer()})()

        def begin(self) -> _BeginContext:
            return _BeginContext()

        def connect(self):
            raise RuntimeError("reflection not supported")

    return _EngineWrapper()


def test_replace_into_uses_insert_upsert_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    statements: list[str] = []

    def side_effect(statement: str):
        lowered = statement.lower()
        if "version_comment" in lowered:
            return RuntimeError("function not found")
        if "version()" in lowered:
            return _MockResult("5.7.99-doris")
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
        side_effect=side_effect,
    )

    df = pd.DataFrame([{"scene": "A", "metric": 1}, {"scene": "B", "metric": 2}])

    caplog.set_level(logging.INFO)
    inserted = replace_into(engine, "bi_table", df, chunk_size=2)

    assert inserted == 2
    inserts = [stmt for stmt in statements if stmt.upper().startswith("INSERT INTO")]
    assert len(inserts) == 1
    assert not any("UPSERT INTO" in stmt.upper() for stmt in statements)
    assert not any("REPLACE INTO" in stmt.upper() for stmt in statements)
    assert "Doris 2.x (mysql-compatible)" in caplog.text
    assert "rows=2" in caplog.text


def test_replace_into_chunks_execution_counts() -> None:
    statements: list[str] = []
    insert_calls = 0

    def side_effect(statement: str):
        nonlocal insert_calls
        lowered = statement.lower()
        if "version_comment" in lowered:
            return RuntimeError("function not found")
        if "version()" in lowered:
            return _MockResult("5.7.99-doris")
        if "insert into" in lowered:
            insert_calls += 1
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
        side_effect=side_effect,
    )

    df = pd.DataFrame([
        {"scene": f"scene-{i}", "metric": i}
        for i in range(116)
    ])

    inserted = replace_into(engine, "bi_table", df, chunk_size=50)

    assert inserted == 116
    assert insert_calls == 3
    assert sum(1 for stmt in statements if stmt.upper().startswith("INSERT INTO")) == 3


def test_replace_into_reflection_fallback_logs(caplog: pytest.LogCaptureFixture) -> None:
    statements: list[str] = []

    def side_effect(statement: str):
        lowered = statement.lower()
        if "version_comment" in lowered:
            return RuntimeError("not supported")
        if "version()" in lowered:
            return _MockResult("5.7.99")
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
        side_effect=side_effect,
    )

    df = pd.DataFrame([{"scene": "A", "metric": 1}])

    caplog.set_level(logging.DEBUG)
    inserted = replace_into(engine, "bi_table", df, chunk_size=1)

    assert inserted == 1
    assert "falling back to synthetic metadata" in caplog.text
    assert any(stmt.upper().startswith("INSERT INTO") for stmt in statements)


def test_version_detection_5799_maps_to_doris_mysql_layer() -> None:
    class _Conn:
        def execute(self, sql, params=None):
            statement = str(sql).lower()
            if "version_comment" in statement:
                raise RuntimeError("not supported")
            if "version()" in statement:
                return _MockResult("5.7.99-abcdef")
            raise AssertionError(f"unexpected statement: {sql}")

    version = _get_doris_version(_Conn())

    assert version == "Doris 2.x (mysql-compatible)"


def test_normalise_records_serialises_containers_and_handles_empty_arrays() -> None:
    records = [
        {
            "ts": pd.Timestamp("2025-11-06 12:00:00"),
            "none_value": None,
            "nan_scalar": float("nan"),
            "numpy_scalar": np.float64(np.nan),
            "list_value": [],
            "tuple_value": ("a", 1),
            "dict_value": {"badge": "New"},
            "ndarray_value": np.array([], dtype=float),
        }
    ]

    normalised = _normalise_records(records)

    assert normalised[0]["ts"].isoformat() == "2025-11-06T12:00:00"
    assert normalised[0]["none_value"] is None
    assert normalised[0]["nan_scalar"] is None
    assert normalised[0]["numpy_scalar"] is None
    assert isinstance(normalised[0]["list_value"], str)
    assert isinstance(normalised[0]["tuple_value"], str)
    assert isinstance(normalised[0]["dict_value"], str)
    assert isinstance(normalised[0]["ndarray_value"], str)
    assert json.loads(normalised[0]["list_value"]) == []
    assert json.loads(normalised[0]["tuple_value"]) == ["a", 1]
    assert json.loads(normalised[0]["dict_value"]) == {"badge": "New"}
    assert json.loads(normalised[0]["ndarray_value"]) == []
