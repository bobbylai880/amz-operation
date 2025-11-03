from __future__ import annotations

from typing import Any, Callable

import pytest

pytest.importorskip("sqlalchemy")

pytest.importorskip("pandas")
import pandas as pd

from scpc.db.io import replace_into


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


def test_replace_into_prefers_upsert_for_doris_two_x() -> None:
    statements: list[str] = []

    def side_effect(statement: str):
        lowered = statement.lower()
        if "version_comment" in lowered:
            return _MockResult("Doris version 2.0.4")
        if "version()" in lowered:
            return _MockResult("5.7.99")
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
        side_effect=side_effect,
    )

    df = pd.DataFrame([{"scene": "A", "metric": 1}, {"scene": "B", "metric": 2}])

    inserted = replace_into(engine, "bi_table", df, chunk_size=2)

    assert inserted == 2
    assert any("UPSERT INTO" in stmt for stmt in statements)
    assert sum(1 for stmt in statements if "REPLACE INTO" in stmt) == 0


def test_replace_into_falls_back_to_replace_when_upsert_rejected() -> None:
    statements: list[str] = []
    call_count = {"upsert": 0}

    def side_effect(statement: str):
        lowered = statement.lower()
        if "version_comment" in lowered:
            return _MockResult("Doris version 2.0.4")
        if "version()" in lowered:
            return _MockResult("5.7.99")
        if "upsert into" in lowered:
            if call_count["upsert"] == 0:
                call_count["upsert"] += 1
                return RuntimeError("Can not change UNIQUE KEY to Merge-On-Write mode")
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
        side_effect=side_effect,
    )

    df = pd.DataFrame([{"scene": "A", "metric": 1}, {"scene": "B", "metric": 2}])

    inserted = replace_into(engine, "bi_table", df, chunk_size=2)

    assert inserted == 2
    assert any("UPSERT INTO" in stmt for stmt in statements)
    assert any("REPLACE INTO" in stmt for stmt in statements)
    assert statements.index(next(stmt for stmt in statements if "REPLACE INTO" in stmt)) > statements.index(
        next(stmt for stmt in statements if "UPSERT INTO" in stmt)
    )


def test_replace_into_uses_replace_when_version_unknown() -> None:
    statements: list[str] = []

    def side_effect(statement: str):
        lowered = statement.lower()
        if "version_comment" in lowered:
            return RuntimeError("function not found")
        if "version()" in lowered:
            return _MockResult("5.6.0")
        return None

    engine = _build_mock_engine(
        "mysql+pymysql://user:pass@doris-host/test",
        statements,
    )

    df = pd.DataFrame([{"scene": "A", "metric": 1}, {"scene": "B", "metric": 2}])

    inserted = replace_into(engine, "bi_table", df, chunk_size=2)

    assert inserted == 2
    assert any("REPLACE INTO" in stmt for stmt in statements)
    assert not any("UPSERT INTO" in stmt for stmt in statements)
