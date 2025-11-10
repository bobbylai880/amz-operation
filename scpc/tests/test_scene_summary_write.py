import time
from datetime import date, datetime
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from scpc.etl.scene_pipeline import (
    LATEST_SUMMARY_WEEK_SQL,
    _resolve_llm_metadata,
    _resolve_summary_week,
    write_scene_summary_to_db,
)


class DummyEngine:
    def connect(self):
        raise AssertionError("connect should not be called in this scenario")


def test_write_scene_summary_to_db_builds_dataframe(monkeypatch):
    captured: dict[str, object] = {}

    def _capture(engine, table, df, chunk_size=500):
        captured["engine"] = engine
        captured["table"] = table
        captured["df"] = df.copy()
        return len(df)

    monkeypatch.setattr("scpc.etl.scene_pipeline.replace_into", _capture)

    engine = object()
    inserted = write_scene_summary_to_db(
        engine,
        scene="Storage Rack",
        marketplace_id="US",
        week="2024W12",
        sunday=date(2024, 3, 17),
        summary_str="需求回暖，建议保持广告投入节奏。",
        confidence=0.83,
        llm_model="deepseek-pro",
        llm_version="v1.0",
    )

    assert inserted == 1
    assert captured["table"] == "bi_amz_scene_summary"
    df = captured["df"]
    assert list(df.columns) == [
        "scene",
        "marketplace_id",
        "week",
        "sunday",
        "confidence",
        "summary_str",
        "llm_model",
        "llm_version",
        "created_at",
        "updated_at",
    ]
    row = df.iloc[0]
    assert row["scene"] == "Storage Rack"
    assert row["marketplace_id"] == "US"
    assert row["week"] == "2024W12"
    assert row["sunday"] == date(2024, 3, 17)
    assert row["confidence"] == pytest.approx(0.83)
    assert row["summary_str"] == "需求回暖，建议保持广告投入节奏。"
    assert row["llm_model"] == "deepseek-pro"
    assert row["llm_version"] == "v1.0"
    assert isinstance(row["created_at"], datetime)
    assert isinstance(row["updated_at"], datetime)
    assert row["created_at"] == row["updated_at"]


def test_write_scene_summary_to_db_updates_timestamps(monkeypatch):
    timestamps: list[datetime] = []

    def _capture(_engine, _table, df, chunk_size=500):
        timestamps.append(df.iloc[0]["updated_at"])
        return len(df)

    monkeypatch.setattr("scpc.etl.scene_pipeline.replace_into", _capture)

    engine = object()
    write_scene_summary_to_db(
        engine,
        scene="Storage Rack",
        marketplace_id="US",
        week="2024W12",
        sunday=date(2024, 3, 17),
        summary_str="需求回暖，建议保持广告投入节奏。",
        confidence=0.83,
        llm_model="deepseek-pro",
        llm_version="v1.0",
    )
    time.sleep(0.01)
    write_scene_summary_to_db(
        engine,
        scene="Storage Rack",
        marketplace_id="US",
        week="2024W12",
        sunday=date(2024, 3, 17),
        summary_str="需求回暖，建议保持广告投入节奏。",
        confidence=0.83,
        llm_model="deepseek-pro",
        llm_version="v1.0",
    )

    assert len(timestamps) == 2
    assert timestamps[0] < timestamps[1]


def test_resolve_summary_week_prefers_features(monkeypatch):
    features = pd.DataFrame(
        {
            "year": [2023, 2024],
            "week_num": [52, 12],
            "start_date": [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-03-17")],
        }
    )
    outputs = {"features": features}

    week, sunday = _resolve_summary_week(outputs, DummyEngine(), "Scene", "US")

    assert week == "2024W12"
    assert sunday == date(2024, 3, 17)


class QueryEngine:
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def connect(self):
        return QueryEngine.Connection()


def test_resolve_summary_week_queries_database_when_missing(monkeypatch):
    calls: list[tuple] = []

    def _read_sql(query, conn, params):
        calls.append((query, params))
        return pd.DataFrame(
            {"year": [2024], "week_num": [15], "start_date": [pd.Timestamp("2024-04-07")]}
        )

    monkeypatch.setattr("scpc.etl.scene_pipeline.pd.read_sql_query", _read_sql)

    week, sunday = _resolve_summary_week({}, QueryEngine(), "Scene", "US")

    assert len(calls) == 1
    recorded_query, recorded_params = calls[0]
    assert str(recorded_query) == str(LATEST_SUMMARY_WEEK_SQL)
    assert recorded_params == {"scene": "Scene", "mk": "US"}
    assert week == "2024W15"
    assert sunday == date(2024, 4, 7)


def test_resolve_llm_metadata_prefers_config_override(monkeypatch):
    monkeypatch.setattr(
        "scpc.etl.scene_pipeline.get_deepseek_settings",
        lambda: SimpleNamespace(model="deepseek-env"),
    )
    monkeypatch.setattr(
        "scpc.llm.summarize_scene._scene_forecast_config",
        lambda: {"model": "deepseek-config", "prompt_version": "v9"},
    )

    model, version = _resolve_llm_metadata()

    assert model == "deepseek-config"
    assert version == "v9"
