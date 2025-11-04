import json
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from scpc.llm.deepseek_client import DeepSeekResponse
from scpc.llm.summarize_scene import SceneSummarizationError, summarize_scene

pd = pytest.importorskip("pandas")


class DummyEngine:
    def connect(self):
        return DummyConnection()


class DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def sample_features():
    base = date(2025, 3, 2)
    current_weeks = [base + timedelta(weeks=offset) for offset in range(4)]
    last_year_past = [dt - timedelta(days=364) for dt in current_weeks]
    pivot = last_year_past[-1]
    last_year_future = [pivot + timedelta(weeks=offset) for offset in range(1, 5)]
    all_dates = sorted(set(current_weeks + last_year_past + last_year_future))

    def _volume_for(dt: date) -> float:
        if dt in current_weeks:
            return 200 + current_weeks.index(dt) * 10
        if dt in last_year_past:
            return 120 + last_year_past.index(dt) * 5
        return 150 + last_year_future.index(dt) * 4

    rows = []
    for dt in all_dates:
        iso = dt.isocalendar()
        rows.append(
            {
                "scene": "X",
                "marketplace_id": "US",
                "year": iso.year,
                "week_num": iso.week,
                "start_date": pd.Timestamp(dt),
                "VOL": _volume_for(dt),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_drivers():
    return pd.DataFrame(
        {
            "scene": ["X", "X", "X"],
            "marketplace_id": ["US", "US", "US"],
            "year": [2025, 2025, 2025],
            "week_num": [12, 12, 12],
            "start_date": [pd.Timestamp("2025-03-23")] * 3,
            "horizon": ["w1", "w1", "w1"],
            "direction": ["pos", "neg", "pos"],
            "keyword": ["alpha", "beta", "gamma"],
            "contrib": [0.25, -0.4, 0.18],
        }
    )


@pytest.fixture
def sample_keyword_volumes():
    base = date(2025, 3, 2)
    current_weeks = [base + timedelta(weeks=offset) for offset in range(4)]
    last_year_past = [dt - timedelta(days=364) for dt in current_weeks]
    pivot = last_year_past[-1]
    last_year_future = [pivot + timedelta(weeks=offset) for offset in range(1, 5)]
    all_dates = sorted(set(current_weeks + last_year_past + last_year_future))
    keywords = {"alpha": 300, "beta": 260, "gamma": 220}
    rows = []
    for keyword, base_vol in keywords.items():
        for idx, dt in enumerate(all_dates):
            iso = dt.isocalendar()
            rows.append(
                {
                    "keyword_norm": keyword,
                    "year": iso.year,
                    "week_num": iso.week,
                    "startDate": pd.Timestamp(dt),
                    "vol": base_vol + idx * 8,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_scene_retries_on_schema_error(
    monkeypatch, sample_features, sample_drivers, sample_keyword_volumes
):
    calls = []

    def fake_read_sql(query, _conn, params):
        text_query = str(query)
        if "bi_amz_scene_features" in text_query:
            return sample_features
        if "bi_amz_scene_drivers" in text_query:
            return sample_drivers
        if "bi_amz_vw_kw_week" in text_query:
            return sample_keyword_volumes
        raise AssertionError(f"Unexpected query: {query}")

    class StubClient:
        def __init__(self):
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            calls.append(_kwargs["facts"])
            if self.calls == 1:
                payload = {
                    "scene_forecast": {"weeks": []},
                    "top_keywords_forecast": [],
                    "confidence": 0.7,
                    "insufficient_data": False,
                    "analysis_summary": "场景数据不足，无法输出有效趋势。",
                }
            else:
                payload = {
                    "scene_forecast": {
                        "weeks": [
                            {
                                "year": 2025,
                                "week_num": 13,
                                "start_date": "2025-03-30",
                                "direction": "up",
                                "pct_change": 1.2,
                            },
                            {
                                "year": 2025,
                                "week_num": 14,
                                "start_date": "2025-04-06",
                                "direction": "up",
                                "pct_change": 1.5,
                            },
                            {
                                "year": 2025,
                                "week_num": 15,
                                "start_date": "2025-04-13",
                                "direction": "flat",
                                "pct_change": 0.4,
                            },
                            {
                                "year": 2025,
                                "week_num": 16,
                                "start_date": "2025-04-20",
                                "direction": "down",
                                "pct_change": -0.6,
                            },
                        ]
                    },
                    "top_keywords_forecast": [
                        {
                            "keyword": "beta",
                            "weeks": [
                                {
                                    "year": 2025,
                                    "week_num": 13,
                                    "start_date": "2025-03-30",
                                    "direction": "up",
                                    "pct_change": 1.1,
                                }
                            ],
                        }
                    ],
                    "confidence": 0.72,
                    "insufficient_data": False,
                    "analysis_summary": "场景未来两周延续上行，关键词beta放量拉动，随后受库存调整回落。",
                    "notes": None,
                }
            return DeepSeekResponse(content=json.dumps(payload), usage={})

        def close(self):
            pass

    settings = SimpleNamespace(base_url="https://api.deepseek.com", model="deepseek", api_key="key", timeout=1)

    monkeypatch.setattr("scpc.llm.summarize_scene.pd.read_sql_query", fake_read_sql)
    monkeypatch.setattr("scpc.llm.summarize_scene.create_client_from_env", lambda settings: StubClient())
    monkeypatch.setattr("scpc.llm.summarize_scene.get_deepseek_settings", lambda: settings)

    result = summarize_scene(engine=DummyEngine(), scene="X", mk="US", topn=5)

    assert "scene_forecast" in result
    assert len(result["scene_forecast"]["weeks"]) == 4
    assert "analysis_summary" in result
    assert calls, "expected DeepSeek to be invoked"
    parsed = json.loads(calls[0])
    assert "scene_recent_4w" in parsed
    assert "response_schema" in parsed
    forecast_guidance = parsed.get("forecast_guidance", {})
    scene_guidance = forecast_guidance.get("scene", {}) if isinstance(forecast_guidance, dict) else {}
    weeks = scene_guidance.get("forecast_weeks", []) if isinstance(scene_guidance, dict) else []
    if weeks:
        first_week = weeks[0]
        assert isinstance(first_week.get("pct_change"), str)
        assert first_week["pct_change"].endswith("%")
        assert "pct_change_value" in first_week


def test_summarize_scene_raises_after_two_schema_errors(
    monkeypatch, sample_features, sample_drivers, sample_keyword_volumes
):
    def fake_read_sql(query, _conn, params):
        text_query = str(query)
        if "bi_amz_scene_features" in text_query:
            return sample_features
        if "bi_amz_scene_drivers" in text_query:
            return sample_drivers
        if "bi_amz_vw_kw_week" in text_query:
            return sample_keyword_volumes
        raise AssertionError(f"Unexpected query: {query}")

    class StubClient:
        def __init__(self):
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            payload = {
                "scene_forecast": {"weeks": []},
                "top_keywords_forecast": [],
                "confidence": 0.5,
            }
            return DeepSeekResponse(content=json.dumps(payload), usage={})

        def close(self):
            pass

    settings = SimpleNamespace(base_url="https://api.deepseek.com", model="deepseek", api_key="key", timeout=1)

    monkeypatch.setattr("scpc.llm.summarize_scene.pd.read_sql_query", fake_read_sql)
    monkeypatch.setattr("scpc.llm.summarize_scene.create_client_from_env", lambda settings: StubClient())
    monkeypatch.setattr("scpc.llm.summarize_scene.get_deepseek_settings", lambda: settings)

    with pytest.raises(SceneSummarizationError) as excinfo:
        summarize_scene(engine=DummyEngine(), scene="X", mk="US", topn=5)

    error = excinfo.value
    assert any("Missing required key" in detail for detail in error.details)
    assert error.raw is not None
