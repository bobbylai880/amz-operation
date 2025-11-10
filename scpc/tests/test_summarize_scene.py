import json
import textwrap
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

import scpc.llm.summarize_scene as summarize_scene_module
from scpc.llm.deepseek_client import DeepSeekResponse
from scpc.llm.summarize_scene import SceneSummarizationError, summarize_scene


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
    current_weeks = [base + timedelta(weeks=offset) for offset in range(6)]
    last_year_past = [dt - timedelta(days=364) for dt in current_weeks]
    pivot = last_year_past[-1]
    last_year_future = [pivot + timedelta(weeks=offset) for offset in range(1, 7)]
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
    current_weeks = [base + timedelta(weeks=offset) for offset in range(6)]
    last_year_past = [dt - timedelta(days=364) for dt in current_weeks]
    pivot = last_year_past[-1]
    last_year_future = [pivot + timedelta(weeks=offset) for offset in range(1, 7)]
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
    monkeypatch, tmp_path, sample_features, sample_drivers, sample_keyword_volumes
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
                    "confidence": 0.7,
                    "insufficient_data": False,
                }
            else:
                payload = {
                    "confidence": 0.72,
                    "insufficient_data": False,
                    "analysis_summary": "场景短期延续上行，需关注竞对动态与库存调整后的波动。",
                    "notes": None,
                }
            return DeepSeekResponse(content=json.dumps(payload), usage={})

        def close(self):
            pass

    settings = SimpleNamespace(base_url="https://api.deepseek.com", model="deepseek", api_key="key", timeout=1)

    monkeypatch.setattr("scpc.llm.summarize_scene.pd.read_sql_query", fake_read_sql)
    monkeypatch.setattr("scpc.llm.summarize_scene.create_client_from_env", lambda settings: StubClient())
    monkeypatch.setattr("scpc.llm.summarize_scene.get_deepseek_settings", lambda: settings)
    monkeypatch.setenv("SCPC_PROMPT_LOG_DIR", str(tmp_path))

    result = summarize_scene(engine=DummyEngine(), scene="X", mk="US", topn=5)

    assert "scene_forecast" in result
    assert len(result["scene_forecast"]["weeks"]) == 6
    assert "analysis_summary" in result
    assert calls, "expected DeepSeek to be invoked"
    parsed = json.loads(calls[0])
    assert "scene_recent_4w" in parsed
    assert "response_schema" in parsed
    forecast_guidance = parsed.get("forecast_guidance", {})
    scene_guidance = forecast_guidance.get("scene", {}) if isinstance(forecast_guidance, dict) else {}
    weeks = scene_guidance.get("forecast_weeks", []) if isinstance(scene_guidance, dict) else []
    if weeks:
        assert len(weeks) == 6
        first_week = weeks[0]
        assert isinstance(first_week.get("pct_change"), str)
        assert first_week["pct_change"].endswith("%")
        assert "pct_change_value" in first_week
        assert "projected_vol" in first_week
    keyword_guidance = forecast_guidance.get("keywords", []) if isinstance(forecast_guidance, dict) else []
    assert len(keyword_guidance) == 3
    for kw in keyword_guidance:
        weeks = kw.get("forecast_weeks", [])
        if weeks:
            assert len(weeks) == 6
            first_kw_week = weeks[0]
            assert "projected_vol" in first_kw_week
            assert isinstance(first_kw_week.get("pct_change"), str)
            assert first_kw_week["pct_change"].endswith("%")

    artifacts = sorted(tmp_path.glob("*.json"))
    assert len(artifacts) == 2
    latest = artifacts[-1]
    stored = json.loads(latest.read_text(encoding="utf-8"))
    assert stored["system_prompt"] == summarize_scene.__globals__["SYSTEM_PROMPT"]
    assert stored["scene"] == "X"
    assert stored["marketplace_id"] == "US"
    assert isinstance(stored["facts"], dict)
    assert stored["facts"]["scene"] == "X"


def test_top_keyword_limit_respects_config(
    monkeypatch, tmp_path, sample_features, sample_drivers, sample_keyword_volumes
):
    config_text = textwrap.dedent(
        """
        scene_forecast:
          output:
            forecast_horizon_weeks: 4
            pct_digits: 2
            vol_digits: 2
            wow_digits: 4
            max_top_keywords: 2
          blend_weights:
            recent_wow: 0.6
            seasonal: 0.4
            dynamic: true
          thresholds:
            flat_band_pct: 0.01
            low_conf_flat_band: 0.02
          quality:
            min_non_missing: 2
            quality_notes_template: ""
          bounds:
            enabled: true
            use_columns: [forecast_p10, forecast_p50, forecast_p90]
            clamp_on_4th_week: true
          prompt_log_dir: storage/prompt_logs
          max_attempts: 2
          temperature: 0.1
          model: deepseek-chat
        """
    ).strip()
    config_path = tmp_path / "scene_forecast_config.yaml"
    config_path.write_text(config_text + "\n", encoding="utf-8")
    monkeypatch.setattr("scpc.llm.summarize_scene.CONFIG_PATH", config_path)
    summarize_scene_module._scene_forecast_config.cache_clear()

    def fake_read_sql(query, _conn, params):
        text_query = str(query)
        if "bi_amz_scene_features" in text_query:
            return sample_features
        if "bi_amz_scene_drivers" in text_query:
            return sample_drivers
        if "bi_amz_vw_kw_week" in text_query:
            return sample_keyword_volumes
        raise AssertionError(f"Unexpected query: {query}")

    calls = []

    class StubClient:
        def generate(self, **_kwargs):
            payload = json.loads(_kwargs["facts"])
            calls.append(payload)
            response = {
                "confidence": 0.8,
                "insufficient_data": False,
                "analysis_summary": "场景保持稳健，核心关键词继续拉动。",
                "notes": None,
            }
            return DeepSeekResponse(content=json.dumps(response), usage={})

        def close(self):
            pass

    settings = SimpleNamespace(
        base_url="https://api.deepseek.com", model="deepseek", api_key="key", timeout=1
    )

    monkeypatch.setattr("scpc.llm.summarize_scene.pd.read_sql_query", fake_read_sql)
    monkeypatch.setattr(
        "scpc.llm.summarize_scene.create_client_from_env", lambda settings: StubClient()
    )
    monkeypatch.setattr("scpc.llm.summarize_scene.get_deepseek_settings", lambda: settings)
    prompt_dir = tmp_path / "logs"
    monkeypatch.setenv("SCPC_PROMPT_LOG_DIR", str(prompt_dir))

    result = summarize_scene(engine=DummyEngine(), scene="X", mk="US", topn=5)

    assert len(result.get("top_keywords_forecast", [])) == 2
    assert calls, "expected DeepSeek to be invoked"
    first_call = calls[0]
    assert len(first_call["top_keywords"]) == 2
    instructions = first_call["output_instructions"]
    assert any("不要输出 scene_forecast" in item for item in instructions)
    assert any("analysis_summary 需以中文总结" in item for item in instructions)
    schema = first_call["response_schema"]
    required = set(schema.get("required", []))
    assert required == {"confidence", "insufficient_data", "analysis_summary"}
    properties = schema.get("properties", {})
    assert "scene_forecast" not in properties
    assert "top_keywords_forecast" not in properties
    summarize_scene_module._scene_forecast_config.cache_clear()


def test_summarize_scene_raises_after_two_schema_errors(
    monkeypatch, tmp_path, sample_features, sample_drivers, sample_keyword_volumes
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
                "confidence": 0.5,
            }
            return DeepSeekResponse(content=json.dumps(payload), usage={})

        def close(self):
            pass

    settings = SimpleNamespace(base_url="https://api.deepseek.com", model="deepseek", api_key="key", timeout=1)

    monkeypatch.setattr("scpc.llm.summarize_scene.pd.read_sql_query", fake_read_sql)
    monkeypatch.setattr("scpc.llm.summarize_scene.create_client_from_env", lambda settings: StubClient())
    monkeypatch.setattr("scpc.llm.summarize_scene.get_deepseek_settings", lambda: settings)
    monkeypatch.setenv("SCPC_PROMPT_LOG_DIR", str(tmp_path))

    with pytest.raises(SceneSummarizationError) as excinfo:
        summarize_scene(engine=DummyEngine(), scene="X", mk="US", topn=5)

    error = excinfo.value
    assert any("Missing required key" in detail for detail in error.details)
    assert error.raw is not None
    artifacts = sorted(tmp_path.glob("*.json"))
    assert len(artifacts) == 2
