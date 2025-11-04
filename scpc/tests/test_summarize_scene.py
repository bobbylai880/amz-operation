import json
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
    return pd.DataFrame(
        {
            "scene": ["X"],
            "marketplace_id": ["US"],
            "year": [2025],
            "week_num": [12],
            "start_date": [pd.Timestamp("2025-03-23")],
        }
    )


@pytest.fixture
def sample_drivers():
    return pd.DataFrame(
        {
            "scene": ["X"],
            "marketplace_id": ["US"],
            "year": [2025],
            "week_num": [12],
            "start_date": [pd.Timestamp("2025-03-23")],
            "horizon": ["w1"],
            "direction": ["pos"],
            "keyword": ["demo"],
            "contrib": [0.3],
            "vol_delta": [1.0],
            "rank_delta": [0.0],
            "clickShare_delta": [0.0],
            "conversionShare_delta": [0.0],
            "is_new_kw": [True],
        }
    )


def test_summarize_scene_retries_on_schema_error(monkeypatch, sample_features, sample_drivers):
    calls = []

    def fake_read_sql(query, _conn, params):
        if "scene_features" in str(query):
            return sample_features
        return sample_drivers

    class StubClient:
        def __init__(self):
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            calls.append(_kwargs["facts"])
            if self.calls == 1:
                payload = {"drivers": [], "insufficient_data": False}
            else:
                payload = {
                    "status": "stable",
                    "drivers": [{"keyword": "demo", "delta": 0.3}],
                    "insufficient_data": False,
                }
            return DeepSeekResponse(content=json.dumps(payload), usage={})

        def close(self):
            pass

    settings = SimpleNamespace(base_url="https://api.deepseek.com", model="deepseek", api_key="key", timeout=1)

    monkeypatch.setattr("scpc.llm.summarize_scene.pd.read_sql_query", fake_read_sql)
    monkeypatch.setattr("scpc.llm.summarize_scene.create_client_from_env", lambda settings: StubClient())
    monkeypatch.setattr("scpc.llm.summarize_scene.get_deepseek_settings", lambda: settings)

    result = summarize_scene(engine=DummyEngine(), scene="X", mk="US", topn=5)

    assert result["status"] == "stable"
    assert calls, "expected DeepSeek to be invoked"
    assert "response_schema" in json.loads(calls[0])


def test_summarize_scene_raises_after_two_schema_errors(monkeypatch, sample_features, sample_drivers):
    def fake_read_sql(query, _conn, params):
        if "scene_features" in str(query):
            return sample_features
        return sample_drivers

    class StubClient:
        def __init__(self):
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            payload = {"drivers": [], "insufficient_data": False}
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
