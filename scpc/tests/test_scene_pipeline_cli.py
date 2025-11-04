import json

import pytest

from scpc.etl.scene_pipeline import main, parse_args

pd = pytest.importorskip("pandas")
sqlalchemy = pytest.importorskip("sqlalchemy")
OperationalError = sqlalchemy.exc.OperationalError


class DummyEngine:
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


def test_parse_args_strips_leading_space_tokens(monkeypatch):
    args = parse_args([" --scene", "浴室袋", " --mk", "US", "--weeks-back", "12", " --write"])
    assert args.scene == "浴室袋"
    assert args.mk == "US"
    assert args.weeks_back == 12
    assert args.write is True
    assert args.scene_topn is None
    assert args.with_llm is False
    assert args.emit_json is False
    assert args.emit_md is False
    assert args.llm_only is False


def test_parse_args_llm_only_enables_with_llm():
    args = parse_args(["--scene", "S", "--mk", "US", "--llm-only"])
    assert args.llm_only is True
    assert args.with_llm is True
    assert args.emit_json is True


def test_parse_args_llm_only_conflicts_with_write():
    with pytest.raises(SystemExit):
        parse_args(["--scene", "S", "--mk", "US", "--llm-only", "--write"])


def test_main_emits_scene_summary_json(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "scene"
    log_dir = tmp_path / "logs"
    features = pd.DataFrame({"year": [2024], "week_num": [12], "start_date": [pd.Timestamp("2024-03-17")]})
    outputs = {"clean": pd.DataFrame(), "features": features, "drivers": pd.DataFrame()}

    monkeypatch.setattr("scpc.etl.scene_pipeline.run_scene_pipeline", lambda *args, **kwargs: outputs)
    monkeypatch.setattr("scpc.etl.scene_pipeline.create_doris_engine", lambda: DummyEngine())
    summary = {"status": "OK", "drivers": [], "insufficient_data": False}
    monkeypatch.setattr("scpc.etl.scene_pipeline.summarize_scene", lambda **kwargs: summary)
    monkeypatch.setenv("SCPC_LOG_DIR", str(log_dir))

    main(
        [
            "--scene",
            "测试场景",
            "--mk",
            "US",
            "--weeks-back",
            "4",
            "--with-llm",
            "--emit-json",
            "--outputs-dir",
            str(outputs_dir),
        ]
    )

    yearweek_dir = outputs_dir / "测试场景" / "US" / "202412"
    artifact = yearweek_dir / "scene_summary.json"
    assert artifact.exists()

    logs = sorted(log_dir.glob("scene_pipeline_*.log"))
    assert logs, "expected log file to be created"
    log_contents = logs[-1].read_text(encoding="utf-8")
    assert "call=summarize_scene" in log_contents


def test_main_emits_scene_markdown(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "scene"
    log_dir = tmp_path / "logs"
    features = pd.DataFrame({"year": [2023], "week_num": [5], "start_date": [pd.Timestamp("2023-02-05")]})
    outputs = {"clean": pd.DataFrame(), "features": features, "drivers": pd.DataFrame()}

    monkeypatch.setattr("scpc.etl.scene_pipeline.run_scene_pipeline", lambda *args, **kwargs: outputs)
    monkeypatch.setattr("scpc.etl.scene_pipeline.create_doris_engine", lambda: DummyEngine())
    summary = {"status": "OK", "drivers": [], "insufficient_data": False}
    monkeypatch.setattr("scpc.etl.scene_pipeline.summarize_scene", lambda **kwargs: summary)
    monkeypatch.setattr("scpc.etl.scene_pipeline.build_scene_markdown", lambda payload: "# Demo\n")
    monkeypatch.setenv("SCPC_LOG_DIR", str(log_dir))

    main(
        [
            "--scene",
            "Another",
            "--mk",
            "DE",
            "--weeks-back",
            "6",
            "--with-llm",
            "--emit-md",
            "--outputs-dir",
            str(outputs_dir),
        ]
    )

    yearweek_dir = outputs_dir / "Another" / "DE" / "202305"
    artifact = yearweek_dir / "scene_report.md"
    assert artifact.exists()
    assert artifact.read_text(encoding="utf-8") == "# Demo\n"

    logs = sorted(log_dir.glob("scene_pipeline_*.log"))
    assert logs, "expected log file to be created"
    log_contents = logs[-1].read_text(encoding="utf-8")
    assert "call=summarize_scene" in log_contents


def test_main_llm_only_invokes_llm_without_etl(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "scene"
    log_dir = tmp_path / "logs"

    monkeypatch.setenv("SCPC_LOG_DIR", str(log_dir))

    def _unexpected(*_args, **_kwargs):  # pragma: no cover - ensures we skip ETL
        raise AssertionError("run_scene_pipeline should not be invoked when --llm-only is set")

    monkeypatch.setattr("scpc.etl.scene_pipeline.run_scene_pipeline", _unexpected)

    dummy_engine = DummyEngine()
    monkeypatch.setattr("scpc.etl.scene_pipeline.create_doris_engine", lambda: dummy_engine)

    summary = {"status": "OK", "drivers": [], "insufficient_data": False}

    called = {"summarize": False, "yearweek": False}

    def _summarize(**_kwargs):
        called["summarize"] = True
        return summary

    monkeypatch.setattr("scpc.etl.scene_pipeline.summarize_scene", _summarize)

    def _yearweek(*_args, **_kwargs):
        called["yearweek"] = True
        return 202412

    monkeypatch.setattr("scpc.etl.scene_pipeline._fetch_latest_yearweek", _yearweek)

    main(
        [
            "--scene",
            "LLMScene",
            "--mk",
            "JP",
            "--llm-only",
            "--outputs-dir",
            str(outputs_dir),
        ]
    )

    assert called["summarize"] is True
    assert called["yearweek"] is True

    yearweek_dir = outputs_dir / "LLMScene" / "JP" / "202412"
    artifact = yearweek_dir / "scene_summary.json"
    assert artifact.exists()

    logs = sorted(log_dir.glob("scene_pipeline_*.log"))
    assert logs, "expected log file to be created"
    log_contents = logs[-1].read_text(encoding="utf-8")
    assert "call=summarize_scene" in log_contents


def test_main_llm_only_emits_json_with_fallback_yearweek(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "scene"
    log_dir = tmp_path / "logs"

    monkeypatch.setenv("SCPC_LOG_DIR", str(log_dir))

    def _unexpected(*_args, **_kwargs):  # pragma: no cover - ensures we skip ETL
        raise AssertionError("run_scene_pipeline should not be invoked when --llm-only is set")

    monkeypatch.setattr("scpc.etl.scene_pipeline.run_scene_pipeline", _unexpected)

    dummy_engine = DummyEngine()
    monkeypatch.setattr("scpc.etl.scene_pipeline.create_doris_engine", lambda: dummy_engine)

    summary = {"status": "OK", "drivers": [], "insufficient_data": False}

    monkeypatch.setattr("scpc.etl.scene_pipeline.summarize_scene", lambda **_kwargs: summary)
    monkeypatch.setattr("scpc.etl.scene_pipeline._fetch_latest_yearweek", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scpc.etl.scene_pipeline._current_yearweek", lambda: 202450)

    main(
        [
            "--scene",
            "Fallback",
            "--mk",
            "UK",
            "--llm-only",
            "--outputs-dir",
            str(outputs_dir),
        ]
    )

    yearweek_dir = outputs_dir / "Fallback" / "UK" / "202450"
    artifact = yearweek_dir / "scene_summary.json"
    assert artifact.exists()
    assert json.loads(artifact.read_text(encoding="utf-8")) == summary

    logs = sorted(log_dir.glob("scene_pipeline_*.log"))
    assert logs, "expected log file to be created"
    log_contents = logs[-1].read_text(encoding="utf-8")
    assert "reason=no_yearweek" in log_contents

def test_main_exits_gracefully_on_operational_error(tmp_path, monkeypatch):
    dummy_engine = DummyEngine()
    monkeypatch.setenv("DORIS_HOST", "183.6.106.112")
    monkeypatch.setenv("DORIS_PORT", "19030")
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SCPC_LOG_DIR", str(log_dir))
    monkeypatch.setattr("scpc.etl.scene_pipeline.create_doris_engine", lambda: dummy_engine)

    os_error = OSError(101, "Network is unreachable")
    error = OperationalError("CONNECT", None, os_error)

    def _raise(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("scpc.etl.scene_pipeline.run_scene_pipeline", _raise)

    with pytest.raises(SystemExit) as excinfo:
        main(["--scene", "X", "--mk", "US", "--weeks-back", "1", "--write"])

    message = str(excinfo.value)
    assert "Failed to connect to Doris at 183.6.106.112:19030" in message
    assert "Network is unreachable" in message
    assert dummy_engine.disposed

    logs = sorted(log_dir.glob("scene_pipeline_*.log"))
    assert logs, "expected log file to be created"
    log_contents = logs[-1].read_text(encoding="utf-8")
    assert "call=run_scene_pipeline" in log_contents
