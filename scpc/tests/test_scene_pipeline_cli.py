import pandas as pd

from scpc.etl.scene_pipeline import main, parse_args


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


def test_main_emits_scene_summary_json(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "scene"
    features = pd.DataFrame({"year": [2024], "week_num": [12], "start_date": [pd.Timestamp("2024-03-17")]})
    outputs = {"clean": pd.DataFrame(), "features": features, "drivers": pd.DataFrame()}

    monkeypatch.setattr("scpc.etl.scene_pipeline.run_scene_pipeline", lambda *args, **kwargs: outputs)
    monkeypatch.setattr("scpc.etl.scene_pipeline.create_doris_engine", lambda: DummyEngine())
    summary = {"status": "OK", "drivers": [], "insufficient_data": False}
    monkeypatch.setattr("scpc.etl.scene_pipeline.summarize_scene", lambda **kwargs: summary)

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


def test_main_emits_scene_markdown(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "scene"
    features = pd.DataFrame({"year": [2023], "week_num": [5], "start_date": [pd.Timestamp("2023-02-05")]})
    outputs = {"clean": pd.DataFrame(), "features": features, "drivers": pd.DataFrame()}

    monkeypatch.setattr("scpc.etl.scene_pipeline.run_scene_pipeline", lambda *args, **kwargs: outputs)
    monkeypatch.setattr("scpc.etl.scene_pipeline.create_doris_engine", lambda: DummyEngine())
    summary = {"status": "OK", "drivers": [], "insufficient_data": False}
    monkeypatch.setattr("scpc.etl.scene_pipeline.summarize_scene", lambda **kwargs: summary)
    monkeypatch.setattr("scpc.etl.scene_pipeline.build_scene_markdown", lambda payload: "# Demo\n")

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
