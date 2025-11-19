from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpc.llm.deepseek_client import DeepSeekError, DeepSeekResponse
from scpc.reports.weekly_scene_report import (
    WeeklySceneReportError,
    WeeklySceneReportGenerator,
    WeeklySceneReportParams,
)
from scpc.settings import DeepSeekSettings


class StubClient:
    def __init__(self, responses: list[str], *, fail_at: int | None = None) -> None:
        self.responses = responses
        self.fail_at = fail_at
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def generate(self, prompt, facts, **kwargs):  # type: ignore[override]
        call_idx = len(self.calls)
        if self.fail_at is not None and call_idx + 1 == self.fail_at:
            raise DeepSeekError("boom")
        self.calls.append({"prompt": prompt, "facts": facts, **kwargs})
        content = self.responses[call_idx]
        return DeepSeekResponse(content=content, usage={})

    def close(self) -> None:
        self.closed = True


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _params(tmp_path: Path) -> WeeklySceneReportParams:
    return WeeklySceneReportParams(
        week="2025-W45",
        scene_tag="浴室袋",
        marketplace_id="US",
        storage_dir=tmp_path,
    )


def _prepare_inputs(tmp_path: Path) -> Path:
    base = tmp_path / "2025-W45" / "浴室袋"
    base.mkdir(parents=True, exist_ok=True)
    _write_json(base / "overall_summary.json", {"week": "2025-W45"})
    _write_json(base / "self_analysis.json", {"overview": {}})
    _write_json(base / "competitor_analysis.json", {"overview": {}})
    _write_json(base / "self_risk_opportunity.json", {"rules": "demo", "risk_asin": []})
    _write_json(base / "competitor_actions.json", {"price_and_rank_moves": []})
    return base


def _prepare_slug_inputs(tmp_path: Path) -> Path:
    base = tmp_path / "2025-W45" / "scene"
    base.mkdir(parents=True, exist_ok=True)
    _write_json(base / "overall_summary.json", {"week": "2025-W45"})
    _write_json(base / "self_analysis.json", {"overview": {}})
    _write_json(base / "competitor_analysis.json", {"overview": {}})
    _write_json(base / "self_risk_opportunity.json", {"rules": "demo", "risk_asin": []})
    _write_json(base / "competitor_actions.json", {"price_and_rank_moves": []})
    return base


def _generator(client: StubClient) -> WeeklySceneReportGenerator:
    settings = DeepSeekSettings(
        base_url="https://example.com",
        model="deepseek-stub",
        api_key="sk-test",
        timeout=30.0,
    )
    return WeeklySceneReportGenerator(client=client, settings=settings)


def test_generator_produces_markdown_files(tmp_path: Path) -> None:
    base = _prepare_inputs(tmp_path)
    responses = [
        "# overall",
        "# self",
        "# competitor",
        "# risk",
        "# actions",
        "# full",
    ]
    client = StubClient(responses)
    generator = _generator(client)

    outputs = generator.run(_params(tmp_path))

    report_dir = base / "reports"
    expected_files = {
        "overall_summary": report_dir / "01_overall_summary.md",
        "self_analysis": report_dir / "02_self_analysis.md",
        "competitor_analysis": report_dir / "03_competitor_analysis.md",
        "self_risk_opportunity": report_dir / "04_self_risk_opportunity.md",
        "competitor_actions": report_dir / "05_competitor_actions.md",
        "traffic_flow": report_dir / "06_traffic_flow.md",
        "keyword_opportunity": report_dir / "07_keyword_opportunity.md",
        "full_report": report_dir / "00_full_report.md",
    }
    assert outputs == expected_files
    for path in expected_files.values():
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("# ")
    assert client.closed is True
    assert len(client.calls) == 6
    modules = client.calls[-1]["facts"]["modules"]
    assert {
        "overall_summary",
        "self_analysis",
        "competitor_analysis",
        "self_risk_opportunity",
        "competitor_actions",
        "traffic_flow",
        "keyword_opportunity",
    }.issubset(modules.keys())
    assert "数据缺失" in modules["traffic_flow"]


def test_generator_raises_when_json_missing(tmp_path: Path) -> None:
    base = tmp_path / "2025-W45" / "浴室袋"
    base.mkdir(parents=True, exist_ok=True)
    _write_json(base / "overall_summary.json", {"week": "2025-W45"})
    client = StubClient(["# overall"])
    generator = _generator(client)

    with pytest.raises(WeeklySceneReportError):
        generator.run(_params(tmp_path))


def test_generator_wraps_deepseek_errors(tmp_path: Path) -> None:
    _prepare_inputs(tmp_path)
    responses = ["# overall"] * 6
    client = StubClient(responses, fail_at=2)
    generator = _generator(client)

    with pytest.raises(WeeklySceneReportError):
        generator.run(_params(tmp_path))


def test_generator_supports_slugged_scene_directory(tmp_path: Path) -> None:
    base = _prepare_slug_inputs(tmp_path)
    responses = ["# overall", "# self", "# competitor", "# risk", "# actions", "# full"]
    client = StubClient(responses)
    generator = _generator(client)

    outputs = generator.run(_params(tmp_path))

    report_dir = base / "reports"
    assert outputs["overall_summary"].parent == report_dir
    assert outputs["traffic_flow"].exists()
    assert outputs["keyword_opportunity"].exists()
    assert report_dir.exists()


def test_generator_reads_optional_markdown_when_present(tmp_path: Path) -> None:
    base = _prepare_inputs(tmp_path)
    report_dir = base / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    flow_text = "# 六、场景流量结构与投放策略\n\n现有内容"
    keyword_text = "# 七、搜索需求与关键词机会\n\n关键词内容"
    (report_dir / "06_traffic_flow.md").write_text(flow_text, encoding="utf-8")
    (report_dir / "07_keyword_opportunity.md").write_text(
        keyword_text, encoding="utf-8"
    )
    responses = ["# overall", "# self", "# competitor", "# risk", "# actions", "# full"]
    client = StubClient(responses)
    generator = _generator(client)

    generator.run(_params(tmp_path))

    modules = client.calls[-1]["facts"]["modules"]
    assert modules["traffic_flow"].startswith(flow_text)
    assert modules["keyword_opportunity"].startswith(keyword_text)
    assert (report_dir / "06_traffic_flow.md").read_text(encoding="utf-8").startswith("# 六")
    assert (report_dir / "07_keyword_opportunity.md").read_text(encoding="utf-8").startswith("# 七")


def test_generator_raises_when_optional_file_unreadable(tmp_path: Path) -> None:
    base = _prepare_inputs(tmp_path)
    report_dir = base / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    flow_path = report_dir / "06_traffic_flow.md"
    flow_path.write_bytes(b"\xff\xfe")
    responses = ["# overall", "# self", "# competitor", "# risk", "# actions", "# full"]
    client = StubClient(responses)
    generator = _generator(client)

    with pytest.raises(WeeklySceneReportError):
        generator.run(_params(tmp_path))

