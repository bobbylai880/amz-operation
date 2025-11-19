from __future__ import annotations

import json
from pathlib import Path

from scpc.llm.deepseek_client import DeepSeekResponse
from scpc.reports import scene_traffic_report as report_module
from scpc.reports.scene_traffic_report import (
    SceneTrafficReportGenerator,
    SceneTrafficReportParams,
)
from scpc.settings import DeepSeekSettings


class StubClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def generate(self, prompt, facts, **kwargs):  # type: ignore[override]
        call_idx = len(self.calls)
        self.calls.append({"prompt": prompt, "facts": facts, **kwargs})
        content = self.responses[call_idx]
        return DeepSeekResponse(content=content, usage={})

    def close(self) -> None:
        self.closed = True


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _params(tmp_path: Path) -> SceneTrafficReportParams:
    return SceneTrafficReportParams(
        week="2025-W45",
        scene_tag="浴室袋",
        marketplace_id="US",
        storage_dir=tmp_path,
    )


def _generator(client: StubClient) -> SceneTrafficReportGenerator:
    settings = DeepSeekSettings(
        base_url="https://example.com",
        model="deepseek-test",
        api_key="sk-test",
        timeout=30.0,
    )
    return SceneTrafficReportGenerator(client=client, settings=settings)


def _prepare_flow_payload() -> dict[str, object]:
    return {
        "week": "2025-W45",
        "scene_tag": "浴室袋",
        "marketplace_id": "US",
        "monday_this": "2025-11-03",
        "monday_last": "2025-10-27",
        "overview": {
            "self": {
                "asin_count": 2,
                "avg_ad_flow_share_this": 0.42,
                "avg_ad_flow_share_last": 0.35,
                "avg_organic_flow_share_this": 0.33,
                "avg_reco_flow_share_this": 0.25,
                "avg_sp_flow_share_this": 0.18,
                "avg_video_flow_share_this": 0.09,
                "avg_brand_flow_share_this": 0.06,
            },
            "competitor": {
                "asin_count": 3,
                "avg_ad_flow_share_this": 0.5,
                "avg_ad_flow_share_last": 0.48,
                "avg_organic_flow_share_this": 0.32,
                "avg_reco_flow_share_this": 0.18,
                "avg_sp_flow_share_this": 0.22,
                "avg_video_flow_share_this": 0.07,
                "avg_brand_flow_share_this": 0.04,
            },
        },
        "ad_change_distribution": {"广告占比大幅提升": 1},
        "top_lists": {
            "self": {
                "ad_increase_top": [
                    {
                        "asin": "B0SELF1",
                        "brand": "Attmu",
                        "hyy_asin": 1,
                        "ad_flow_share_this": 0.6,
                        "ad_flow_share_last": 0.3,
                        "ad_flow_share_diff": 0.3,
                        "organic_flow_share_this": 0.3,
                        "reco_flow_share_this": 0.1,
                        "sp_flow_share_this": 0.2,
                        "video_flow_share_this": 0.05,
                        "brand_flow_share_this": 0.04,
                        "ad_change_type": "广告占比大幅提升",
                        "traffic_mix_type": "广告主导型流量",
                    }
                ],
                "ad_decrease_top": [],
                "ad_heavy_this": [],
                "organic_heavy_this": [],
            },
            "competitor": {
                "ad_increase_top": [],
                "ad_decrease_top": [
                    {
                        "asin": "B0COMP1",
                        "brand": "LuxBrand",
                        "hyy_asin": 0,
                        "ad_flow_share_this": 0.35,
                        "ad_flow_share_last": 0.5,
                        "ad_flow_share_diff": -0.15,
                        "organic_flow_share_this": 0.4,
                        "reco_flow_share_this": 0.25,
                        "sp_flow_share_this": 0.15,
                        "video_flow_share_this": 0.1,
                        "brand_flow_share_this": 0.02,
                        "ad_change_type": "广告占比大幅下降",
                        "traffic_mix_type": "自然主导型流量",
                    }
                ],
                "ad_heavy_this": [],
                "organic_heavy_this": [],
            },
        },
    }


def _prepare_keyword_payload() -> dict[str, object]:
    return {
        "week": "2025-W45",
        "scene_tag": "浴室袋",
        "marketplace_id": "US",
        "sunday_this": "2025-11-09",
        "sunday_last": "2025-11-02",
        "scene_head_keywords": {
            "this_week": [
                {
                    "keyword": "dorm shower caddy",
                    "scene_kw_share_this": 0.2,
                    "scene_kw_self_share_this": 0.12,
                    "scene_kw_comp_share_this": 0.08,
                    "search_volume_this": 120000,
                    "rank_this": 1,
                }
            ],
            "last_week": [
                {
                    "keyword": "travel shower caddy",
                    "scene_kw_share_last": 0.18,
                    "scene_kw_self_share_last": 0.11,
                    "scene_kw_comp_share_last": 0.07,
                    "search_volume_last": 95000,
                    "rank_last": 1,
                }
            ],
            "diff": {
                "keywords_added": [
                    {
                        "keyword": "camping shower caddy",
                        "scene_kw_share_this": 0.07,
                        "search_volume_this": 80000,
                    }
                ],
                "keywords_removed": [
                    {
                        "keyword": "boho shower caddy",
                        "scene_kw_share_last": 0.05,
                        "search_volume_last": 30000,
                    }
                ],
                "keywords_common": [
                    {
                        "keyword": "dorm shower caddy",
                        "scene_kw_share_this": 0.2,
                        "scene_kw_share_last": 0.19,
                        "scene_kw_self_share_this": 0.12,
                        "scene_kw_self_share_last": 0.11,
                        "scene_kw_comp_share_this": 0.08,
                        "scene_kw_comp_share_last": 0.08,
                        "search_volume_this": 120000,
                        "search_volume_last": 100000,
                        "search_volume_diff": 20000,
                        "search_volume_change_rate": 0.2,
                    }
                ],
            },
        },
        "asin_keyword_profile_change": {
            "self": [
                {
                    "asin": "B0SELF1",
                    "brand": "Attmu",
                    "hyy_asin": 1,
                    "change_score": 0.72,
                    "change_type": "关键词画像变化显著",
                    "head_keywords_this": [
                        {
                            "keyword": "dorm shower caddy",
                            "share": 0.5,
                            "search_volume_this": 120000,
                        }
                    ],
                    "head_keywords_last": [
                        {
                            "keyword": "travel shower caddy",
                            "share": 0.4,
                            "search_volume_last": 95000,
                        }
                    ],
                    "keywords_added": ["dorm shower caddy"],
                    "keywords_removed": ["travel shower caddy"],
                }
            ],
            "competitor": [
                {
                    "asin": "B0COMP1",
                    "brand": "LuxBrand",
                    "hyy_asin": 0,
                    "change_score": 0.65,
                    "change_type": "关键词画像变化显著",
                    "head_keywords_this": [
                        {
                            "keyword": "camping shower caddy",
                            "share": 0.3,
                            "search_volume_this": 80000,
                        }
                    ],
                    "head_keywords_last": [
                        {
                            "keyword": "bathroom caddy",
                            "share": 0.2,
                            "search_volume_last": 40000,
                        }
                    ],
                    "keywords_added": ["camping shower caddy"],
                    "keywords_removed": ["bathroom caddy"],
                }
            ],
        },
        "keyword_asin_contributors": {
            "this_week": [
                {
                    "keyword": "dorm shower caddy",
                    "top_asin": [
                        {
                            "asin": "B0SELF1",
                            "brand": "Attmu",
                            "hyy_asin": 1,
                            "effective_impr_share_this": 0.2,
                        },
                        {
                            "asin": "B0COMP1",
                            "brand": "LuxBrand",
                            "hyy_asin": 0,
                            "effective_impr_share_this": 0.15,
                        },
                    ],
                }
            ]
        },
        "keyword_opportunity_by_volume": {
            "high_volume_low_self": [
                {
                    "keyword": "camping shower caddy",
                    "search_volume_this": 80000,
                    "search_volume_last": 60000,
                    "search_volume_change_rate": 0.33,
                    "scene_kw_share_this": 0.07,
                    "scene_kw_self_share_this": 0.01,
                    "scene_kw_comp_share_this": 0.06,
                }
            ],
            "rising_demand_self_lagging": [
                {
                    "keyword": "travel shower caddy",
                    "search_volume_this": 95000,
                    "search_volume_last": 60000,
                    "search_volume_change_rate": 0.58,
                    "scene_kw_share_this": 0.18,
                    "scene_kw_self_share_this": 0.03,
                    "scene_kw_self_share_last": 0.02,
                }
            ],
        },
    }


def test_generator_writes_both_chapters(tmp_path: Path) -> None:
    params = _params(tmp_path)
    base_dir = tmp_path / params.week / params.scene_tag / "traffic"
    flow_payload = _prepare_flow_payload()
    keyword_payload = _prepare_keyword_payload()
    _write_json(base_dir / "flow_change.json", flow_payload)
    _write_json(base_dir / "keyword_change.json", keyword_payload)

    responses = [
        "# 六、场景流量结构与投放策略\n\n本周流量结构描述。",
        "# 七、搜索需求与关键词机会\n\n关键词描述。",
    ]
    client = StubClient(responses)
    generator = _generator(client)

    outputs = generator.run(params)

    flow_md = outputs["flow"]
    keyword_md = outputs["keyword"]
    assert flow_md.read_text(encoding="utf-8").startswith("# 六、场景流量结构与投放策略")
    assert keyword_md.read_text(encoding="utf-8").startswith("# 七、搜索需求与关键词机会")
    assert flow_md.parent == keyword_md.parent
    assert client.closed is True
    assert len(client.calls) == 2
    assert client.calls[0]["prompt"] == report_module.FLOW_SYSTEM_PROMPT
    assert client.calls[1]["prompt"] == report_module.KEYWORD_SYSTEM_PROMPT
    assert client.calls[0]["facts"]["chapter"] == report_module.FLOW_TITLE
    assert client.calls[1]["facts"]["chapter"] == report_module.KEYWORD_TITLE
    keyword_facts = client.calls[1]["facts"]
    assert keyword_facts["scene_head_keywords"] == keyword_payload["scene_head_keywords"]
    assert (
        keyword_facts["keyword_opportunity_by_volume"]
        == keyword_payload["keyword_opportunity_by_volume"]
    )
    assert (
        keyword_facts["keyword_asin_contributors"]
        == keyword_payload["keyword_asin_contributors"]
    )
    assert (
        keyword_facts["asin_keyword_profile_change"]
        == keyword_payload["asin_keyword_profile_change"]
    )
    flags = keyword_facts["data_quality_flags"]
    assert flags["rank_metrics_missing"] is True
    assert flags["keyword_opportunity_missing"] is False
    assert flags["asin_contributor_missing"] is False
    assert flags["search_volume_last_missing"] is False
    assert "keyword_change_json" not in keyword_facts
    assert keyword_facts["style_notes"].get("no_code_block") is True


def test_generator_handles_missing_flow_json(tmp_path: Path) -> None:
    params = _params(tmp_path)
    base_dir = tmp_path / params.week / params.scene_tag / "traffic"
    _write_json(base_dir / "keyword_change.json", _prepare_keyword_payload())

    responses = ["# 七、搜索需求与关键词机会\n"]
    client = StubClient(responses)
    generator = _generator(client)

    outputs = generator.run(params)

    flow_md = outputs["flow"].read_text(encoding="utf-8")
    assert "flow_change.json" in flow_md
    keyword_md = outputs["keyword"].read_text(encoding="utf-8")
    assert keyword_md.startswith("# 七、搜索需求与关键词机会")
    assert len(client.calls) == 1


def test_generator_handles_missing_keyword_json(tmp_path: Path) -> None:
    params = _params(tmp_path)
    base_dir = tmp_path / params.week / params.scene_tag / "traffic"
    _write_json(base_dir / "flow_change.json", _prepare_flow_payload())

    responses = ["# 六、场景流量结构与投放策略\n"]
    client = StubClient(responses)
    generator = _generator(client)

    outputs = generator.run(params)

    flow_md = outputs["flow"].read_text(encoding="utf-8")
    assert flow_md.startswith("# 六、场景流量结构与投放策略")
    keyword_md = outputs["keyword"].read_text(encoding="utf-8")
    assert "keyword_change.json" in keyword_md
    assert len(client.calls) == 1

