"""LLM powered generation of the traffic focused report chapters."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from scpc.llm.deepseek_client import DeepSeekClient, DeepSeekError, DeepSeekResponse
from scpc.llm.deepseek_client import create_client_from_env
from scpc.settings import DeepSeekSettings, get_deepseek_settings

LOGGER = logging.getLogger(__name__)

FLOW_OUTPUT_FILENAME = "06_traffic_flow.md"
KEYWORD_OUTPUT_FILENAME = "07_keyword_opportunity.md"
FLOW_TITLE = "六、场景流量结构与投放策略"
KEYWORD_TITLE = "七、搜索需求与关键词机会"

FLOW_SYSTEM_PROMPT = (
    "你是一名资深的亚马逊运营负责人（Category Manager），负责某一品类在单一站点的经营。"
    " 你熟悉 SP/SB/SBV 广告位、自然流量、推荐流量、排名与竞品打法，并会严格引用输入 JSON 的事实。"
    " 当引用 ASIN 时务必带上品牌信息（如“品牌（ASIN: XXX）”），如果品牌缺失请写“品牌未知（ASIN: XXX）”。"
    " 若输入中缺少上一周数据，需要在报告中说明“数据缺失”，禁止臆测。输出必须为中文 Markdown。"
)

KEYWORD_SYSTEM_PROMPT = (
    "你是一名资深的亚马逊运营负责人，接收的 JSON facts 已拆分为 context、scene_head_keywords、keyword_opportunity_by_volume、keyword_asin_contributors、asin_keyword_profile_change、data_quality_flags、output_requirements、style_notes。"
    " 你必须严格依赖这些字段写出第七章《搜索需求与关键词机会》，禁止臆造任何关键词、ASIN、品牌或数值。"
    " 【结构要求】输出以“# 七、搜索需求与关键词机会”开头，并按照 7.1~7.4 固定顺序书写；任一小节若缺数据，必须写“数据缺失，仅能做静态观察”。"
    " 【数据使用】search_volume_change_rate 判定需求趋势：≥ +0.3=明显上升，+0.1~+0.3=略有上升，-0.1~+0.1=基本稳定，≤ -0.1=需求走弱；scene_kw_self_share/scene_kw_comp_share 与上一周差值 ≥ ±0.03 必须描述份额流入或流失。"
    " 排名变化只能依据 *_best_organic_rank_*/ *_best_ad_rank_* 及 organic_rank_trend/ad_rank_trend/organic_rank_diff/ad_rank_diff（如 new/lost/up/down/stable），不得自行推算；若某字段为 null 或 status=missing，必须在文中说明“数据缺失”。"
    " 【7.1】引用 scene_head_keywords.this_week / last_week / diff 中的搜索量、share、rank_this/last 与 *_status，结合 data_quality_flags 标注上一周缺失场景时写“仅能做静态观察”。"
    " 【7.2】按照 keyword_opportunity_by_volume.*.opportunity_type 分组（如 high_volume_low_self、rising_demand_self_lagging、organic_good_ad_gap、ad_heavy_but_not_dominant），逐组回答“搜索需求→曝光结构→排名格局”，并引用自营/竞品 share 与 rank/status；若某分组列表为空需声明“该类机会暂无数据”。"
    " 【7.3】为 7.2 中的每个重点关键词列出 ≥3 个 ASIN，必须来自 keyword_asin_contributors.this_week.top_asin，按曝光贡献（effective_impr_share_this，可称 kw_impr_share）排序，自营优先；引用格式固定为“自营/竞品 品牌（ASIN: XXXXX）”，品牌缺失写“品牌未知（ASIN: XXXXX）”。"
    " 当 ASIN 排名字段 organic_rank_this/ad_rank_this 缺失时写“该词下自然/广告排位数据缺失，仅能根据曝光 share 评估”；若 organic_rank_trend/ad_rank_trend=up/down/new/lost/stable，要翻译成通俗语言说明排名上升/下滑/新增/跌出；若 ASIN 也在 asin_keyword_profile_change 中出现，需要补充其 head_keywords_this/last 或 keywords_added 透露的画像迁移。"
    " 【7.4】按 opportunity_type 总结可执行动作，点名 keyword+ASIN 组合，并说明需要的资源（Listing 优化/广告/新品）；若前序数据缺失导致无法给出策略，要明确说明原因。"
    " 【其他】必须参考 data_quality_flags 提示的数据空缺，Markdown 内禁止代码块；引用 ASIN/关键词均需来自输入 JSON；语气保持专业、分点展开。"
)


@dataclass(slots=True)
class SceneTrafficReportParams:
    """Parameters required to render the traffic chapters."""

    week: str
    scene_tag: str
    marketplace_id: str
    storage_dir: Path


class SceneTrafficReportError(RuntimeError):
    """Raised when the traffic Markdown report cannot be produced."""


class SceneTrafficReportGenerator:
    """Create Markdown chapters focused on flow and keyword movements."""

    def __init__(
        self,
        *,
        client: DeepSeekClient | None = None,
        settings: DeepSeekSettings | None = None,
        temperature: float = 0.2,
    ) -> None:
        if settings is None:
            settings = get_deepseek_settings()
        self._settings = settings
        self._client = client or create_client_from_env(settings=settings)
        self._temperature = temperature

    def run(self, params: SceneTrafficReportParams) -> dict[str, Path]:
        """Generate both Markdown chapters (flow + keyword)."""

        metadata = {
            "week": params.week,
            "scene_tag": params.scene_tag,
            "marketplace_id": params.marketplace_id,
        }
        base_dir = self._resolve_scene_dir(
            params.storage_dir, params.week, params.scene_tag
        )
        traffic_dir = base_dir / "traffic"
        report_dir = base_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "scene_traffic_report_start",
            extra={
                "week": params.week,
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "storage": str(params.storage_dir),
            },
        )

        outputs: dict[str, Path] = {}
        try:
            outputs["flow"] = self._process_flow_section(
                traffic_dir / "flow_change.json",
                report_dir / FLOW_OUTPUT_FILENAME,
                metadata,
            )
            outputs["keyword"] = self._process_keyword_section(
                traffic_dir / "keyword_change.json",
                report_dir / KEYWORD_OUTPUT_FILENAME,
                metadata,
            )
        finally:
            self._client.close()

        LOGGER.info(
            "scene_traffic_report_done",
            extra={
                "week": params.week,
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "flow_path": str(outputs.get("flow")),
                "keyword_path": str(outputs.get("keyword")),
            },
        )
        return outputs

    # ------------------------------------------------------------------
    # Flow section
    # ------------------------------------------------------------------
    def _process_flow_section(
        self,
        json_path: Path,
        output_path: Path,
        metadata: Mapping[str, str],
    ) -> Path:
        try:
            flow_data = self._load_json(json_path)
        except SceneTrafficReportError as exc:
            LOGGER.error(
                "scene_traffic_report_flow_missing",
                extra={"path": str(json_path), "error": str(exc)},
            )
            return self._write_placeholder(
                output_path,
                FLOW_TITLE,
                "由于本周缺少流量结构数据（flow_change.json 未生成或解析失败），本章暂无法给出分析。",
            )

        facts = self._build_flow_facts(metadata, flow_data)
        content = self._render_section(FLOW_TITLE, FLOW_SYSTEM_PROMPT, facts)
        output_path.write_text(content, encoding="utf-8")
        LOGGER.info(
            "scene_traffic_report_flow_written",
            extra={"path": str(output_path), "bytes": output_path.stat().st_size},
        )
        return output_path

    def _build_flow_facts(
        self, metadata: Mapping[str, str], flow_data: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        overview = flow_data.get("overview") or {}
        self_overview = overview.get("self") or {}
        competitor_overview = overview.get("competitor") or {}
        flags = {
            "monday_last_missing": _is_missing_value(flow_data.get("monday_last")),
            "self_last_missing": _is_missing_value(
                self_overview.get("avg_ad_flow_share_last")
            ),
            "competitor_last_missing": _is_missing_value(
                competitor_overview.get("avg_ad_flow_share_last")
            ),
        }
        output_requirements = [
            "一级标题必须为“六、场景流量结构与投放策略”。",
            "依次包含 6.1~6.4 四个小节，可根据语义轻微调整小节标题。",
            "6.1 说明场景整体广告/自然/推荐流量结构的本周 vs 上周变化；若 data_quality_flags 指示上一周缺失，需要写明“数据缺失”。",
            "6.2 明确自营 vs 竞品在广告依赖度、自然位依赖度、视频/品牌广告投入的差异。",
            "6.3 结合 top_lists 中的代表 ASIN，描述广告占比提升/下降、广告主导型、自然主导型等典型打法。",
            "6.4 给出下周投放策略建议，至少 3 条，并点名需要加减投放的自营 ASIN（带品牌）。",
            "每次引用 ASIN 必须写成“品牌（ASIN: xxx）”，品牌缺失时使用“品牌未知（ASIN: xxx）”。",
            "禁止粘贴原始 JSON，所有数值引用需来自 flow_change_json，可用“约 XX%”的描述。",
            "若 JSON 提示某一类数据缺失或某周为空，只能做静态分析，不得臆测环比。",
        ]
        return {
            "chapter": FLOW_TITLE,
            "context": metadata,
            "flow_change_json": flow_data,
            "json_notes": {
                "overview": "overview.self/competitor 提供广告/自然/推荐占比均值，ad_change_distribution 汇总广告变化桶。",
                "top_lists": "top_lists.* 列出广告占比大幅变化、广告主导/自然主导 ASIN，包含自营与竞品。",
            },
            "data_quality_flags": flags,
            "output_requirements": output_requirements,
            "style_notes": {
                "tone": "先结论后解释，最后落到可执行动作。",
                "length": "全文建议 400-800 字。",
            },
        }

    # ------------------------------------------------------------------
    # Keyword section
    # ------------------------------------------------------------------
    def _process_keyword_section(
        self,
        json_path: Path,
        output_path: Path,
        metadata: Mapping[str, str],
    ) -> Path:
        try:
            keyword_data = self._load_json(json_path)
        except SceneTrafficReportError as exc:
            LOGGER.error(
                "scene_traffic_report_keyword_missing",
                extra={"path": str(json_path), "error": str(exc)},
            )
            return self._write_placeholder(
                output_path,
                KEYWORD_TITLE,
                "由于本周缺少关键词结构数据（keyword_change.json 未生成或解析失败），本章暂无法给出分析。",
            )

        facts = self._build_keyword_facts(metadata, keyword_data)
        content = self._render_section(KEYWORD_TITLE, KEYWORD_SYSTEM_PROMPT, facts)
        output_path.write_text(content, encoding="utf-8")
        LOGGER.info(
            "scene_traffic_report_keyword_written",
            extra={"path": str(output_path), "bytes": output_path.stat().st_size},
        )
        return output_path

    def _build_keyword_facts(
        self, metadata: Mapping[str, str], keyword_data: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        scene_keywords = keyword_data.get("scene_head_keywords") or {}
        this_keywords = scene_keywords.get("this_week") or []
        last_keywords = scene_keywords.get("last_week") or []
        opportunity = keyword_data.get("keyword_opportunity_by_volume") or {}
        contributors = keyword_data.get("keyword_asin_contributors") or {}
        asin_profile_change = keyword_data.get("asin_keyword_profile_change") or {}

        def _has_volume(entries: Iterable[Mapping[str, Any]], field: str) -> bool:
            for entry in entries:
                value = entry.get(field)
                if value is not None and str(value).strip() != "":
                    return True
            return False

        def _has_fields(entries: Iterable[Mapping[str, Any]], fields: list[str]) -> bool:
            for entry in entries:
                for field in fields:
                    value = entry.get(field)
                    if value is not None and str(value).strip() != "":
                        return True
            return False

        def _has_entries(node: Any) -> bool:
            if isinstance(node, Mapping):
                return any(_has_entries(value) for value in node.values())
            if isinstance(node, list):
                return any(node)
            return bool(node)

        def _has_top_asin(entries: Iterable[Mapping[str, Any]]) -> bool:
            for entry in entries:
                top_asin = entry.get("top_asin") or []
                if top_asin:
                    return True
            return False

        top_asin_block = contributors.get("this_week") or []
        flags = {
            "sunday_last_missing": _is_missing_value(keyword_data.get("sunday_last")),
            "last_week_pool_missing": not bool(scene_keywords.get("last_week")),
            "search_volume_this_missing": not _has_volume(
                this_keywords, "search_volume_this"
            ),
            "search_volume_last_missing": bool(last_keywords)
            and not _has_volume(last_keywords, "search_volume_last"),
            "rank_metrics_missing": not _has_fields(
                this_keywords,
                [
                    "self_best_organic_rank_this",
                    "self_best_ad_rank_this",
                    "comp_best_organic_rank_this",
                    "comp_best_ad_rank_this",
                ],
            ),
            "keyword_opportunity_missing": not _has_entries(opportunity),
            "asin_contributor_missing": not _has_top_asin(top_asin_block),
            "asin_profile_change_missing": not _has_entries(asin_profile_change),
        }
        output_requirements = [
            "一级标题必须为“七、搜索需求与关键词机会”。",
            "全文固定包含 7.1~7.4 小节，即便缺数据也要写“数据缺失，仅能做静态观察”。",
            "所有趋势判断必须直接引用 search_volume_change_rate、scene_kw_share/scene_kw_self_share/scene_kw_comp_share 的 this/last/diff 以及 organic_rank_trend/ad_rank_trend/organic_rank_diff/ad_rank_diff，禁止自行推算。",
            "若 data_quality_flags 提示上一周缺失、搜索量缺失或 rank 数据缺失，需在对应段落中明确告知并仅做静态描述。",
            "7.1 需结合 scene_head_keywords.this_week/last_week/diff 说明搜索需求层级、场景份额、自营 vs 竞品 share 与自然/广告位状态。",
            "7.2 必须按照 keyword_opportunity_by_volume.*.opportunity_type 分组，若某组为空需写“该类机会暂无数据”，每组内部按“搜索需求→曝光结构→自然/广告排位”阐述。",
            "7.3 针对 7.2 的每个重点关键词列出至少 3 个 ASIN，均来自 keyword_asin_contributors.this_week.top_asin，描述曝光贡献（effective_impr_share_this）、自营/竞品身份、自然/广告排名及 trend（缺失需写“数据缺失”），并补充 asin_keyword_profile_change.* 中的画像迁移（如适用）。",
            "引用 ASIN 时必须写“自营/竞品 品牌（ASIN: XXXXX）”，品牌缺失写“品牌未知（ASIN: XXXXX）”，并说明若缺 rank/status 只能依据曝光 share。",
            "7.4 需按 opportunity_type 汇总下一步动作，点名 keyword+ASIN 组合及需要的资源（Listing 优化/广告/上新等），引用前文数据作为证据。",
            "Markdown 文本禁止使用代码块，禁止编造 JSON 中不存在的关键词、ASIN 或品牌。",
        ]
        return {
            "chapter": KEYWORD_TITLE,
            "context": metadata,
            "scene_head_keywords": scene_keywords,
            "keyword_opportunity_by_volume": opportunity,
            "keyword_asin_contributors": contributors,
            "asin_keyword_profile_change": asin_profile_change,
            "data_quality_flags": flags,
            "json_notes": {
                "scene_head_keywords": "scene_head_keywords.this_week/last_week/diff 覆盖头部需求池、排名与份额变化。",
                "keyword_opportunity_by_volume": "keyword_opportunity_by_volume.* 携带 search_volume、share、rank/status 与 opportunity_type。",
                "keyword_asin_contributors": "keyword_asin_contributors.this_week.top_asin 提供每个关键词的主要 ASIN 及曝光贡献与排名。",
                "asin_keyword_profile_change": "asin_keyword_profile_change.self/competitor 描述 ASIN 的关键词画像迁移。",
            },
            "output_requirements": output_requirements,
            "style_notes": {
                "tone": "以数据驱动的诊断与建议，逐步收敛到关键词×ASIN 行动。",
                "length": "建议 600-1200 字，可视数据量适度延长。",
                "no_code_block": True,
            },
        }
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _render_section(
        self,
        title: str,
        system_prompt: str,
        facts: Mapping[str, Any],
    ) -> str:
        response = self._invoke_llm(system_prompt, facts)
        content = response.content.strip()
        if not content:
            raise SceneTrafficReportError(f"LLM returned empty content for {title}")
        header = f"# {title}"
        if not content.startswith(header):
            content = f"{header}\n\n{content}"
        if not content.endswith("\n"):
            content += "\n"
        return content

    def _load_json(self, path: Path) -> Mapping[str, Any]:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SceneTrafficReportError(f"Missing JSON file: {path}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise SceneTrafficReportError(f"Invalid JSON structure in {path}") from exc
        LOGGER.info(
            "scene_traffic_report_loaded_json",
            extra={"path": str(path), "bytes": len(raw)},
        )
        if not isinstance(data, Mapping):
            raise SceneTrafficReportError(f"JSON root must be an object: {path}")
        return data

    def _write_placeholder(self, path: Path, title: str, message: str) -> Path:
        content = f"# {title}\n\n> {message}\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _invoke_llm(
        self, system_prompt: str, facts: Mapping[str, Any]
    ) -> DeepSeekResponse:
        try:
            return self._client.generate(
                prompt=system_prompt,
                facts=facts,
                model=self._settings.model,
                temperature=self._temperature,
                response_format="text",
            )
        except DeepSeekError as exc:
            raise SceneTrafficReportError("DeepSeek request failed") from exc

    def _resolve_scene_dir(self, storage: Path, week: str, scene_tag: str) -> Path:
        week_dir = storage / week
        candidates = list(_scene_dir_candidates(scene_tag))
        for candidate in candidates:
            candidate_dir = week_dir / candidate
            traffic_dir = candidate_dir / "traffic"
            if traffic_dir.exists():
                if candidate != candidates[0]:
                    LOGGER.info(
                        "scene_traffic_report_scene_dir_fallback",
                        extra={"week": week, "scene_tag": scene_tag, "selected": candidate},
                    )
                return candidate_dir
        return week_dir / candidates[0]


def _scene_dir_candidates(scene_tag: str) -> Iterable[str]:
    cleaned = scene_tag.strip()
    seen: set[str] = set()
    if cleaned:
        seen.add(cleaned)
        yield cleaned
    slug = _slugify_scene_tag(scene_tag)
    if slug not in seen:
        yield slug


def _slugify_scene_tag(scene_tag: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in scene_tag)
    safe = safe.strip("_")
    return safe or "scene"


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


__all__ = [
    "SceneTrafficReportGenerator",
    "SceneTrafficReportParams",
    "SceneTrafficReportError",
]

