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
    "你是一名资深的亚马逊运营负责人，擅长从关键词与 ASIN 贡献数据拆解用户需求并制定投放策略。"
    " 输入 JSON 至少包含 scene_head_keywords（本周/上周及 diff）、keyword_opportunity_by_volume（high_volume_low_self 与 rising_demand_self_lagging）、keyword_asin_contributors.this_week.top_asin、asin_keyword_profile_change.self/competitor。"
    " 你需要针对每个重点关键词同时回答三件事：1）搜索需求规模与趋势（search_volume_this/last/change_rate）；2）在整个场景内的曝光结构与自营 vs 竞品 share（scene_kw_share_*、scene_kw_self_share_*、scene_kw_comp_share_*）；3）自营/竞品在该词下的自然/广告排位格局（self_best_organic_rank_*/self_best_ad_rank_* 及 *_status）。"
    " search_volume_change_rate ≥ +0.3 视为“明显上升”，+0.1~+0.3 为“略有上升”，-0.1~+0.1 视为“基本稳定”，≤ -0.1 则说明需求走弱；scene_kw_self_share 或 scene_kw_comp_share 较上周变化 ≥ ±0.03 需要强调份额挤压/回流。"
    " rank/status 字段需要翻译成易懂语言：strong 表示自然位进入前 16 或广告位进入前 4，medium 表示仍在可见区但非头部，weak 表示排位靠后，missing 需明确“数据缺失”；若 rank_this 与 rank_last 差值 ≥ ±10 需说明“排名大幅变化”。"
    " 重点关注：rank_this ∈ [1,10] 的头部关键词、keyword_opportunity_by_volume 中已打标的机会词、asin_keyword_profile_change 中 change_type 指向“关键词画像变化显著”的 ASIN 所覆盖的关键词。"
    " 在 7.2 小节需按照 opportunity_type（如 high_volume_organic_gap / ad_heavy_but_not_dominant / organic_good_ad_gap / rising_demand_full_gap）分组，结合搜索量、share、rank 状态说明机会本质与风险。"
    " 在 7.3 小节，每个代表性关键词下至少选择 3~5 个 ASIN 深度拆解，优先自营 ASIN，再补充曝光高的竞品，引用 keyword_asin_contributors.this_week.top_asin 中的 effective_impr_share_this（可称为 kw_impr_share）表达曝光贡献；若 ASIN 也出现在 asin_keyword_profile_change.* 中，需要补充其 head_keywords_this/last 或 keywords_added 透露的画像迁移。"
    " 引用 ASIN 时必须写成“自营/竞品 品牌（ASIN: XXXXX）”，品牌缺失写“品牌未知（ASIN: XXXXX）”；若该 ASIN 在某 keyword 下缺少自然/广告 rank 数据，需要明确“该词下自然/广告排位数据缺失，仅能根据曝光 share 评估”。"
    " 全文保持中文 Markdown，结构固定为 7.1~7.4，先结论后解释，禁止创造输入之外的关键词、ASIN 或数值。"
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

        def _has_volume(entries: Iterable[Mapping[str, Any]], field: str) -> bool:
            for entry in entries:
                value = entry.get(field)
                if value is not None and str(value).strip() != "":
                    return True
            return False

        flags = {
            "sunday_last_missing": _is_missing_value(keyword_data.get("sunday_last")),
            "last_week_pool_missing": not bool(scene_keywords.get("last_week")),
            "search_volume_this_missing": not _has_volume(
                this_keywords, "search_volume_this"
            ),
            "search_volume_last_missing": bool(last_keywords)
            and not _has_volume(last_keywords, "search_volume_last"),
        }
        output_requirements = [
            "一级标题必须为“七、搜索需求与关键词机会”。",
            "全文需组织为 7.1~7.4 四个小节：7.1 场景搜索需求概览，7.2 关键词机会分类（按 opportunity_type 归类），7.3 关键词 × ASIN 深度解剖，7.4 归纳行动建议。",
            "无论 7.1/7.2/7.3 在讨论哪个关键词，都要按“搜索需求→曝光结构→自然/广告排位”三步输出：search_volume_change_rate ≥ +30% 记为“明显上升”、+10%~+30% 为“略有上升”、-10%~+10% 视为“基本稳定”、≤ -10% 为“需求走弱”；scene_kw_self_share 或 scene_kw_comp_share 与上一周差值 ≥ ±0.03 需要点明份额回流/流失；自营/竞品 rank 的差值 ≥ ±10 需判定为大幅变化。",
            "7.1 应结合 scene_head_keywords.this_week/last_week/diff 的 search_volume_*、scene_kw_share_*、rank_this/last、self/comp 的 best_organic/ad_rank 与 *_status，交代核心词池的需求层级、结构变化与排位格局，若上一周词池或搜索量缺失需声明只做静态观察。",
            "7.2 必须按照 keyword_opportunity_by_volume.*.opportunity_type 分组阐述，优先覆盖 rank_this ∈ [1,10] 的头部关键词、机会列表中打标词，以及 asin_keyword_profile_change.* 中 change_type 为“关键词画像变化显著”的 ASIN 所涉及的关键词；每组都要说明自营/竞品 share 以及自然/广告位差异。",
            "7.3 需针对 7.2 中的每个代表性关键词列出 3~5 个 ASIN（自营优先，其次竞品），引用 keyword_asin_contributors.this_week.top_asin 的 effective_impr_share_this（可称为 kw_impr_share）描述曝光贡献；若 ASIN 同时出现在 asin_keyword_profile_change.* 中，要结合 head_keywords_this/last 或 keywords_added 说明关键词画像迁移。",
            "每个 ASIN 的描述必须包含：品牌+ASIN 的格式“自营/竞品 品牌（ASIN: XXXXX）”，该关键词的搜索量/趋势、该 ASIN 在该词下的曝光 share、自然/广告排位（缺失需写“数据缺失”）以及清晰的运营建议；若 keyword 层缺少 rank/status，也要声明依据不足。",
            "7.4 需基于不同 opportunity_type 总结可执行动作，明确优先处理的 keyword+ASIN 组合、资源投入（Listing 优化/广告/新品布局）与守攻策略，并引用前文证据。",
            "禁止创造输入中不存在的关键词、品牌或 ASIN，所有结论必须引用 keyword_change.json、keyword_opportunity_by_volume、keyword_asin_contributors 或 asin_keyword_profile_change 中的事实。",
        ]
        return {
            "chapter": KEYWORD_TITLE,
            "context": metadata,
            "keyword_change_json": keyword_data,
            "json_notes": {
                "scene_head_keywords": "scene_head_keywords.this_week/last_week 表示场景层的头部需求池，diff 中包含新增/退出/共通关键词。",
                "asin_keyword_profile_change": "asin_keyword_profile_change.self/competitor 描述各 ASIN 的关键词画像变化与 change_score。",
                "keyword_asin_contributors": "keyword_asin_contributors.this_week 呈现每个关键词贡献最高的 ASIN 列表。",
                "keyword_opportunity_by_volume": "keyword_opportunity_by_volume.* 中的每条记录都带有自然/广告 rank/status 以及 opportunity_type，便于按打法归类。",
            },
            "data_quality_flags": flags,
            "output_requirements": output_requirements,
            "style_notes": {
                "tone": "以数据支撑的场景归纳，并延伸到 ASIN 级别的诊断与建议，逻辑清晰、分点展开。",
                "length": "建议 600-1200 字，可视数据量适度延长。",
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

