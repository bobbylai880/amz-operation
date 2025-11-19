"""Generate Markdown weekly reports from prepared scene JSON modules."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from scpc.llm.deepseek_client import DeepSeekClient, DeepSeekError, DeepSeekResponse
from scpc.llm.deepseek_client import create_client_from_env
from scpc.settings import DeepSeekSettings, get_deepseek_settings

LOGGER = logging.getLogger(__name__)
SAFE_SCENE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(slots=True)
class WeeklySceneReportParams:
    """Container for CLI parameters used by the report generator."""

    week: str
    scene_tag: str
    marketplace_id: str
    storage_dir: Path


class WeeklySceneReportError(RuntimeError):
    """Raised when the weekly scene Markdown report generation fails."""


@dataclass(frozen=True)
class ModuleSpec:
    """Description of a module-level LLM call."""

    key: str
    input_filename: str
    output_filename: str
    title_template: str
    instruction_template: str


COMMON_STYLE_GUIDE = (
    "请严格基于 module_data JSON 输出中文 Markdown，保持资深亚马逊运营负责人的口吻。"
    " 必须引用输入中的事实，禁止编造 ASIN、品牌或数值；若列表为空需明确说明未识别到符合条件的对象。"
    " 输出结构需包含标题、小节与要点列表，并附上可执行建议。"
)

SYSTEM_PROMPT = (
    "你是一名资深的亚马逊运营负责人，熟悉搜索排序、价格策略、优惠机制、A+ 内容和竞品分析。"
    " 你的分析必须以输入 JSON 为依据，不得捏造额外数据。若信息不足，应在文中解释假设，并保持中文输出。"
)

MODULE_SPECS: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        key="overall_summary",
        input_filename="overall_summary.json",
        output_filename="01_overall_summary.md",
        title_template="一、市场整体概览（{scene_tag}，{marketplace_id}，{week})",
        instruction_template=(
            "你负责{scene_tag}在{marketplace_id}站点，周度 {week} 的大盘复盘。"
            " 请结合 module_data 中的 ASIN 体量、自营/竞品结构、排名变化、价格动作、优惠动作、badge 波动，"
            " 输出市场规模与结构、排名走势、价格与促销环境、badge 波动含义，并给出 2-3 条下周策略建议。"
        ),
    ),
    ModuleSpec(
        key="self_analysis",
        input_filename="self_analysis.json",
        output_filename="02_self_analysis.md",
        title_template="二、自营 ASIN 表现复盘",
        instruction_template=(
            "你在分析我方 ASIN 在 {marketplace_id} 站点 {scene_tag} 场景周 {week} 的表现。"
            " 请概括整体趋势（改善/稳定/走弱），指出表现突出的 ASIN 与下滑 ASIN，结合 rank/price/promo 列表做原因判断，"
            " 并给出 3-5 条可执行的 TODO（最好明确到 ASIN/动作）。"
        ),
    ),
    ModuleSpec(
        key="competitor_analysis",
        input_filename="competitor_analysis.json",
        output_filename="03_competitor_analysis.md",
        title_template="三、竞品格局与动向",
        instruction_template=(
            "站在我方运营视角，请根据 module_data 描述竞品整体走势、冲榜/退场竞品与品牌层面的扩张或收缩，"
            " 点明对我方最大的压力与空间。确保引用 rank_up/down、price/promo 行为佐证。"
        ),
    ),
    ModuleSpec(
        key="self_risk_opportunity",
        input_filename="self_risk_opportunity.json",
        output_filename="04_self_risk_opportunity.md",
        title_template="四、自营风险与机会监控",
        instruction_template=(
            "复述 module_data.rules 中的筛选逻辑，并对 risk_asin 与 opportunity_asin 逐条给出问题/机会说明和建议。"
            " 若列表为空需说明“本周按当前规则未识别到XX”。最后补充阈值或规则是否需要调整的思考。"
        ),
    ),
    ModuleSpec(
        key="competitor_actions",
        input_filename="competitor_actions.json",
        output_filename="05_competitor_actions.md",
        title_template="五、竞品关键动作雷达",
        instruction_template=(
            "梳理竞品在价格、促销、内容&badge 三条线的动作，指出需重点盯防的 3-5 个 ASIN，并评估我方是否要跟进类似动作。"
        ),
    ),
)

FULL_REPORT_TITLE = "00_full_report.md"
FULL_REPORT_INSTRUCTION = (
    "你是一名 Amazon.com 资深运营负责人，现需基于 modules 输入（01~07 章 Markdown）生成 {scene_tag} 在 {marketplace_id} 站点"
    "、周次 {week} 的《00 场景整合周报》。"
    " 输出必须严格遵循以下结构，并仅引用 modules 中已有事实："
    " 1）报告范围与数据边界：说明场景/站点/周次、自营&竞品样本覆盖、数据缺口；"
    " 2）本周总评与关键判断：一句话概括 + 市场&竞品 / 自营健康度 / 流量&搜索三条判断；"
    " 3）本周优先行动清单：列出 5-10 条“对象+动作+依据+期望方向”的行动，依据需追溯到对应模块；"
    " 4）市场与竞品格局：拆 6.1 场景整体、6.2 竞品快照、6.3 竞品动作雷达，引用 01/03/05 数据；"
    " 5）自营经营复盘：总结核心 ASIN 成绩单（02）、规则命中与阈值判断（04）、流量结构与打法（06）；"
    " 6）流量入口与搜索需求：结合 06/07 描述整体渠道、代表关键词（2-4 个）与 ASIN 曝光格局；"
    " 7）综合行动方案与风险提示：提炼 1-3 条战略方向，按条线拆解行动，并点出数据局限/监控状态。"
    " 所有章节需使用简体中文 Markdown，小节编号可沿用 1~7；不得编造新指标或预测未来行为。"
)


@dataclass(frozen=True)
class MarkdownChapter:
    key: str
    filename: str
    title: str


OPTIONAL_MARKDOWN_MODULES: tuple[MarkdownChapter, ...] = (
    MarkdownChapter(
        key="traffic_flow",
        filename="06_traffic_flow.md",
        title="六、场景流量结构与投放策略",
    ),
    MarkdownChapter(
        key="keyword_opportunity",
        filename="07_keyword_opportunity.md",
        title="七、搜索需求与关键词机会",
    ),
)


class WeeklySceneReportGenerator:
    """Co-ordinate LLM calls to transform JSON modules into Markdown reports."""

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

    def run(self, params: WeeklySceneReportParams) -> dict[str, Path]:
        """Generate module Markdown files plus a full report."""

        metadata = {
            "week": params.week,
            "scene_tag": params.scene_tag,
            "marketplace_id": params.marketplace_id,
        }
        base_dir = self._resolve_scene_dir(params.storage_dir, params.week, params.scene_tag)
        report_dir = base_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "weekly_scene_report_start",
            extra={
                "week": params.week,
                "scene_tag": params.scene_tag,
                "marketplace": params.marketplace_id,
                "storage": str(params.storage_dir),
            },
        )

        json_payloads: dict[str, Any] = {}
        for spec in MODULE_SPECS:
            input_path = base_dir / spec.input_filename
            json_payloads[spec.key] = self._load_json(input_path)

        outputs: dict[str, Path] = {}
        module_texts: dict[str, str] = {}
        try:
            for spec in MODULE_SPECS:
                rendered = self._render_module(spec, metadata, json_payloads[spec.key])
                output_path = report_dir / spec.output_filename
                output_path.write_text(rendered, encoding="utf-8")
                outputs[spec.key] = output_path
                module_texts[spec.key] = rendered
                LOGGER.info(
                    "weekly_scene_report_module_done",
                    extra={
                        "module": spec.key,
                        "path": str(output_path),
                        "bytes": output_path.stat().st_size,
                    },
                )

            optional_modules = self._load_optional_markdown_modules(report_dir)
            module_texts.update({key: text for key, (text, _) in optional_modules.items()})
            for key, (_, path) in optional_modules.items():
                outputs[key] = path
            full_report = self._render_full_report(metadata, module_texts)
            full_report_path = report_dir / FULL_REPORT_TITLE
            full_report_path.write_text(full_report, encoding="utf-8")
            outputs["full_report"] = full_report_path
            LOGGER.info(
                "weekly_scene_report_full_done",
                extra={
                    "path": str(full_report_path),
                    "bytes": full_report_path.stat().st_size,
                },
            )
        finally:
            self._client.close()

        return outputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_scene_dir(self, storage: Path, week: str, scene_tag: str) -> Path:
        """Locate the directory containing the JSON modules for ``scene_tag``."""

        week_dir = storage / week
        candidates = list(_scene_dir_candidates(scene_tag))
        for candidate in candidates:
            candidate_dir = week_dir / candidate
            if all((candidate_dir / spec.input_filename).exists() for spec in MODULE_SPECS):
                if candidate != candidates[0]:
                    LOGGER.info(
                        "weekly_scene_report_scene_dir_fallback",
                        extra={
                            "week": week,
                            "scene_tag": scene_tag,
                            "selected": candidate,
                        },
                    )
                return candidate_dir

        # Default to the first candidate to keep error messages intuitive.
        return week_dir / candidates[0]

    def _load_json(self, path: Path) -> Any:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            LOGGER.error(
                "weekly_scene_report_missing_json %s",
                path,
                extra={"path": str(path)},
            )
            raise WeeklySceneReportError(f"Required JSON file missing: {path}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.error("weekly_scene_report_bad_json", extra={"path": str(path)})
            raise WeeklySceneReportError(f"Invalid JSON structure in {path}") from exc
        LOGGER.info(
            "weekly_scene_report_loaded_json",
            extra={"path": str(path), "bytes": len(raw)},
        )
        return data

    def _render_module(
        self,
        spec: ModuleSpec,
        metadata: Mapping[str, str],
        module_data: Any,
    ) -> str:
        instructions = spec.instruction_template.format(**metadata)
        title = spec.title_template.format(**metadata)
        facts = {
            "task": instructions,
            "style_guide": COMMON_STYLE_GUIDE,
            "length_hint": "单个模块建议控制在300-600字，紧扣数据给出洞察与行动。",
            "title": title,
            "metadata": metadata,
            "module_key": spec.key,
            "module_data": module_data,
        }
        response = self._invoke_llm(facts)
        content = response.content.strip()
        if not content:
            raise WeeklySceneReportError(
                f"Module {spec.key} returned empty content from LLM"
            )
        if not content.startswith("#"):
            content = f"# {title}\n\n{content}"
        return content if content.endswith("\n") else content + "\n"

    def _render_full_report(
        self,
        metadata: Mapping[str, str],
        module_texts: Mapping[str, str],
    ) -> str:
        facts = {
            "task": FULL_REPORT_INSTRUCTION.format(**metadata),
            "style_guide": COMMON_STYLE_GUIDE,
            "metadata": metadata,
            "modules": module_texts,
        }
        response = self._invoke_llm(facts)
        content = response.content.strip()
        if not content:
            raise WeeklySceneReportError("Full report generation returned empty content")
        return content if content.endswith("\n") else content + "\n"

    def _invoke_llm(self, facts: Mapping[str, Any]) -> DeepSeekResponse:
        try:
            return self._client.generate(
                prompt=SYSTEM_PROMPT,
                facts=facts,
                model=self._settings.model,
                temperature=self._temperature,
                response_format="text",
            )
        except DeepSeekError as exc:
            raise WeeklySceneReportError("DeepSeek request failed") from exc

    def _load_optional_markdown_modules(
        self, report_dir: Path
    ) -> dict[str, tuple[str, Path]]:
        modules: dict[str, tuple[str, Path]] = {}
        for chapter in OPTIONAL_MARKDOWN_MODULES:
            path = report_dir / chapter.filename
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover
                    LOGGER.error(
                        "weekly_scene_report_optional_read_failed",
                        extra={"module_key": chapter.key, "path": str(path)},
                    )
                    raise WeeklySceneReportError(
                        f"Unable to read optional module {chapter.filename}"
                    ) from exc
                modules[chapter.key] = (content, path)
            else:
                placeholder = (
                    f"# {chapter.title}\n\n> 数据缺失：未在 {path.name} 找到对应章节，"
                    "本周整合报告需在相关段落说明只能引用现有章节的素材。\n"
                )
                try:
                    path.write_text(placeholder, encoding="utf-8")
                except OSError as exc:  # pragma: no cover - unlikely but logged
                    LOGGER.error(
                        "weekly_scene_report_optional_write_failed",
                        extra={"module_key": chapter.key, "path": str(path)},
                    )
                    raise WeeklySceneReportError(
                        f"Unable to create placeholder for {chapter.filename}"
                    ) from exc
                LOGGER.warning(
                    "weekly_scene_report_optional_missing",
                    extra={"module_key": chapter.key, "path": str(path)},
                )
                modules[chapter.key] = (placeholder, path)
        return modules


def _scene_dir_candidates(scene_tag: str) -> Iterable[str]:
    """Yield potential storage directory names for ``scene_tag``."""

    cleaned = scene_tag.strip()
    seen: set[str] = set()
    if cleaned:
        seen.add(cleaned)
        yield cleaned
    slug = SAFE_SCENE_TAG_RE.sub("_", cleaned or scene_tag)
    slug = slug.strip("_") or "scene"
    if slug not in seen:
        yield slug


__all__ = [
    "WeeklySceneReportGenerator",
    "WeeklySceneReportParams",
    "WeeklySceneReportError",
]

