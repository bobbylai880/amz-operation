"""Markdown report assembly helpers."""
from __future__ import annotations

from typing import Mapping, Sequence


def build_report(
    scene: Mapping[str, object],
    competition: Mapping[str, object],
    parent: Mapping[str, object],
    child: Mapping[str, object],
    budget_plan: Sequence[Mapping[str, object]],
) -> str:
    """Generate a human readable weekly report."""

    lines: list[str] = []
    lines.append("# 周度运营复盘\n")
    lines.append("## 场景（Scene）\n")
    scene_forecast = scene.get("scene_forecast")
    if isinstance(scene_forecast, Mapping):
        weeks = scene_forecast.get("weeks", []) or []
    else:
        weeks = []
    if weeks:
        first_week = weeks[0]
        if isinstance(first_week, Mapping):
            direction = first_week.get("direction", "未知")
            pct = _format_pct(first_week.get("pct_change"))
            lines.append(f"状态：未来趋势 {direction}（{pct}）\n")
        else:
            lines.append("状态：未来趋势未知\n")
    else:
        lines.append(f"状态：{scene.get('status', '未知')}\n")
    lines.append("## 竞争（Competition）\n")
    lines.append(f"强度：{competition.get('intensity', 'N/A')}\n")
    lines.append("## 父体（Parent）\n")
    lines.append(f"主短板：{parent.get('primary_weak_step', 'N/A')}\n")
    lines.append("## 子体（Child）\n")
    lines.append(f"重点子体：{len(child.get('items', []))}\n")
    lines.append("## 预算计划\n")
    for entry in budget_plan:
        lines.append(
            "- {asin}: {from_} -> {to} ({status})".format(
                asin=entry.get("asin", "?"),
                from_=entry.get("current_budget", 0),
                to=entry.get("proposed_budget", 0),
                status="待审核" if entry.get("pending_review") else "自动执行",
            )
        )
    return "\n".join(lines)


def _format_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "?"
    return f"{number:.1f}%"


def build_scene_markdown(summary: Mapping[str, object]) -> str:
    """Render a concise Markdown section for a single scene summary."""

    lines: list[str] = ["# 场景级总结", ""]
    confidence = summary.get("confidence")
    if confidence is not None:
        try:
            conf_val = float(confidence)
        except (TypeError, ValueError):
            conf_val = None
        if conf_val is not None:
            lines.append(f"置信度：{conf_val:.2f}\n")

    analysis_summary = summary.get("analysis_summary")
    if isinstance(analysis_summary, str) and analysis_summary.strip():
        lines.append("## 综合分析\n")
        lines.append(analysis_summary.strip() + "\n")

    forecast = summary.get("scene_forecast", {})
    weeks = forecast.get("weeks", []) if isinstance(forecast, Mapping) else []
    if weeks:
        lines.append("## 场景预测（未来4周）\n")
        for week in weeks:
            if not isinstance(week, Mapping):
                continue
            year = week.get("year")
            week_num = week.get("week_num")
            start_date = week.get("start_date", "?")
            direction = week.get("direction", "?")
            pct = _format_pct(week.get("pct_change"))
            if isinstance(week_num, int):
                label = f"{year}W{week_num:02d}"
            else:
                label = f"{year}W{week_num}"
            lines.append(f"- {label} ({start_date}): {direction} ({pct})")
        lines.append("")

    keyword_forecast = summary.get("top_keywords_forecast", []) or []
    if keyword_forecast:
        lines.append("## 关键词预测\n")
        for entry in keyword_forecast:
            if not isinstance(entry, Mapping):
                continue
            keyword = entry.get("keyword", "?")
            lines.append(f"- {keyword}")
            for week in entry.get("weeks", []) or []:
                if not isinstance(week, Mapping):
                    continue
                year = week.get("year")
                week_num = week.get("week_num")
                direction = week.get("direction", "?")
                pct = _format_pct(week.get("pct_change"))
                if isinstance(week_num, int):
                    label = f"{year}W{week_num:02d}"
                else:
                    label = f"{year}W{week_num}"
                lines.append(f"  - {label}: {direction} ({pct})")
        lines.append("")

    if summary.get("insufficient_data"):
        lines.append("> 数据覆盖不足，结论需谨慎对待。\n")
    notes = summary.get("notes")
    if notes:
        lines.append("## 备注\n")
        lines.append(str(notes))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


__all__ = ["build_report", "build_scene_markdown"]
