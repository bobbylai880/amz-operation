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


def build_scene_markdown(summary: Mapping[str, object]) -> str:
    """Render a concise Markdown section for a single scene summary."""

    lines: list[str] = ["# 场景级总结", ""]
    status = summary.get("status", "未知")
    lines.append(f"状态：{status}\n")
    drivers = summary.get("drivers", []) or []
    if drivers:
        lines.append("## 关键驱动词\n")
        for driver in drivers:
            keyword = driver.get("keyword") or driver.get("kw") or "?"
            delta = driver.get("delta")
            if delta is None:
                delta = driver.get("contrib")
            lines.append(f"- {keyword}: {delta}")
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
