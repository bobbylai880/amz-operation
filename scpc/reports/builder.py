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


__all__ = ["build_report"]
