"""CLI entry point for the weekly scene Markdown report generator."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Sequence

from scpc.reports.weekly_scene_report import (
    WeeklySceneReportError,
    WeeklySceneReportGenerator,
    WeeklySceneReportParams,
)

LOGGER = logging.getLogger(__name__)
WEEK_PATTERN = re.compile(r"^\d{4}-W\d{2}$")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown weekly reports for a scene and marketplace"
    )
    parser.add_argument("--week", required=True, help="ISO week string, e.g. 2025-W45")
    parser.add_argument("--scene_tag", required=True, help="Scene tag, e.g. 浴室袋")
    parser.add_argument(
        "--marketplace",
        required=True,
        help="Marketplace ID matching previous ETL outputs, e.g. US",
    )
    parser.add_argument(
        "--storage",
        default="storage/weekly_report",
        help="Root directory where JSON modules and Markdown reports are stored",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not WEEK_PATTERN.match(args.week):
        LOGGER.error("Invalid week format", extra={"week": args.week})
        return 1

    params = WeeklySceneReportParams(
        week=args.week,
        scene_tag=args.scene_tag,
        marketplace_id=args.marketplace,
        storage_dir=Path(args.storage),
    )

    generator = WeeklySceneReportGenerator()
    try:
        generator.run(params)
    except WeeklySceneReportError as exc:
        LOGGER.error(
            "weekly_scene_report_failed",
            extra={
                "week": args.week,
                "scene_tag": args.scene_tag,
                "marketplace": args.marketplace,
                "error": str(exc),
            },
        )
        return 1
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception(
            "weekly_scene_report_unexpected_error",
            extra={
                "week": args.week,
                "scene_tag": args.scene_tag,
                "marketplace": args.marketplace,
            },
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())

