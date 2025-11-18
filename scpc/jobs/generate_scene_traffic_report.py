"""CLI entry point for scene traffic Markdown chapters."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Sequence

from scpc.reports.scene_traffic_report import (
    SceneTrafficReportError,
    SceneTrafficReportGenerator,
    SceneTrafficReportParams,
)

LOGGER = logging.getLogger(__name__)
WEEK_PATTERN = re.compile(r"^\d{4}-W\d{2}$")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scene traffic Markdown chapters (flow + keywords)"
    )
    parser.add_argument("--week", required=True, help="ISO week string, e.g. 2025-W45")
    parser.add_argument("--scene_tag", required=True, help="Scene tag, e.g. 浴室袋")
    parser.add_argument(
        "--marketplace",
        required=True,
        help="Marketplace ID matching ETL outputs, e.g. US",
    )
    parser.add_argument(
        "--storage",
        default="storage/weekly_report",
        help="Root directory where traffic JSON and reports are stored",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not WEEK_PATTERN.match(args.week):
        LOGGER.error("Invalid week format", extra={"week": args.week})
        return 1

    params = SceneTrafficReportParams(
        week=args.week,
        scene_tag=args.scene_tag,
        marketplace_id=args.marketplace,
        storage_dir=Path(args.storage),
    )

    generator = SceneTrafficReportGenerator()
    try:
        generator.run(params)
    except SceneTrafficReportError as exc:
        LOGGER.error(
            "scene_traffic_report_failed",
            extra={
                "week": args.week,
                "scene_tag": args.scene_tag,
                "marketplace": args.marketplace,
                "error": str(exc),
            },
        )
        return 1
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.exception(
            "scene_traffic_report_unexpected_error",
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

