"""CLI entry point for the ASIN week diff ETL job."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Sequence

from scpc.etl.asin_week_diff import CONFIG_PATH, WeekDiffJobError, run_week_diff_etl

LOGGER = logging.getLogger(__name__)
WEEK_PATTERN = re.compile(r"^\d{4}-W\d{2}$")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Amazon ASIN week-over-week diffs")
    parser.add_argument("--week", required=True, help="ISO week string, e.g. 2025-W45")
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Path to the YAML file defining price/rank/promo rule labels",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not WEEK_PATTERN.match(args.week):
        LOGGER.error("Invalid week format", extra={"week": args.week})
        return 1
    config_path = Path(args.config)
    try:
        run_week_diff_etl(args.week, config_path=config_path)
    except WeekDiffJobError as exc:
        LOGGER.error("Week diff ETL failed", extra={"week": args.week, "error": str(exc)})
        return 1
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("Unexpected failure during week diff ETL", extra={"week": args.week})
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
