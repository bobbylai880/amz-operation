"""Utilities for retrieving raw data from the transactional database.

This module centralises all SQL reads and ensures queries are parameterised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass(slots=True)
class ExtractParams:
    """Container for the common parameters used in weekly extracts."""

    iso_weeks: Sequence[str]
    scene_id: str | None = None
    parent_id: str | None = None


class DataExtractor:
    """Helper responsible for issuing parameterised read queries.

    The extractor keeps no state besides the SQLAlchemy engine instance.  Each
    method returns *all* rows for the requested entity as a list of mappings.
    Callers are expected to convert the result into Pandas frames or other
    structures before entering the feature layer.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def fetch_scene_keywords(self, scene_id: str) -> list[Mapping[str, object]]:
        """Return the keyword mapping for the supplied scene identifier."""

        stmt = text(
            """
            SELECT scene_id, keyword
            FROM scene_keywords
            WHERE scene_id = :scene_id
            ORDER BY keyword
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt, {"scene_id": scene_id}).mappings().all()
        return list(rows)

    def fetch_keyword_metrics(self, keywords: Iterable[str], iso_weeks: Sequence[str]) -> list[Mapping[str, object]]:
        """Return search metrics for the given keyword collection."""

        if not keywords or not iso_weeks:
            return []
        stmt = text(
            """
            SELECT keyword, iso_week, search_volume
            FROM keyword_weekly_metrics
            WHERE keyword IN :keywords AND iso_week IN :iso_weeks
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                stmt,
                {"keywords": tuple(keywords), "iso_weeks": tuple(iso_weeks)},
            ).mappings().all()
        return list(rows)

    def fetch_parent_funnel(self, params: ExtractParams) -> list[Mapping[str, object]]:
        """Fetch weekly parent funnel metrics for the specified parent."""

        stmt = text(
            """
            SELECT parent_id, iso_week, impr_ads, clicks, sessions,
                   orders, revenue, buybox_pct, bsr_main
            FROM parent_funnel_weekly
            WHERE parent_id = :parent_id AND iso_week IN :iso_weeks
            ORDER BY iso_week
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                stmt,
                {"parent_id": params.parent_id, "iso_weeks": tuple(params.iso_weeks)},
            ).mappings().all()
        return list(rows)

    def fetch_child_funnel(self, child_asins: Sequence[str], iso_weeks: Sequence[str]) -> list[Mapping[str, object]]:
        """Fetch weekly child level funnel metrics for the supplied ASINs."""

        if not child_asins or not iso_weeks:
            return []
        stmt = text(
            """
            SELECT child_asin, iso_week, impr_ads, clicks, orders, revenue
            FROM child_funnel_weekly
            WHERE child_asin IN :child_asins AND iso_week IN :iso_weeks
            ORDER BY child_asin, iso_week
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                stmt,
                {"child_asins": tuple(child_asins), "iso_weeks": tuple(iso_weeks)},
            ).mappings().all()
        return list(rows)

    def fetch_inventory(self, child_asins: Sequence[str], iso_week: str) -> list[Mapping[str, object]]:
        """Retrieve inventory coverage for the provided child ASINs."""

        if not child_asins:
            return []
        stmt = text(
            """
            SELECT child_asin, iso_week, woc_fba, woc_local, woc_overseas,
                   sla_local_days, transfer_leadtime_days,
                   inbound_fba_woc_7d, inbound_fba_woc_14d
            FROM inventory_woc
            WHERE child_asin IN :child_asins AND iso_week = :iso_week
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                stmt,
                {"child_asins": tuple(child_asins), "iso_week": iso_week},
            ).mappings().all()
        return list(rows)

    def fetch_ads(self, child_asins: Sequence[str], iso_week: str) -> list[Mapping[str, object]]:
        """Fetch weekly advertising performance split by channel."""

        if not child_asins:
            return []
        stmt = text(
            """
            SELECT child_asin, iso_week, channel, spend, clicks, impressions,
                   cpc, acos, roas
            FROM ads_weekly
            WHERE child_asin IN :child_asins AND iso_week = :iso_week
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                stmt,
                {"child_asins": tuple(child_asins), "iso_week": iso_week},
            ).mappings().all()
        return list(rows)

    def fetch_profit_estimates(self, child_asins: Sequence[str], iso_week: str) -> list[Mapping[str, object]]:
        """Retrieve gross profit estimates for child ASINs."""

        if not child_asins:
            return []
        stmt = text(
            """
            SELECT child_asin, iso_week, est_gross_profit_unit, gross_profit_total
            FROM profit_estimates
            WHERE child_asin IN :child_asins AND iso_week = :iso_week
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                stmt,
                {"child_asins": tuple(child_asins), "iso_week": iso_week},
            ).mappings().all()
        return list(rows)


__all__ = ["DataExtractor", "ExtractParams"]
