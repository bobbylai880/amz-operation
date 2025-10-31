"""Child level profit-first feature engineering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class ChildFinancials:
    """Derived metrics prioritising profitable allocation decisions."""

    effective_woc: float
    risk_level: str
    gmroi_gross: float | None
    gmroi_net_ads: float | None
    ppad: float | None


def compute_effective_woc(
    woc_fba: float | None,
    woc_overseas: float | None,
    woc_local: float | None,
    alpha_fba: float = 1.0,
    alpha_transfer: float = 0.8,
    alpha_local: float = 0.5,
) -> float:
    """Calculate weighted weeks-of-coverage across fulfilment channels."""

    def _safe(value: float | None) -> float:
        return float(value or 0.0)

    return (
        alpha_fba * _safe(woc_fba)
        + alpha_transfer * _safe(woc_overseas)
        + alpha_local * _safe(woc_local)
    )


def classify_risk(effective_woc: float, theta_weeks: float) -> str:
    """Classify inventory risk for a single SKU."""

    if effective_woc <= 0:
        return "HIGH"
    if effective_woc < theta_weeks:
        return "LOW"
    return "NONE"


def compute_profitability_metrics(
    orders: float | None,
    revenue: float | None,
    ad_spend: float | None,
    gross_profit_unit: float | None,
    gross_profit_total: float | None,
    effective_woc: float,
    implied_cost_rate: float = 0.55,
) -> tuple[float | None, float | None, float | None]:
    """Compute GMROI (gross & net ads) and PPAD."""

    orders = float(orders or 0.0)
    revenue = float(revenue or 0.0)
    ad_spend = float(ad_spend or 0.0)
    gross_profit_unit = None if gross_profit_unit is None else float(gross_profit_unit)
    gross_profit_total = None if gross_profit_total is None else float(gross_profit_total)

    if gross_profit_total is None:
        gross_profit_total = orders * (gross_profit_unit or 0.0)

    if effective_woc <= 0:
        avg_inventory_cost = 0.0
    else:
        avg_inventory_cost = revenue * implied_cost_rate * effective_woc

    gmroi_gross = None
    gmroi_net = None
    ppad = None

    if avg_inventory_cost > 0:
        gmroi_gross = gross_profit_total / avg_inventory_cost if avg_inventory_cost else None
        gmroi_net = (
            (gross_profit_total - ad_spend) / avg_inventory_cost
            if avg_inventory_cost and gross_profit_total is not None
            else None
        )

    if ad_spend > 0:
        ppad = gross_profit_total / ad_spend if gross_profit_total is not None else None

    return gmroi_gross, gmroi_net, ppad


def build_child_financials(
    record: Mapping[str, object],
    alpha_config: Mapping[str, float],
    theta_weeks: float,
    implied_cost_rate: float = 0.55,
) -> ChildFinancials:
    """Aggregate financial insights for a single child ASIN."""

    effective_woc = compute_effective_woc(
        record.get("woc_fba"),
        record.get("woc_overseas"),
        record.get("woc_local"),
        alpha_fba=float(alpha_config.get("fba", 1.0)),
        alpha_transfer=float(alpha_config.get("transfer", 0.8)),
        alpha_local=float(alpha_config.get("fbm", 0.5)),
    )
    risk = classify_risk(effective_woc, theta_weeks)
    gmroi_gross, gmroi_net, ppad = compute_profitability_metrics(
        orders=record.get("orders"),
        revenue=record.get("revenue"),
        ad_spend=record.get("ad_spend"),
        gross_profit_unit=record.get("est_gross_profit_unit"),
        gross_profit_total=record.get("gross_profit_total"),
        effective_woc=effective_woc,
        implied_cost_rate=implied_cost_rate,
    )
    return ChildFinancials(
        effective_woc=effective_woc,
        risk_level=risk,
        gmroi_gross=gmroi_gross,
        gmroi_net_ads=gmroi_net,
        ppad=ppad,
    )


__all__ = [
    "ChildFinancials",
    "build_child_financials",
    "classify_risk",
    "compute_effective_woc",
    "compute_profitability_metrics",
]
