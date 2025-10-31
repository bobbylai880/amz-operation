"""Unit tests for child feature engineering."""
from __future__ import annotations

from scpc.features.child import (
    ChildFinancials,
    build_child_financials,
    classify_risk,
    compute_effective_woc,
    compute_profitability_metrics,
)


def test_effective_woc_weighting() -> None:
    assert compute_effective_woc(1.0, 2.0, 3.0) == 1.0 * 1.0 + 2.0 * 0.8 + 3.0 * 0.5


def test_classify_risk_thresholds() -> None:
    assert classify_risk(0.0, 1.0) == "HIGH"
    assert classify_risk(0.5, 1.0) == "LOW"
    assert classify_risk(2.0, 1.0) == "NONE"


def test_compute_profitability_metrics_handles_missing_profit() -> None:
    gmroi_gross, gmroi_net, ppad = compute_profitability_metrics(
        orders=10,
        revenue=100.0,
        ad_spend=20.0,
        gross_profit_unit=None,
        gross_profit_total=None,
        effective_woc=2.0,
    )
    assert gmroi_gross is not None
    assert gmroi_net is not None
    assert ppad is not None


def test_build_child_financials_combines_metrics() -> None:
    record = {
        "woc_fba": 1.0,
        "woc_overseas": 1.0,
        "woc_local": 0.5,
        "orders": 10,
        "revenue": 100.0,
        "ad_spend": 10.0,
        "est_gross_profit_unit": 5.0,
    }
    fin = build_child_financials(record, {"fba": 1.0, "transfer": 0.8, "fbm": 0.5}, theta_weeks=1.0)
    assert isinstance(fin, ChildFinancials)
    assert fin.risk_level in {"LOW", "NONE"}
