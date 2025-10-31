"""Unit tests for parent feature calculations."""
from __future__ import annotations

import math

from scpc.features.parent import LMDIResult, compute_lead_stock_ok, compute_lmdi


def test_compute_lmdi_handles_zero_values() -> None:
    current = {"impr": 1000.0, "ctr": 0.05, "cvr": 0.2}
    previous = {"impr": 800.0, "ctr": 0.04, "cvr": 0.1}
    result = compute_lmdi(current, previous)
    assert isinstance(result, LMDIResult)
    assert math.isfinite(result.c_impr)
    assert math.isfinite(result.c_ctr)
    assert math.isfinite(result.c_cvr)
    assert math.isclose(result.contrib_impr + result.contrib_ctr + result.contrib_cvr, 1.0, rel_tol=1e-6)


def test_compute_lead_stock_ok_flags_risky_skus() -> None:
    lead_skus = [
        {"child_asin": "A", "effective_woc": 0.5, "click_share": 0.3},
        {"child_asin": "B", "effective_woc": 2.0, "click_share": 0.2},
    ]
    score, risks = compute_lead_stock_ok(lead_skus, theta_weeks=1.0)
    assert score == 0.7
    assert risks[0]["child_asin"] == "A"
