"""Parent level feature engineering utilities."""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Mapping


@dataclass(slots=True)
class LMDIResult:
    """Represents the additive log-mean decomposition result."""

    c_impr: float
    c_ctr: float
    c_cvr: float
    contrib_impr: float
    contrib_ctr: float
    contrib_cvr: float


def compute_lmdi(current: Mapping[str, float], previous: Mapping[str, float]) -> LMDIResult:
    """Perform an additive LMDI decomposition between two funnel states.

    Parameters
    ----------
    current:
        Mapping containing the current week's ``impr``, ``ctr`` and ``cvr``.
    previous:
        Mapping containing the prior week's ``impr``, ``ctr`` and ``cvr``.

    Returns
    -------
    LMDIResult
        Decomposition components and normalised contribution ratios.

    Notes
    -----
    The function clamps inputs to a minimum positive epsilon in order to avoid
    taking the logarithm of zero which would otherwise raise ``ValueError``.
    """

    epsilon = 1e-9
    current_impr = max(current.get("impr", 0.0), epsilon)
    previous_impr = max(previous.get("impr", 0.0), epsilon)
    current_ctr = max(current.get("ctr", 0.0), epsilon)
    previous_ctr = max(previous.get("ctr", 0.0), epsilon)
    current_cvr = max(current.get("cvr", 0.0), epsilon)
    previous_cvr = max(previous.get("cvr", 0.0), epsilon)

    c_impr = log(current_impr) - log(previous_impr)
    c_ctr = log(current_ctr) - log(previous_ctr)
    c_cvr = log(current_cvr) - log(previous_cvr)

    total = c_impr + c_ctr + c_cvr or epsilon
    contrib_impr = c_impr / total
    contrib_ctr = c_ctr / total
    contrib_cvr = c_cvr / total

    return LMDIResult(
        c_impr=c_impr,
        c_ctr=c_ctr,
        c_cvr=c_cvr,
        contrib_impr=contrib_impr,
        contrib_ctr=contrib_ctr,
        contrib_cvr=contrib_cvr,
    )


def compute_lead_stock_ok(
    lead_skus: Iterable[Mapping[str, object]],
    theta_weeks: float,
) -> tuple[float, list[dict[str, object]]]:
    """Evaluate inventory sufficiency for traffic-driving SKUs.

    Parameters
    ----------
    lead_skus:
        Iterable of dictionaries with ``effective_woc`` and ``click_share``.
    theta_weeks:
        Minimum desired coverage in weeks. SKUs below the threshold contribute
        to the risk score proportionally to their click share.

    Returns
    -------
    tuple
        ``lead_stock_ok`` ratio in the range ``[0, 1]`` and a list of risk
        descriptors capturing insufficient SKUs.
    """

    total_risk = 0.0
    risks: list[dict[str, object]] = []
    for sku in lead_skus:
        effective_woc = float(sku.get("effective_woc", 0.0) or 0.0)
        click_share = float(sku.get("click_share", 0.0) or 0.0)
        if effective_woc < theta_weeks:
            total_risk += click_share
            risks.append(
                {
                    "child_asin": sku.get("child_asin"),
                    "effective_woc": effective_woc,
                    "click_share": click_share,
                }
            )
    lead_stock_ok = max(0.0, 1.0 - total_risk)
    return (lead_stock_ok, risks)


__all__ = ["LMDIResult", "compute_lmdi", "compute_lead_stock_ok"]
