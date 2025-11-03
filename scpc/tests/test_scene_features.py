from __future__ import annotations

import numpy as np

from scpc.etl.scene_features import compute_scene_features
from scpc.tests.data import build_clean_panel_sample, build_coverage_table, build_weight_table


def test_scene_features_basic_metrics() -> None:
    clean_panel = build_clean_panel_sample()
    weights = build_weight_table()
    coverage = build_coverage_table()
    result = compute_scene_features(clean_panel, weights, coverage, scene="浴室架", marketplace_id="US").data

    assert len(result) == 10

    first_week = result.iloc[0]
    expected_vol = 400 + 0.6 * 150
    assert first_week["VOL"] == int(round(expected_vol))
    assert np.isnan(first_week["wow"])

    second_week = result.iloc[1]
    prev_vol = expected_vol
    current_vol = 440 + 0.6 * 162
    assert np.isclose(second_week["wow"], current_vol / prev_vol - 1, rtol=1e-6)
    assert np.isclose(second_week["breadth_wow_pos"], 1.0)

    eighth_week = result.iloc[7]
    assert eighth_week["new_kw_share"] > 0

    last_week = result.iloc[-1]
    assert last_week["confidence"] <= 0.55
    assert all(last_week[col] is not None for col in ["forecast_p10", "forecast_p50", "forecast_p90"])
