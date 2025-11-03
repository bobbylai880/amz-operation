from __future__ import annotations

from datetime import date

import numpy as np

from scpc.etl.scene_drivers import compute_scene_drivers
from scpc.tests.data import (
    build_clean_panel_sample,
    build_features_for_drivers,
    build_raw_facts_for_drivers,
    build_weight_table,
)


def test_scene_drivers_wow_contributions() -> None:
    clean_panel = build_clean_panel_sample(num_weeks=2)
    target_mask = (clean_panel["keyword_norm"] == "shower bag") & (clean_panel["start_date"] > clean_panel["start_date"].min())
    clean_panel.loc[target_mask, "vol_s"] = 120.0
    raw = build_raw_facts_for_drivers()
    features = build_features_for_drivers()
    weights = build_weight_table()

    result = compute_scene_drivers(
        clean_panel,
        raw,
        features,
        weights,
        scene="浴室架",
        marketplace_id="US",
        topn=5,
    ).data

    # Only week 2 should produce WoW drivers (week 1 lacks baseline).
    wow_drivers = result[(result["start_date"] == date(2024, 1, 14)) & (result["horizon"] == "WoW")]
    assert len(wow_drivers) == 2

    pos = wow_drivers[wow_drivers["direction"] == "pos"].iloc[0]
    neg = wow_drivers[wow_drivers["direction"] == "neg"].iloc[0]

    assert pos["keyword"] == "shower caddy"
    assert np.isclose(pos["contrib"], (440 - 400) / 570, rtol=1e-6)
    assert pos["rank_delta"] == -2

    assert neg["keyword"] == "shower bag"
    assert np.isclose(neg["contrib"], (120 - 150) * 0.6 / 570, rtol=1e-6)
    assert neg["rank_delta"] == 7
