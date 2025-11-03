from __future__ import annotations

import numpy as np

from scpc.etl.scene_clean import clean_keyword_panel
from scpc.tests.data import build_keyword_facts_with_gaps


def test_clean_keyword_panel_handles_short_and_long_gaps() -> None:
    facts, weeks = build_keyword_facts_with_gaps()
    clean = clean_keyword_panel(facts, week_index=weeks).data

    # Each keyword should contain all requested weeks.
    alpha = clean[clean["keyword_norm"] == "shower caddy"].set_index("start_date")
    beta = clean[clean["keyword_norm"] == "shower bag"].set_index("start_date")
    assert set(alpha.index) == set(weeks)
    assert set(beta.index) == set(weeks)

    # Short gaps (<3 weeks) are interpolated and smoothed (3-week rolling mean).
    assert np.isclose(alpha.loc[weeks[1], "vol_s"], 469.4, rtol=1e-6)

    # Long gaps (>2 weeks) remain flagged.
    assert beta.loc[weeks[1], "gap_flag"] == 1
    assert beta.loc[weeks[2], "gap_flag"] == 1
    assert beta.loc[weeks[3], "gap_flag"] == 1
    assert np.isnan(beta.loc[weeks[1], "vol_s"])

    # Winsor bounds should be carried through each row.
    assert alpha["winsor_low"].nunique() == 1
    assert alpha["winsor_high"].nunique() == 1

    # Start dates are aligned to Sundays.
    assert all(dt.weekday() == 6 for dt in clean["start_date"])  # 6 == Sunday
