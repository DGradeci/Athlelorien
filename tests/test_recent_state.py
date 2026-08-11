import numpy as np
import pandas as pd

from analysis.levy_paper.scripts.create_final_fig4_fig5_polished import recent_state_by_run


def test_recent_state_uses_prior_five_seconds_and_recent_tie_break():
    frame = pd.DataFrame(
        {
            "_run_id": ["run"] * 7,
            "_age_s": np.arange(0.5, 7.5, 1.0),
            "_dt_s": [1.0] * 7,
            "_state": ["low", "high", "low", "high", "mid", "mid", "high"],
        }
    )
    result = recent_state_by_run(frame, window_s=5.0)
    expected = pd.Series(
        [None, "low", "high", "low", "high", "high", "mid"],
        dtype=object,
    )
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)


def test_recent_state_does_not_use_future_or_current_state():
    frame = pd.DataFrame(
        {
            "_run_id": ["a", "a", "b", "b"],
            "_age_s": [0.5, 1.5, 0.5, 1.5],
            "_dt_s": [1.0] * 4,
            "_state": ["low", "high", "mid", "low"],
        }
    )
    result = recent_state_by_run(frame, window_s=5.0)
    expected = pd.Series([None, "low", None, "mid"], dtype=object)
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)
