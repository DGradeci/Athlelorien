import numpy as np
import pandas as pd

from analysis.levy_paper.modeling.state_survival import (
    _age_exposure,
    _late_integral,
    _prepare_intervals,
    _probability_from_exposure,
)


def test_integrated_interval_probability_matches_survival_ratio():
    mu = 1.3
    a0 = 2.4
    start = np.array([0.0, 4.0, 20.0])
    end = start + 1.0
    probability = _probability_from_exposure(_age_exposure(start, end, mu, a0))
    expected = 1.0 - ((end + a0) / (start + a0)) ** (-mu)
    np.testing.assert_allclose(probability, expected)


def test_smooth_late_integral_is_zero_before_onset_and_continuous():
    values = _late_integral(np.array([0.0, 19.9, 20.0, 20.1]), tc=20.0, tau=5.0)
    np.testing.assert_allclose(values[:3], 0.0)
    assert 0.0 < values[3] < 0.1


def test_administrative_censoring_uses_absolute_record_end():
    frame = pd.DataFrame(
        {
            "season": [2020, 2020],
            "match_id": ["2020-01-01", "2020-01-01"],
            "team": ["team", "team"],
            "source_key": ["A", "A"],
            "match_phase": ["1H", "1H"],
            "run_uid": ["old-long", "latest-short"],
            "age_start_s": [9.0, 0.0],
            "age_end_s": [10.0, 1.0],
            "dt_s": [1.0, 1.0],
            "event": [1, 1],
            "run_duration_s": [10.0, 1.0],
            "p_group": [0.5, 0.5],
            "t1": [100.0, 200.0],
        }
    )
    prepared = _prepare_intervals(frame)
    assert prepared.loc[prepared["run_uid"].eq("old-long"), "observed_termination"].item() == 1
    assert prepared.loc[prepared["run_uid"].eq("latest-short"), "censored_event"].item()
