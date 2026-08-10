from pathlib import Path

import pandas as pd

from analysis.levy_paper.scripts.build_multiseason_data_cache import (
    SeasonConfig,
    _filter_match_dates,
    _missing_schedule_inputs,
)


def test_filter_match_dates_keeps_requested_dates():
    matches = pd.DataFrame(
        {
            "date": ["2020-07-03", "2020-08-16", "2021-10-16"],
            "home": ["A", "B", "C"],
        }
    )

    selected = _filter_match_dates(matches, ["2020-08-16", "2021-10-16"])

    assert selected["date"].tolist() == ["2020-08-16", "2021-10-16"]


def test_filter_match_dates_none_preserves_table():
    matches = pd.DataFrame({"date": ["2020-07-03"]})

    selected = _filter_match_dates(matches, None)

    assert selected is matches


def test_missing_schedule_inputs_reports_only_absent_paths(tmp_path: Path):
    present = tmp_path / "present.xlsx"
    present.touch()
    absent = tmp_path / "absent.xlsx"
    configs = [
        SeasonConfig("2020", present, "A", "B"),
        SeasonConfig("2021", absent, "A", "B"),
    ]

    assert _missing_schedule_inputs(configs) == [absent]
