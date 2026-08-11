"""Build the processed model cache used by publication Figures 4 and 5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.levy_paper.modeling import CrossfitConfig, build_crossfitted_state_survival_cache


DEFAULT_INPUT = REPO_ROOT / "analysis" / "levy_paper" / "data" / "processed" / "all_team_2020_2021_sticky_active" / "hazard_intervals_2020_2021_all_teams_pitchfix_sticky_active.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "analysis" / "levy_paper" / "data" / "processed" / "figure4_fig5_crossfitted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Primary one-second hazard interval parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination processed model-cache folder")
    parser.add_argument("--bootstrap", type=int, default=500, help="Physical-fixture bootstrap replicates for displayed uncertainty")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing primary interval cache: {args.input}\n"
            "Build it with build_multiseason_data_cache.py (AWS route), or install the optional processed-cache bundle."
        )
    summary = build_crossfitted_state_survival_cache(
        args.input,
        args.output,
        CrossfitConfig(bootstrap_replicates=args.bootstrap),
    )
    print(json.dumps({"input": str(args.input), "output": str(args.output), **summary}, indent=2))


if __name__ == "__main__":
    main()
