"""Reproduce the publication figures through the supported cache or AWS route."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
PAPER_ROOT = REPO_ROOT / "analysis" / "levy_paper"
SCRIPTS = PAPER_ROOT / "scripts"


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(SCRIPTS / script), *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cache", "aws"), default="cache")
    parser.add_argument("--bootstrap", type=int, default=500, help="Fixture-bootstrap replicates for Figures 4/5")
    parser.add_argument("--skip-model-refit", action="store_true", help="Use an existing Figure 4/5 processed model cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "aws":
        run("build_multiseason_data_cache.py")

    if not args.skip_model_refit:
        run("build_figure4_figure5_model_cache.py", "--bootstrap", str(args.bootstrap))

    for script in (
        "create_real_match_transport_mechanism_4panel.py",
        "create_final_figure2_transport_phenotype.py",
        "create_final_figure3_order_transport.py",
        "create_final_fig4_fig5_polished.py",
        "sync_final_figures.py",
        "create_supplementary_figures_from_source.py",
        "sync_formal_supplementary_material.py",
        "sanitize_publication_metadata.py",
        "check_publication_reproducibility.py",
    ):
        run(script)


if __name__ == "__main__":
    main()
