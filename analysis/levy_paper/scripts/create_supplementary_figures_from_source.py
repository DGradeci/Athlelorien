"""Rebuild formal Figures S1 and S2 from their tracked source-data CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.levy_paper.util.paper_utils import configure_paper_plotting, draw_panel_letter


PAPER_ROOT = REPO_ROOT / "analysis" / "levy_paper"
SUPPLEMENT = PAPER_ROOT / "figures" / "supplementary_material"
SOURCE = SUPPLEMENT / "source_data"
OUTPUT = SUPPLEMENT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

BLACK = "black"
RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
GREY = "0.5"


def save(fig: plt.Figure, stem: str, *, close_figure: bool = True) -> list[Path]:
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = OUTPUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.04)
        if extension == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
        paths.append(path)
    if close_figure:
        plt.close(fig)
    return paths


def style_axes(*axes: plt.Axes) -> None:
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.8)


def figure_s1(*, close_figure: bool = True) -> tuple[plt.Figure, list[Path]]:
    curves = pd.read_csv(SOURCE / "figureS1_panelA_model_curves.csv")
    contrasts = pd.read_csv(SOURCE / "figureS1_panelB_lofo_contrasts.csv").sort_values("plot_order")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.82})

    empirical = curves.drop_duplicates("duration_s").sort_values("duration_s")
    ax_a.plot(empirical["duration_s"], empirical["empirical_ccdf"], "o", color=BLACK, ms=3.7, label="empirical", zorder=5)
    styles = {
        "Geometric (discrete exponential)": (GREY, "--", "Geometric\n(discrete exponential)"),
        "log-normal": ("#ef9696", "--", "log-normal"),
        "conventional truncated power law": (BLACK, "-", "conv. truncated power law"),
        "generalised-cutoff power law": (GREEN, "-", "gen.-cutoff power law"),
    }
    for label, (color, linestyle, plot_label) in styles.items():
        model = curves.loc[curves["plot_label"].eq(label)].sort_values("duration_s")
        if not model.empty:
            ax_a.plot(model["duration_s"], model["model_ccdf"], color=color, ls=linestyle, lw=2.0, label=plot_label)
    ax_a.set(xscale="log", yscale="log", xlabel="Run duration $T$ (s)", ylabel=r"$P(T \geq t)$")
    ax_a.set_ylim(8e-6, 1.2)
    ax_a.legend(frameon=False, loc="lower left", fontsize=8.2, handlelength=2.1)

    labels = {
        "Geometric (discrete exponential)": "Geometric\n(discrete exponential)",
        "pure power law": "pure power law",
        "log-normal": "log-normal",
        "Weibull": "Weibull",
        "conventional truncated power law": "conv. truncated power law",
        "generalised-cutoff power law": "gen.-cutoff power law",
    }
    y = np.arange(len(contrasts))
    values = contrasts["delta_logL_per_run_vs_best"].to_numpy(float)
    low = contrasts["ci_low"].to_numpy(float)
    high = contrasts["ci_high"].to_numpy(float)
    colors = [GREEN if np.isclose(value, 0.0) else BLACK for value in values]
    for yi, value, lo, hi, color in zip(y, values, low, high, colors):
        ax_b.errorbar(value, yi, xerr=[[value - lo], [hi - value]], fmt="o", color=color, ecolor=color, capsize=2, ms=4.2)
    ax_b.axvline(0, color=GREY, ls=":", lw=1.2)
    ax_b.set_yticks(y, [labels.get(name, name) for name in contrasts["model"]])
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"LOFO $\Delta$ log L per run vs best")
    ax_b.spines["left"].set_position(("outward", 2))
    style_axes(ax_a, ax_b)
    draw_panel_letter(ax_a, "A", x=-0.16, y=1.05, fontsize=13)
    draw_panel_letter(ax_b, "B", x=-0.16, y=1.05, fontsize=13)
    return fig, save(fig, "figureS1_duration_model_comparison", close_figure=close_figure)


def figure_s2(*, close_figure: bool = True) -> tuple[plt.Figure, list[Path]]:
    source = pd.read_csv(SOURCE / "figureS2_segmentation_robustness_source.csv")
    styles = {
        "low_speed_guard_0p25": (RED, r"0.25 m s$^{-1}$ turn guard"),
        "low_speed_guard_0p50": (BLUE, r"0.50 m s$^{-1}$ turn guard"),
        "primary_recomputed": (BLACK, "Primary segmentation"),
        "rolling_median_3": (GREY, "Three-sample rolling median"),
    }
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.4, 3.65), gridspec_kw={"wspace": 0.30})
    for variant, (color, label) in styles.items():
        hazard = source.loc[source["variant"].eq(variant) & source["quantity"].eq("hazard")].sort_values("age_mid_s")
        ax_a.plot(hazard["age_mid_s"], hazard["termination_probability"], color=color, lw=2.0, label=label)
        ax_a.fill_between(hazard["age_mid_s"], hazard["ci_low"], hazard["ci_high"], color=color, alpha=0.11, linewidth=0)
        ccdf = source.loc[source["variant"].eq(variant) & source["quantity"].eq("ccdf")].sort_values("duration_s")
        ax_b.plot(ccdf["duration_s"], ccdf["ccdf"], color=color, lw=2.0, label=label)
    ax_a.set(xlabel="Run age (s)", ylabel="Interval termination probability", xlim=(0, 35), ylim=(0, 0.44))
    ax_a.legend(frameon=False, loc="upper right", fontsize=8.2)
    ax_b.set(xlabel="Run duration $T$ (s)", ylabel=r"$P(T \geq t)$", yscale="log", xlim=(1, 60), ylim=(1e-4, 1.05))
    ax_b.legend(frameon=False, loc="lower left", fontsize=8.2)
    style_axes(ax_a, ax_b)
    draw_panel_letter(ax_a, "A", x=-0.15, y=1.05, fontsize=13)
    draw_panel_letter(ax_b, "B", x=-0.15, y=1.05, fontsize=13)
    return fig, save(fig, "figureS2_segmentation_robustness_final", close_figure=close_figure)


def main() -> None:
    configure_paper_plotting(base=10)
    _fig_s1, paths_s1 = figure_s1()
    _fig_s2, paths_s2 = figure_s2()
    paths = [*paths_s1, *paths_s2]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
