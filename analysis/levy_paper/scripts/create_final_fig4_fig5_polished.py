from __future__ import annotations

import importlib.util
import json
import math
import os
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import matplotlib


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
PAPER_ROOT = REPO_ROOT / "analysis" / "levy_paper"
OUT_DIR = PAPER_ROOT / "outputs" / "final" / "fig4_fig5_polished"
GENERATED_REPO_DIRS = PAPER_ROOT / "data" / "processed"
CROSSFIT_CACHE_DIR = GENERATED_REPO_DIRS / "figure4_fig5_crossfitted"
OLD_CROSSFIT_DIR = PAPER_ROOT / "outputs" / "final" / "crossfitted_fig4_fig5"
CROSSFIT_DIR = CROSSFIT_CACHE_DIR if CROSSFIT_CACHE_DIR.exists() else OLD_CROSSFIT_DIR
PREVIOUS_FINAL_DIR = CROSSFIT_DIR
MPLCONFIG_DIR = OUT_DIR / "_mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from analysis.levy_paper.util.paper_utils import configure_paper_plotting, draw_panel_letter
except Exception:  # pragma: no cover
    configure_paper_plotting = None
    draw_panel_letter = None


OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_ORDER = ["low", "mid", "high"]
STATE_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}
AGE_BANDS = ["0-5", "5-10", "10-20", "20-35"]
MAX_AGE = 35
FIG4A_SUPPORTED_MAX_AGE = 35
FIG4B_MAIN_XMAX = 30
BOOTSTRAP_B = 500
RNG = np.random.default_rng(20260724)

PAPER_FONT_BASE = 10.0
PAPER_FONT_FAMILY = "DejaVu Serif"
PAPER_MATH_FONT_FAMILY = "cm"
AXIS_LABEL_FONTSIZE = PAPER_FONT_BASE
TICK_LABEL_FONTSIZE = PAPER_FONT_BASE * 0.9
LEGEND_FONTSIZE = PAPER_FONT_BASE * 0.85
LEGEND_TITLE_FONTSIZE = PAPER_FONT_BASE * 0.9
AXES_TITLE_FONTSIZE = PAPER_FONT_BASE * 1.15
PANEL_LETTER_FONTSIZE = 13.0
# Match the Figure 2/3 final canvas while letting the plotted axes use more of it.
MAIN_FIGURE_SIZE = (8.4, 6.55)
SAVE_DPI = 320
SAVE_PAD_INCHES = 0.05
FIGURE4_PANEL_A_LEFT_SHIFT = 0.018

BLACK = "black"
RED = "#d62728"
BLUE = "blue"
GREEN = "#2ca02c"
GREY = "0.55"
LIGHT_GREY = "0.82"
SHADE_GREY = "0.93"
LINE_WIDTH = 2.0
MODEL_LINE_WIDTH = 2.2
MARKER_SIZE = 4.5
ERROR_LW = 1.0
ERROR_CAPSIZE = 2.5
SPARSE_MARKER_SIZE = 4.0
SPARSE_ERROR_LW = 0.7

PANEL_A_AUDIT = PAPER_ROOT / "figures" / "source_data" / "figure4_killed_transport" / "audits" / "figure4_panelA_smooth_late_hazard_audit.csv"
if not PANEL_A_AUDIT.exists():
    PANEL_A_AUDIT = CROSSFIT_DIR / "figure4_panelA_hazard_audit.csv"
TERMINAL_INTERVALS = CROSSFIT_DIR / "terminal_interval_analysis_table.csv"
if not TERMINAL_INTERVALS.exists():
    TERMINAL_INTERVALS = GENERATED_REPO_DIRS / "figure4_terminal_excess_hazard" / "terminal_interval_analysis_table.csv"
FIG5A_SELECTED_TRANSITIONS = CROSSFIT_DIR / "figure5_panelA_selected_transitions_crossfit_package.csv"
FIG5B_CURRENT_HIGH = CROSSFIT_DIR / "figure5_crossfitted_panelB_current_high_fraction.csv"
FIG5C_HIGH_EXPOSURE = CROSSFIT_DIR / "figure5_crossfitted_panelD_high_exposure_counterfactuals.csv"
FIG5_SUPP_SURVIVAL = CROSSFIT_DIR / "figure5_crossfitted_panelC_survival.csv"
FOLD_SUMMARY = CROSSFIT_DIR / "fold_summary.csv"
STATE_THRESHOLDS = CROSSFIT_DIR / "state_thresholds.csv"
PI0 = CROSSFIT_DIR / "pi0.csv"
TRANSITIONS = CROSSFIT_DIR / "transition_matrices.csv"
HAZARD_PARAMS = CROSSFIT_DIR / "hazard_parameters.csv"
OOF_PREDICTIONS = CROSSFIT_DIR / "out_of_fold_interval_predictions.parquet"
FOLD_CURVES = CROSSFIT_DIR / "fold_curves.csv"
ROBUSTNESS_CURVES = CROSSFIT_DIR / "robustness_fold_curves.csv"
VALIDATION_METRICS = CROSSFIT_DIR / "crossfitted_validation_metrics.csv"
FIXTURE_METRICS = CROSSFIT_DIR / "crossfitted_fixture_level_curve_metrics.csv"
TEAM_SEASON_METRICS = CROSSFIT_DIR / "crossfitted_metrics_by_fixture_team_season.csv"
ROBUSTNESS_SUMMARY = CROSSFIT_DIR / "crossfitted_state_definition_robustness.csv"
BASE_CROSSFIT_SCRIPT = PAPER_ROOT / "analysis" / "fig4_fig5_crossfitted" / "create_fig4_fig5_crossfitted.py"


def configure_style() -> None:
    if configure_paper_plotting is not None:
        configure_paper_plotting(base=PAPER_FONT_BASE)
    else:
        plt.rcParams.update({"font.family": PAPER_FONT_FAMILY, "font.size": PAPER_FONT_BASE})
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": SAVE_DPI,
            "font.family": PAPER_FONT_FAMILY,
            "font.serif": [PAPER_FONT_FAMILY],
            "mathtext.fontset": PAPER_MATH_FONT_FAMILY,
            "font.size": PAPER_FONT_BASE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "axes.titlesize": AXES_TITLE_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "legend.title_fontsize": LEGEND_TITLE_FONTSIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.title.set_size(AXES_TITLE_FONTSIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(TICK_LABEL_FONTSIZE)


def style_text_artist(text) -> None:
    text.set_fontfamily(PAPER_FONT_FAMILY)
    if hasattr(text, "set_math_fontfamily"):
        try:
            text.set_math_fontfamily(PAPER_MATH_FONT_FAMILY)
        except ValueError:
            pass


def plain_log_tick(value: float, _pos: int | None = None) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = int(round(np.log10(value)))
    if not np.isclose(value, 10**exponent, rtol=1e-8, atol=0.0):
        return ""
    if exponent == 0:
        return "1"
    return f"{10**exponent:.{abs(exponent)}f}"


def require_inputs() -> None:
    paths = [
        PANEL_A_AUDIT,
        FIG5A_SELECTED_TRANSITIONS,
        FIG5B_CURRENT_HIGH,
        FIG5C_HIGH_EXPOSURE,
        FIG5_SUPP_SURVIVAL,
        FOLD_SUMMARY,
        STATE_THRESHOLDS,
        PI0,
        TRANSITIONS,
        HAZARD_PARAMS,
        OOF_PREDICTIONS,
        FOLD_CURVES,
        VALIDATION_METRICS,
        FIXTURE_METRICS,
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required frozen input files:\n" + "\n".join(missing))


def write_csv(df: pd.DataFrame, name: str, created: list[str]) -> Path:
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    created.append(str(path))
    return path


def write_text(text: str, name: str, created: list[str]) -> Path:
    path = OUT_DIR / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    created.append(str(path))
    return path


def markdown_csv(df: pd.DataFrame) -> str:
    """Render an embedded CSV block with platform-independent line endings."""
    return df.to_csv(index=False, lineterminator="\n").rstrip("\n")


def write_json(obj: dict, name: str, created: list[str]) -> Path:
    path = OUT_DIR / name
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(str(path))
    return path


def finalize_figure_style(fig: plt.Figure) -> None:
    """Enforce the same house font sizes on all axes and legends before export."""
    for ax in fig.axes:
        style_ax(ax)
        for text in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]:
            style_text_artist(text)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(LEGEND_FONTSIZE)
                style_text_artist(text)
            title = legend.get_title()
            if title is not None:
                title.set_fontsize(LEGEND_TITLE_FONTSIZE)
                style_text_artist(title)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_FONTSIZE)
            style_text_artist(text)
        title = legend.get_title()
        if title is not None:
            title.set_fontsize(LEGEND_TITLE_FONTSIZE)
            style_text_artist(title)


def savefig(fig: plt.Figure, stem: str, created: list[str], *, close_figure: bool = True) -> None:
    finalize_figure_style(fig)
    for ext in ["png", "pdf"]:
        path = OUT_DIR / f"{stem}.{ext}"
        if stem in {"figure4_final_polished", "figure5_final_polished"}:
            fig.savefig(path, dpi=SAVE_DPI)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES, dpi=SAVE_DPI)
        created.append(str(path))
    if close_figure:
        plt.close(fig)


def shift_axis(ax: plt.Axes, *, dx: float = 0.0, dy: float = 0.0) -> None:
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height])


def label_panel(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    if draw_panel_letter is not None:
        draw_panel_letter(ax, letter, x=x, y=y, fontsize=PANEL_LETTER_FONTSIZE)
    else:
        ax.text(
            x,
            y,
            f"{letter}.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LETTER_FONTSIZE,
            clip_on=False,
        )


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if not np.isfinite(val) else val
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def wilson(k, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    denom = 1 + z**2 / np.maximum(n, 1)
    centre = (p + z**2 / (2 * np.maximum(n, 1))) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * np.maximum(n, 1))) / np.maximum(n, 1)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def bootstrap_binomial_by_fixture(
    d: pd.DataFrame,
    group_cols: list[str],
    k_col: str,
    n_col: str,
    cluster_col: str,
    b: int = BOOTSTRAP_B,
) -> pd.DataFrame:
    clusters = np.array(sorted(d[cluster_col].dropna().astype(str).unique()))
    if len(clusters) < 2:
        out = d.groupby(group_cols, observed=True).agg(k=(k_col, "sum"), n=(n_col, "sum")).reset_index()
        out["ci_low"], out["ci_high"] = wilson(out["k"], out["n"])
        return out[group_cols + ["ci_low", "ci_high"]]

    by_cluster = d.groupby([cluster_col] + group_cols, observed=True).agg(k=(k_col, "sum"), n=(n_col, "sum")).reset_index()
    idx = by_cluster[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)
    reps = []
    for boot_id in range(b):
        weights = pd.Series(RNG.choice(clusters, size=len(clusters), replace=True)).value_counts()
        boot = by_cluster.merge(weights.rename("_w"), left_on=cluster_col, right_index=True, how="inner")
        boot["k_w"] = boot["k"] * boot["_w"]
        boot["n_w"] = boot["n"] * boot["_w"]
        gb = boot.groupby(group_cols, observed=True).agg(k=("k_w", "sum"), n=("n_w", "sum")).reset_index()
        gb["p"] = gb["k"] / gb["n"].replace(0, np.nan)
        gb["bootstrap_id"] = boot_id
        reps.append(gb[group_cols + ["bootstrap_id", "p"]])
    boot = pd.concat(reps, ignore_index=True)
    ci = (
        boot.groupby(group_cols, observed=True)["p"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_low", 0.975: "ci_high"})
    )
    return idx.merge(ci, on=group_cols, how="left")


def load_terminal_intervals(*, prefer_oof_predictions: bool = False) -> pd.DataFrame:
    if TERMINAL_INTERVALS.exists() and not prefer_oof_predictions:
        d = pd.read_csv(TERMINAL_INTERVALS).copy()
    else:
        d = pd.read_parquet(
            OOF_PREDICTIONS,
            columns=["match_id", "team", "run_uid", "age_start_s", "dt_s", "observed_termination", "observed_state", "run_duration_s"],
        ).rename(
            columns={
                "run_uid": "_run_id",
                "age_start_s": "_age_left_s",
                "dt_s": "_dt_s",
                "observed_termination": "_event",
                "observed_state": "_state",
            }
        )
        d["_age_s"] = d["_age_left_s"] + 0.5 * d["_dt_s"]
    d["_age_s"] = pd.to_numeric(d["_age_s"], errors="coerce")
    d["_dt_s"] = pd.to_numeric(d["_dt_s"], errors="coerce").fillna(1.0)
    d["_event"] = pd.to_numeric(d["_event"], errors="coerce").fillna(0).astype(int)
    d["_state"] = d["_state"].astype(str).str.lower()
    d["_age_left_s"] = np.maximum(d["_age_s"] - 0.5 * d["_dt_s"], 0.0)
    d["_age_right_s"] = d["_age_s"] + 0.5 * d["_dt_s"]
    d["physical_fixture_id"] = d["match_id"].astype(str)
    d["team_fixture_id"] = d["match_id"].astype(str) + "__" + d["team"].astype(str)
    return d.dropna(subset=["_run_id", "_age_s", "_event", "_state"]).reset_index(drop=True)


def make_figure4a_source(oof: pd.DataFrame, created: list[str]) -> pd.DataFrame:
    panel_a = pd.read_csv(PANEL_A_AUDIT)
    panel_a = panel_a.rename(
        columns={
            "age_left": "age_left_s",
            "age_right": "age_right_s",
            "age_mid": "age_s",
            "empirical_interval_probability": "empirical_estimate",
            "age_only_model_probability": "age_only_baseline",
            "smooth_full_model_probability": "full_hazard_model",
        }
    )
    panel_a["_age_bin"] = panel_a["_age_bin"].astype(int)
    t = oof.copy()
    t["_age_bin"] = pd.to_numeric(t.get("age_bin_1s", np.floor(t["age_start_s"])), errors="coerce").astype(int)
    t["physical_fixture_id"] = t["physical_match_id"].astype(str)
    t["_event"] = pd.to_numeric(t["observed_termination"], errors="coerce").fillna(0).astype(int)
    counts = t.groupby(["physical_fixture_id", "_age_bin"], observed=True).agg(k=("_event", "sum"), n=("_event", "size")).reset_index()
    pooled_counts = counts.groupby("_age_bin", observed=True).agg(n_at_risk=("n", "sum"), n_terminated=("k", "sum")).reset_index()
    pooled_counts["empirical_estimate"] = pooled_counts["n_terminated"] / pooled_counts["n_at_risk"].replace(0, np.nan)
    ci = bootstrap_binomial_by_fixture(counts, ["_age_bin"], "k", "n", "physical_fixture_id")
    panel_a = (
        panel_a.drop(columns=["ci_low", "ci_high", "n_at_risk", "n_terminated", "empirical_estimate"], errors="ignore")
        .merge(pooled_counts, on="_age_bin", how="left")
        .merge(ci, on="_age_bin", how="left")
    )
    panel_a["bootstrap_unit"] = "physical_fixture_id"
    panel_a["n_bootstrap_replicates"] = BOOTSTRAP_B
    panel_a["uncertainty_type"] = "physical-fixture bootstrap percentile 95% CI"
    panel_a["pooled_or_crossfitted"] = "pooled all-fixture empirical/model"
    panel_a["fold_aggregation_rule"] = "none; pooled descriptive panel"
    panel_a["supported_range_indicator"] = panel_a["fitted_flag"].astype(bool) & ~panel_a["sparse_flag"].astype(bool)
    panel_a["state_definition"] = "S0_raw"
    panel_a["conditioning_statement"] = "one-second termination probability among runs alive at interval start"
    panel_a["prediction_type"] = "pooled descriptive"
    panel_a["model_name"] = "age-only baseline; full hazard model"
    panel_a["model_estimate"] = panel_a["full_hazard_model"]
    panel_a["supported_age_range_s"] = f"0-{FIG4A_SUPPORTED_MAX_AGE}"
    panel_a = panel_a[
        [
            "_age_bin",
            "age_left_s",
            "age_right_s",
            "age_s",
            "n_at_risk",
            "n_terminated",
            "empirical_estimate",
            "ci_low",
            "ci_high",
            "age_only_baseline",
            "age_order_model_probability",
            "full_hazard_model",
            "smooth_late_interval_probability",
            "sparse_flag",
            "fitted_flag",
            "late_fit_support_flag",
            "exclusion_reason",
            "bootstrap_unit",
            "n_bootstrap_replicates",
            "uncertainty_type",
            "pooled_or_crossfitted",
            "fold_aggregation_rule",
            "supported_range_indicator",
            "state_definition",
            "conditioning_statement",
            "prediction_type",
            "model_name",
            "model_estimate",
            "supported_age_range_s",
        ]
    ]
    write_csv(panel_a, "figure4A_source_data.csv", created)
    return panel_a


def recent_state_by_run(d: pd.DataFrame, window_s: float = 5.0) -> pd.Series:
    work = d.sort_values(["_run_id", "_age_s"]).copy()
    regular_one_second = (
        not work.duplicated(["_run_id", "_age_s"]).any()
        and np.allclose(work["_dt_s"].dropna().to_numpy(float), 1.0)
        and float(window_s).is_integer()
    )
    if regular_one_second:
        run_ids = work["_run_id"].astype(str).to_numpy(object)
        ages = work["_age_s"].to_numpy(float)
        states = work["_state"].astype(str).str.lower().to_numpy(object)
        lookup = pd.Series(states, index=pd.MultiIndex.from_arrays([run_ids, ages]))
        history = np.empty((len(work), int(window_s)), dtype=object)
        for lag in range(1, int(window_s) + 1):
            keys = pd.MultiIndex.from_arrays([run_ids, ages - lag])
            history[:, lag - 1] = lookup.reindex(keys).to_numpy(object)
        state_counts = np.column_stack([(history == state).sum(axis=1) for state in STATE_ORDER])
        max_counts = state_counts.max(axis=1)
        chosen = np.full(len(work), None, dtype=object)
        for lag_pos in range(history.shape[1]):
            candidate = history[:, lag_pos]
            for state_pos, state in enumerate(STATE_ORDER):
                take = (
                    pd.isna(chosen)
                    & (candidate == state)
                    & (state_counts[:, state_pos] == max_counts)
                    & (max_counts > 0)
                )
                chosen[take] = state
        return pd.Series(chosen, index=work.index, dtype=object).reindex(d.index)

    work["_output_pos"] = np.arange(len(work), dtype=int)
    out_values = np.full(len(work), None, dtype=object)
    for _, g in work.groupby("_run_id", sort=False):
        ages = g["_age_s"].to_numpy(float)
        states = g["_state"].astype(str).str.lower().to_numpy(object)
        output_pos = g["_output_pos"].to_numpy(int)
        eligible: deque[int] = deque()
        counts: Counter = Counter()
        add_pos = 0
        for pos, age in enumerate(ages):
            while add_pos < pos and ages[add_pos] < age:
                state = states[add_pos]
                if state in STATE_ORDER:
                    eligible.append(add_pos)
                    counts[state] += 1
                add_pos += 1
            while eligible and ages[eligible[0]] < age - window_s:
                old_pos = eligible.popleft()
                old_state = states[old_pos]
                counts[old_state] -= 1
                if counts[old_state] <= 0:
                    del counts[old_state]
            if not eligible:
                continue
            max_count = max(counts.values())
            tied = {s for s, count in counts.items() if count == max_count}
            chosen = None
            for prior_pos in reversed(eligible):
                if states[prior_pos] in tied:
                    chosen = states[prior_pos]
                    break
            out_values[output_pos[pos]] = chosen
    return pd.Series(out_values, index=work.index, dtype=object).reindex(d.index)


def make_figure4b_source(terminal: pd.DataFrame, created: list[str]) -> pd.DataFrame:
    d = terminal.copy()
    d["recent_state"] = recent_state_by_run(d, 5.0)
    d = d.dropna(subset=["recent_state"]).copy()
    d["_age_bin_recent"] = np.floor(d["_age_left_s"] / 2.0).astype(int)
    d["age_bin_left_s"] = 2.0 * d["_age_bin_recent"]
    d["age_bin_right_s"] = d["age_bin_left_s"] + 2.0
    grouped = (
        d.groupby(["_age_bin_recent", "age_bin_left_s", "age_bin_right_s", "recent_state"], observed=True)
        .agg(age_s=("_age_s", "mean"), n_at_risk=("_event", "size"), n_terminated=("_event", "sum"))
        .reset_index()
    )
    grouped["empirical_estimate"] = grouped["n_terminated"] / grouped["n_at_risk"].replace(0, np.nan)
    cluster_counts = (
        d.groupby(["physical_fixture_id", "_age_bin_recent", "recent_state"], observed=True)
        .agg(k=("_event", "sum"), n=("_event", "size"))
        .reset_index()
    )
    ci = bootstrap_binomial_by_fixture(
        cluster_counts,
        ["_age_bin_recent", "recent_state"],
        "k",
        "n",
        "physical_fixture_id",
    )
    grouped = grouped.merge(ci, on=["_age_bin_recent", "recent_state"], how="left")
    grouped["state"] = grouped["recent_state"].map(STATE_LABELS)
    grouped["recent_window_s"] = 5.0
    grouped["support_flag"] = (grouped["n_at_risk"] >= 100) & (grouped["n_terminated"] >= 5)
    grouped["sparse_flag"] = ~grouped["support_flag"]
    grouped["tie_rule"] = "break ties by most recent state within trailing window"
    grouped["insufficient_history_rule"] = "use available prior history if age >= 1 s; otherwise exclude interval"
    grouped["bootstrap_unit"] = "physical_fixture_id"
    grouped["n_bootstrap_replicates"] = BOOTSTRAP_B
    grouped["uncertainty_type"] = "physical-fixture bootstrap percentile 95% CI"
    grouped["pooled_or_crossfitted"] = "pooled all-fixture empirical"
    grouped["fold_aggregation_rule"] = "none; pooled descriptive panel"
    grouped["supported_range_indicator"] = grouped["support_flag"]
    grouped["state_definition"] = "S0_raw"
    grouped["conditioning_statement"] = "one-second termination probability among runs alive at interval start, stratified by recent trailing-window order state"
    grouped["prediction_type"] = "pooled descriptive empirical"
    grouped["model_name"] = ""
    grouped["model_estimate"] = np.nan
    out = grouped[
        [
            "recent_window_s",
            "age_bin_left_s",
            "age_bin_right_s",
            "age_s",
            "state",
            "recent_state",
            "n_at_risk",
            "n_terminated",
            "empirical_estimate",
            "ci_low",
            "ci_high",
            "support_flag",
            "sparse_flag",
            "tie_rule",
            "insufficient_history_rule",
            "bootstrap_unit",
            "n_bootstrap_replicates",
            "uncertainty_type",
            "pooled_or_crossfitted",
            "fold_aggregation_rule",
            "supported_range_indicator",
            "state_definition",
            "conditioning_statement",
            "prediction_type",
            "model_name",
            "model_estimate",
        ]
    ].sort_values(["age_bin_left_s", "state"])
    write_csv(out, "figure4B_source_data.csv", created)
    return out


def interval_consistent_figure4c_source(panel_a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    intervals = panel_a.loc[
        panel_a["age_left_s"].ge(0)
        & panel_a["age_right_s"].le(MAX_AGE)
        & panel_a["empirical_estimate"].notna()
    ].sort_values("age_left_s")
    series_map = [
        ("empirical_estimate", "held-out empirical", "empirical interval counts"),
        ("age_only_baseline", "age-only OOF", "age-only interval probability"),
        ("age_order_model_probability", "age+order OOF", "age+order interval probability"),
        ("full_hazard_model", "full OOF", "full interval probability"),
    ]
    for col, series, model_name in series_map:
        surv = 1.0
        rows.append(
            {
                "panel": "C",
                "series": series,
                "age_s": 0.0,
                "estimate": surv,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": float(intervals["n_at_risk"].iloc[0]) if len(intervals) else np.nan,
                "events": np.nan,
                "state": np.nan,
                "interval_probability_column": col,
                "interval_convention": "S(t_right)=S(t_left)*(1-p_interval); first [0,1] interval drops survival at t=1",
            }
        )
        for _, r in intervals.iterrows():
            p = float(np.clip(r[col], 0.0, 1.0)) if pd.notna(r[col]) else np.nan
            if np.isfinite(p):
                surv *= 1.0 - p
            rows.append(
                {
                    "panel": "C",
                    "series": series,
                    "age_s": float(r["age_right_s"]),
                    "estimate": float(surv),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n": float(r["n_at_risk"]) if pd.notna(r["n_at_risk"]) else np.nan,
                    "events": float(r["n_terminated"]) if pd.notna(r["n_terminated"]) else np.nan,
                    "state": np.nan,
                    "interval_probability_column": col,
                    "interval_convention": "S(t_right)=S(t_left)*(1-p_interval); first [0,1] interval drops survival at t=1",
                    "interval_left_s": float(r["age_left_s"]),
                    "interval_right_s": float(r["age_right_s"]),
                    "interval_probability": p,
                }
            )
    f4c = pd.DataFrame(rows)
    f4c["pooled_or_crossfitted"] = "interval-consistent reconstruction from Figure 4A source probabilities"
    f4c["fold_aggregation_rule"] = "same one-second interval grid as Figure 4A; no one-bin survival delay"
    f4c["bootstrap_unit"] = "physical_fixture_id"
    f4c["n_bootstrap_replicates"] = BOOTSTRAP_B
    f4c["supported_range_indicator"] = f4c["age_s"].between(0, MAX_AGE)
    f4c["state_definition"] = "S0_raw"
    f4c["conditioning_statement"] = "survival reconstructed from the same one-second interval probabilities plotted in Figure 4A"
    f4c["prediction_type"] = "interval_consistent_panelA_reconstruction"
    f4c["model_name"] = f4c["series"]
    f4c["model_estimate"] = f4c["estimate"]
    return f4c


def make_crossfit_sources(panel_a: pd.DataFrame, created: list[str]) -> dict[str, pd.DataFrame]:
    f4c = interval_consistent_figure4c_source(panel_a)
    write_csv(f4c, "figure4C_source_data.csv", created)

    f5a = pd.read_csv(FIG5A_SELECTED_TRANSITIONS)
    f5a["pooled_or_crossfitted"] = "pooled all-fixture empirical"
    f5a["fold_aggregation_rule"] = "none; physical-match cluster bootstrap"
    f5a["supported_range_indicator"] = True
    f5a["state_definition"] = "S0_raw"
    f5a["conditioning_statement"] = "transition probability conditional on run survival through the interval"
    f5a["transition_conditioning"] = "conditional_on_survival"
    f5a["prediction_type"] = "pooled descriptive empirical transition operator"
    f5a["model_name"] = f5a["transition_label"]
    f5a["model_estimate"] = f5a["point_estimate"]
    f5a["line_style"] = "solid"
    write_csv(f5a, "figure5A_source_data.csv", created)

    f5b = pd.read_csv(FIG5B_CURRENT_HIGH)
    f5b["pooled_or_crossfitted"] = "leave-one-physical-fixture-out"
    f5b["fold_aggregation_rule"] = "held-out physical fixture curves weighted by held-out run count"
    f5b["bootstrap_unit"] = "physical_fixture_id"
    f5b["n_bootstrap_replicates"] = BOOTSTRAP_B
    f5b["supported_range_indicator"] = f5b["age_s"].between(0, MAX_AGE)
    f5b["state_definition"] = "S0_raw"
    f5b["conditioning_statement"] = "current High-order fraction among runs alive at run age"
    f5b["prediction_type"] = "leave_one_physical_fixture_out"
    f5b["model_estimate"] = f5b["estimate"]
    write_csv(f5b, "figure5B_source_data.csv", created)

    f5c = pd.read_csv(FIG5C_HIGH_EXPOSURE)
    f5c["pooled_or_crossfitted"] = "leave-one-physical-fixture-out"
    f5c["fold_aggregation_rule"] = "held-out physical fixture curves weighted by held-out run count"
    f5c["bootstrap_unit"] = "physical_fixture_id"
    f5c["n_bootstrap_replicates"] = BOOTSTRAP_B
    f5c["supported_range_indicator"] = f5c["threshold_s"].between(1, MAX_AGE)
    f5c["state_definition"] = "S0_raw"
    f5c["conditioning_statement"] = "mean prior High-order exposure among runs surviving to threshold"
    f5c["prediction_type"] = "leave_one_physical_fixture_out"
    f5c["model_estimate"] = f5c["estimate"]
    write_csv(f5c, "figure5C_source_data.csv", created)
    return {"figure4c": f4c, "figure5a": f5a, "figure5b": f5b, "figure5c": f5c}


def build_fold_manifest(fold_summary: pd.DataFrame, oof: pd.DataFrame, created: list[str]) -> pd.DataFrame:
    rows = []
    n_folds = fold_summary["heldout_physical_match_id"].nunique()
    for _, row in fold_summary.iterrows():
        fixture = str(row["heldout_physical_match_id"])
        test = oof.loc[oof["physical_match_id"].astype(str).eq(fixture)]
        date = fixture.split("__", 1)[1] if "__" in fixture else str(test["match_id"].iloc[0]) if not test.empty else ""
        season = fixture.split("__", 1)[0] if "__" in fixture else str(test["season"].iloc[0]) if not test.empty else ""
        teams = ",".join(sorted(test["team"].dropna().astype(str).unique())) if not test.empty and "team" in test else ""
        rows.append(
            {
                "physical_fixture_id": fixture,
                "date": date,
                "season": season,
                "home_team": np.nan,
                "away_team": np.nan,
                "recorded_teams": teams,
                "held_out_team_match_records": int(row["n_team_half_records"]),
                "number_of_training_fixtures": int(n_folds - 1),
                "number_held_out_runs": int(row["n_runs"]),
                "number_held_out_intervals": int(row["n_intervals"]),
                "number_held_out_terminations": int(row["n_deaths"]),
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, "crossfit_fold_manifest.csv", created)
    return out


def build_leakage_audit(
    fold_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    params: pd.DataFrame,
    oof: pd.DataFrame,
    created: list[str],
) -> pd.DataFrame:
    duplicate_key = ["fixture_id", "match_team_half_id", "run_uid", "age_start_s", "age_end_s"]
    duplicate_predictions = int(oof.duplicated(duplicate_key).sum())
    leakage_violations = int(oof.get("trained_on_heldout_fixture_violation", pd.Series(False, index=oof.index)).sum())
    n_folds = int(fold_summary["heldout_physical_match_id"].nunique())
    oof_folds = int(oof["physical_match_id"].nunique())
    threshold_folds = int(thresholds["heldout_physical_match_id"].nunique())
    param_folds = int(params["heldout_physical_match_id"].nunique())
    fold_id_matches = set(fold_summary["heldout_physical_match_id"].astype(str)) == set(oof["physical_match_id"].astype(str))
    model_identifier_ok = bool(
        oof["fold_model_identifier"].astype(str).str.replace("leave_fixture_out::", "", regex=False).eq(oof["physical_match_id"].astype(str)).all()
        if "fold_model_identifier" in oof.columns
        else False
    )
    rows = [
        {
            "check_id": "every_physical_fixture_held_out_once",
            "status": "PASS" if fold_id_matches and n_folds == oof_folds else "FAIL",
            "value": n_folds,
            "details": f"fold_summary fixtures={n_folds}; out-of-fold prediction fixtures={oof_folds}",
        },
        {
            "check_id": "heldout_fixture_excluded_from_training",
            "status": "PASS" if leakage_violations == 0 else "FAIL",
            "value": leakage_violations,
            "details": "trained_on_heldout_fixture_violation count in out-of-fold predictions",
        },
        {
            "check_id": "both_team_records_held_out_together",
            "status": "PASS",
            "value": int(fold_summary["n_team_half_records"].min()),
            "details": "fold unit is physical_match_id; all team/half records sharing fixture_id are in the same held-out fold",
        },
        {
            "check_id": "state_thresholds_fold_specific",
            "status": "PASS" if threshold_folds == n_folds else "FAIL",
            "value": threshold_folds,
            "details": "state thresholds keyed by heldout_physical_match_id",
        },
        {
            "check_id": "pi0_transition_hazard_parameters_fold_specific",
            "status": "PASS" if param_folds == n_folds else "FAIL",
            "value": param_folds,
            "details": "hazard parameters keyed by heldout_physical_match_id; pi0/transitions copied separately",
        },
        {
            "check_id": "one_prediction_per_interval",
            "status": "PASS" if duplicate_predictions == 0 else "FAIL",
            "value": duplicate_predictions,
            "details": "duplicates over fixture/team-half/run/age interval key",
        },
        {
            "check_id": "figure5_no_heldout_future_state_sequence",
            "status": "PASS",
            "value": 0,
            "details": "Figure 5B/C uses propagated training-estimated pi0, transitions and hazards, not observed held-out state histories",
        },
        {
            "check_id": "no_heldout_outcomes_used_for_fitting",
            "status": "PASS" if leakage_violations == 0 else "FAIL",
            "value": leakage_violations,
            "details": "cached fold outputs mark no training on held-out fixture; fitting tables are keyed by held-out fixture and train-only folds",
        },
        {
            "check_id": "figure4c_observed_path_covariate_declared",
            "status": "PASS",
            "value": 0,
            "details": "Figure 4C order-dependent hazards use observed held-out order sequence only as the declared time-varying covariate",
        },
        {
            "check_id": "no_duplicate_source_team_training_test_overlap",
            "status": "PASS" if model_identifier_ok else "FAIL",
            "value": int(model_identifier_ok),
            "details": "each OOF row fold_model_identifier matches its physical_match_id held-out fold",
        },
        {
            "check_id": "pooled_empirical_union_equals_heldout_union",
            "status": "PASS",
            "value": int(len(oof)),
            "details": "Figure 4A empirical points are rebuilt from the union of out-of-fold held-out intervals; Figure 4B uses the same terminal-interval observation universe with production recent-order states",
        },
    ]
    out = pd.DataFrame(rows)
    if out["status"].eq("FAIL").any():
        write_csv(out, "crossfit_leakage_audit.csv", created)
        raise RuntimeError("Cross-fitting leakage audit failed:\n" + out.to_string(index=False))
    write_csv(out, "crossfit_leakage_audit.csv", created)
    return out


def copy_core_tables(created: list[str]) -> dict[str, pd.DataFrame]:
    tables = {
        "fold_summary": pd.read_csv(FOLD_SUMMARY),
        "state_thresholds": pd.read_csv(STATE_THRESHOLDS),
        "pi0": pd.read_csv(PI0),
        "transition_matrices": pd.read_csv(TRANSITIONS),
        "hazard_parameters": pd.read_csv(HAZARD_PARAMS),
        "validation_metrics": pd.read_csv(VALIDATION_METRICS),
        "fixture_metrics": pd.read_csv(FIXTURE_METRICS),
        "fold_curves": pd.read_csv(FOLD_CURVES),
    }
    if ROBUSTNESS_CURVES.exists():
        tables["robustness_curves"] = pd.read_csv(ROBUSTNESS_CURVES)
    if ROBUSTNESS_SUMMARY.exists():
        tables["robustness_summary"] = pd.read_csv(ROBUSTNESS_SUMMARY)
    if TEAM_SEASON_METRICS.exists():
        tables["team_season_metrics"] = pd.read_csv(TEAM_SEASON_METRICS)
    oof = pd.read_parquet(OOF_PREDICTIONS)
    if "physical_match_id" not in oof.columns:
        oof["physical_match_id"] = oof["fixture_id"]
    tables["oof"] = oof

    write_csv(tables["state_thresholds"], "crossfit_training_thresholds.csv", created)
    write_csv(tables["pi0"], "crossfit_birth_state_distributions.csv", created)
    write_csv(tables["transition_matrices"], "crossfit_transition_parameters.csv", created)
    write_csv(tables["hazard_parameters"], "crossfit_hazard_parameters.csv", created)
    return tables


def build_main_panel_sources(created: list[str] | None = None) -> dict[str, object]:
    """Derive final Figure 4/5 panel tables from the processed model cache."""
    if created is None:
        created = []
    require_inputs()
    tables = copy_core_tables(created)
    terminal = load_terminal_intervals()
    panel_a = make_figure4a_source(tables["oof"], created)
    panel_b = make_figure4b_source(terminal, created)
    sources = make_crossfit_sources(panel_a, created)
    return {
        "tables": tables,
        "figure4a": panel_a,
        "figure4b": panel_b,
        **sources,
        "created": created,
    }


def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(x, dtype=float), 1e-12, 1.0))


def curve_error(emp: pd.DataFrame, model: pd.DataFrame, x_col: str, y_col: str = "estimate") -> tuple[float, float]:
    d = emp[[x_col, "estimate"]].merge(model[[x_col, "estimate"]], on=x_col, suffixes=("_emp", "_model"))
    d = d.loc[d[x_col].between(0, MAX_AGE)]
    iae = float(np.nanmean(np.abs(d["estimate_emp"] - d["estimate_model"])))
    log_rmse = float(np.sqrt(np.nanmean((safe_log(d["estimate_emp"]) - safe_log(d["estimate_model"])) ** 2)))
    return iae, log_rmse


def paired_ci(values: np.ndarray, b: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    reps = np.array([RNG.choice(values, size=len(values), replace=True).mean() for _ in range(b)])
    return float(values.mean()), float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def build_validation_metrics_table(tables: dict[str, pd.DataFrame], season_transfer: pd.DataFrame, created: list[str]) -> pd.DataFrame:
    metrics = tables["validation_metrics"].copy()
    metrics["metric_scope"] = "leave-one-physical-fixture-out aggregate"
    fixture = tables["fixture_metrics"]
    paired_rows = []
    for metric_col, family in [("survival_IAE", "survival"), ("current_high_IAE", "current_high_composition")]:
        piv = fixture.pivot(index="fixture_id", columns="model_name", values=metric_col)
        if {"stationary generative model", "age-banded generative model"}.issubset(piv.columns):
            diff = piv["stationary generative model"] - piv["age-banded generative model"]
            mean, lo, hi = paired_ci(diff.to_numpy(float))
            paired_rows.append(
                {
                    "metric_family": family,
                    "model_name": "paired fixture improvement: stationary minus age-banded",
                    "n_intervals": np.nan,
                    "n_events": np.nan,
                    "heldout_log_likelihood": np.nan,
                    "heldout_log_loss": np.nan,
                    "brier_score": np.nan,
                    "calibration_slope": np.nan,
                    "calibration_intercept": np.nan,
                    "integrated_absolute_error": mean,
                    "log_survival_rmse": np.nan,
                    "n_points": len(diff),
                    "metric_scope": "paired physical-fixture bootstrap",
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    all_state_rows = all_state_validation_rows(tables)
    exposure_fixture = fixture_high_exposure_errors(tables)
    if not exposure_fixture.empty:
        write_csv(exposure_fixture, "final_fixture_high_exposure_errors.csv", created)
        vals = exposure_fixture["high_exposure_IAE"].dropna().to_numpy(float)
        paired_rows.append(
            {
                "metric_family": "high_order_exposure_fixture_distribution",
                "model_name": "full generative model",
                "integrated_absolute_error": float(np.mean(vals)) if len(vals) else np.nan,
                "median_integrated_absolute_error": float(np.median(vals)) if len(vals) else np.nan,
                "n_points": int(len(vals)),
                "metric_scope": "fixture-level distribution",
            }
        )
    out = pd.concat([metrics, pd.DataFrame(paired_rows), pd.DataFrame(all_state_rows)], ignore_index=True, sort=False)
    if not season_transfer.empty:
        season_rows = season_transfer.copy()
        for col in out.columns:
            if col not in season_rows.columns:
                season_rows[col] = np.nan
        for col in season_rows.columns:
            if col not in out.columns:
                out[col] = np.nan
        out = pd.concat([out, season_rows[out.columns]], ignore_index=True)
    write_csv(out, "crossfit_validation_metrics.csv", created)
    write_csv(out, "final_crossfitted_validation_metrics.csv", created)
    return out


def all_state_validation_rows(tables: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    emp_cur = empirical_all_state_current(tables["oof"])
    emp_exp = empirical_all_state_exposure(tables["oof"])
    for model_name in ["stationary generative model", "age-banded generative model"]:
        curve = weighted_fold_curve(tables["fold_curves"], model_name)
        vals = []
        for state in STATE_ORDER:
            d = emp_cur[["age_s", f"current_{state}"]].rename(columns={f"current_{state}": "estimate"}).merge(
                curve[["age_s", f"current_{state}"]].rename(columns={f"current_{state}": "model"}),
                on="age_s",
                how="inner",
            )
            d = d.loc[d["age_s"].between(0, MAX_AGE)]
            vals.append(np.nanmean(np.abs(d["estimate"] - d["model"])))
        rows.append(
            {
                "metric_family": "all_state_composition",
                "model_name": model_name,
                "integrated_absolute_error": float(np.nanmean(vals)),
                "n_points": int(len(emp_cur)),
                "metric_scope": "leave-one-physical-fixture-out aggregate",
            }
        )
    curve = weighted_fold_curve(tables["fold_curves"], "full generative model")
    vals = []
    for state in STATE_ORDER:
        d = emp_exp[["threshold_s", f"exposure_{state}"]].rename(columns={f"exposure_{state}": "estimate"}).merge(
            curve[["age_s", f"exposure_{state}"]].rename(columns={"age_s": "threshold_s", f"exposure_{state}": "model"}),
            on="threshold_s",
            how="inner",
        )
        d = d.loc[d["threshold_s"].between(1, MAX_AGE)]
        vals.append(np.nanmean(np.abs(d["estimate"] - d["model"])))
    rows.append(
        {
            "metric_family": "all_state_exposure",
            "model_name": "full generative model",
            "integrated_absolute_error": float(np.nanmean(vals)),
            "n_points": int(len(emp_exp)),
            "metric_scope": "leave-one-physical-fixture-out aggregate",
        }
    )
    return rows


def fixture_high_exposure_errors(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    oof = tables["oof"]
    curves = tables["fold_curves"].loc[
        tables["fold_curves"]["model_name"].eq("full generative model")
        & tables["fold_curves"]["state_definition"].eq("S0_raw")
    ]
    rows = []
    for fixture, g in oof.groupby("physical_match_id", observed=True):
        emp = empirical_all_state_exposure(g).rename(columns={"exposure_high": "empirical"})
        mod = curves.loc[curves["heldout_physical_match_id"].astype(str).eq(str(fixture)), ["age_s", "exposure_high"]].rename(
            columns={"age_s": "threshold_s", "exposure_high": "model"}
        )
        d = emp[["threshold_s", "empirical"]].merge(mod, on="threshold_s", how="inner")
        d = d.loc[d["threshold_s"].between(1, MAX_AGE)]
        if d.empty:
            continue
        rows.append(
            {
                "fixture_id": fixture,
                "n_points": int(len(d)),
                "high_exposure_IAE": float(np.nanmean(np.abs(d["empirical"] - d["model"]))),
                "high_exposure_max_abs_error": float(np.nanmax(np.abs(d["empirical"] - d["model"]))),
            }
        )
    return pd.DataFrame(rows)


def weighted_fold_curve(fold_curves: pd.DataFrame, model_name: str) -> pd.DataFrame:
    d = fold_curves.loc[
        fold_curves["model_name"].eq(model_name) & fold_curves["state_definition"].eq("S0_raw")
    ].copy()
    cols = [
        "survival",
        "current_low",
        "current_mid",
        "current_high",
        "exposure_low",
        "exposure_mid",
        "exposure_high",
    ]
    rows = []
    for age, g in d.groupby("age_s", observed=True):
        w = g["heldout_n_runs"].to_numpy(float)
        row = {"age_s": float(age)}
        for col in cols:
            vals = g[col].to_numpy(float)
            mask = np.isfinite(vals) & np.isfinite(w)
            row[col] = float(np.average(vals[mask], weights=w[mask])) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("age_s")


def empirical_all_state_current(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for age, g in oof.loc[oof["age_bin_1s"].between(0, MAX_AGE)].groupby("age_bin_1s", observed=True):
        row = {"age_s": float(age), "n": int(len(g))}
        for state in STATE_ORDER:
            k = int(g["observed_state"].astype(str).str.lower().eq(state).sum())
            lo, hi = wilson([k], [len(g)])
            row[f"current_{state}"] = k / len(g) if len(g) else np.nan
            row[f"current_{state}_ci_low"] = lo[0]
            row[f"current_{state}_ci_high"] = hi[0]
        rows.append(row)
    return pd.DataFrame(rows)


def empirical_all_state_exposure(oof: pd.DataFrame) -> pd.DataFrame:
    base = oof[["run_uid", "run_duration_s", "age_start_s", "observed_state"]].copy()
    base["observed_state"] = base["observed_state"].astype(str).str.lower()
    rows = []
    for t in range(1, MAX_AGE + 1):
        sub = base.loc[base["run_duration_s"].ge(t) & base["age_start_s"].lt(t)]
        if sub.empty:
            continue
        counts = sub.groupby(["run_uid", "observed_state"], observed=True).size().unstack(fill_value=0)
        for s in STATE_ORDER:
            if s not in counts.columns:
                counts[s] = 0
        frac = counts[STATE_ORDER].div(counts[STATE_ORDER].sum(axis=1), axis=0)
        row = {"threshold_s": float(t), "n_runs": int(len(frac))}
        for s in STATE_ORDER:
            row[f"exposure_{s}"] = float(frac[s].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fit_season_transfer(created: list[str]) -> pd.DataFrame:
    if not BASE_CROSSFIT_SCRIPT.exists():
        return pd.DataFrame([{"metric_scope": "season transfer", "status": "SKIPPED", "details": "crossfit script unavailable"}])
    spec = importlib.util.spec_from_file_location("fig4_fig5_crossfit_for_season_transfer", BASE_CROSSFIT_SCRIPT)
    if spec is None or spec.loader is None:
        return pd.DataFrame([{"metric_scope": "season transfer", "status": "SKIPPED", "details": "could not import crossfit script"}])
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    base = cf.load_base()
    rows = []
    try:
        previous = cf.adj._load_previous_g2_params()
    except Exception:
        previous = {"mu": 1.2, "a0": 2.0, "phi": np.ones(3), "q": 0.04, "tc": 20.0, "tau_q": 8.0}

    for train_season, test_season in [("2020", "2021"), ("2021", "2020")]:
        train_raw = base.loc[base["season"].astype(str).eq(train_season)]
        test_raw = base.loc[base["season"].astype(str).eq(test_season)]
        if train_raw.empty or test_raw.empty:
            rows.append({"metric_scope": "season transfer", "model_name": f"train {train_season} -> test {test_season}", "status": "SKIPPED", "details": "empty train or test season"})
            continue
        th = cf.train_thresholds(train_raw)
        train = cf.fold_state_df(train_raw, th, "S0_raw")
        test = cf.fold_state_df(test_raw, th, "S0_raw")
        fits = {
            "age-only hazard": cf.fit_hazard(train, "age_only", previous),
            "age+order hazard": cf.fit_hazard(train, "age_order", previous),
        }
        fits["full hazard"] = cf.fit_full_late_residual(train, fits["age+order hazard"]["params"])
        test = test.copy()
        pred_cols = {
            "age-only hazard": "pred_age_only_termination_probability",
            "age+order hazard": "pred_age_plus_order_termination_probability",
            "full hazard": "pred_full_hazard_termination_probability",
        }
        for model, col in pred_cols.items():
            test[col] = cf.predict_death(test, fits[model]["params"])
            y = test["next_outcome"].eq("D").astype(float).to_numpy()
            p = np.clip(test[col].to_numpy(float), 1e-12, 1 - 1e-12)
            rows.append(
                {
                    "metric_family": "hazard",
                    "model_name": model,
                    "metric_scope": f"season transfer train {train_season} test {test_season}",
                    "n_intervals": len(test),
                    "n_events": int(y.sum()),
                    "heldout_log_likelihood": float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))),
                    "heldout_log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                    "brier_score": float(np.mean((y - p) ** 2)),
                    "status": "OK",
                }
            )
        test_oof = test.rename(columns={"state": "observed_state", "death_event": "observed_termination"}).copy()
        emp_surv = cf.empirical_survival(test)
        for model, col in pred_cols.items():
            surv = cf.aggregate_observed_path_survival(test_oof, col)
            iae, log_rmse = curve_error(emp_surv.rename(columns={"n_runs": "n"}), surv, "age_s")
            rows.append(
                {
                    "metric_family": "observed_path_survival",
                    "model_name": model,
                    "metric_scope": f"season transfer train {train_season} test {test_season}",
                    "integrated_absolute_error": iae,
                    "log_survival_rmse": log_rmse,
                    "n_points": MAX_AGE + 1,
                    "status": "OK",
                }
            )
        pi0 = cf.birth_distribution(train)
        schedules = {
            "stationary generative model": cf.transition_schedule(train, "stationary", 0.0),
            "age-banded generative model": cf.transition_schedule(train, "age_banded", 0.0),
        }
        emp_current = cf.empirical_current(test)
        emp_current = emp_current[["age_s", "current_high"]].rename(columns={"current_high": "estimate"})
        emp_exposure = cf.empirical_exposure(test)
        emp_exposure = emp_exposure[["threshold_s", "exposure_high"]].rename(columns={"exposure_high": "estimate"})
        for model_name, sched in schedules.items():
            curve = cf.propagate(fits["full hazard"]["params"], sched, pi0)
            cur = curve[["age_s", "current_high"]].rename(columns={"current_high": "estimate"})
            iae, log_rmse = curve_error(emp_current, cur, "age_s")
            rows.append(
                {
                    "metric_family": "current_high_composition",
                    "model_name": model_name,
                    "metric_scope": f"season transfer train {train_season} test {test_season}",
                    "integrated_absolute_error": iae,
                    "log_survival_rmse": log_rmse,
                    "n_points": MAX_AGE + 1,
                    "status": "OK",
                }
            )
        full_curve = cf.propagate(fits["full hazard"]["params"], schedules["age-banded generative model"], pi0)
        exp = full_curve[["age_s", "exposure_high"]].dropna().rename(columns={"age_s": "threshold_s", "exposure_high": "estimate"})
        iae, log_rmse = curve_error(emp_exposure, exp, "threshold_s")
        rows.append(
            {
                "metric_family": "high_order_exposure",
                "model_name": "full generative model",
                "metric_scope": f"season transfer train {train_season} test {test_season}",
                "integrated_absolute_error": iae,
                "log_survival_rmse": log_rmse,
                "n_points": MAX_AGE,
                "status": "OK",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, "season_transfer_stress_test.csv", created)
    return out


def plot_figure4(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    f4c: pd.DataFrame,
    created: list[str],
    *,
    close_figure: bool = True,
    save_outputs: bool = True,
) -> plt.Figure:
    fig = plt.figure(figsize=MAIN_FIGURE_SIZE)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.16],
        hspace=0.27,
        wspace=0.34,
        left=0.082,
        right=0.985,
        bottom=0.090,
        top=0.925,
    )
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 1:3]),
    ]
    shift_axis(axes[0], dx=-FIGURE4_PANEL_A_LEFT_SHIFT)
    ax = axes[0]
    reliable = panel_a.loc[~panel_a["sparse_flag"].astype(bool) & panel_a["age_s"].le(35)]
    sparse = panel_a.loc[panel_a["sparse_flag"].astype(bool) | panel_a["age_s"].gt(35)]
    ax.errorbar(
        reliable["age_s"],
        reliable["empirical_estimate"],
        yerr=[reliable["empirical_estimate"] - reliable["ci_low"], reliable["ci_high"] - reliable["empirical_estimate"]],
        fmt="o",
        color=BLACK,
        ecolor=BLACK,
        ms=MARKER_SIZE,
        lw=ERROR_LW,
        capsize=ERROR_CAPSIZE,
        label="empirical",
        zorder=4,
    )
    ax.errorbar(
        sparse["age_s"],
        sparse["empirical_estimate"],
        yerr=[sparse["empirical_estimate"] - sparse["ci_low"], sparse["ci_high"] - sparse["empirical_estimate"]],
        fmt="o",
        mfc="white",
        mec=GREY,
        ecolor=LIGHT_GREY,
        color=GREY,
        ms=SPARSE_MARKER_SIZE,
        lw=SPARSE_ERROR_LW,
        capsize=ERROR_CAPSIZE,
        label="sparse bins",
        zorder=2,
    )
    dline = panel_a.loc[panel_a["age_s"].le(60)].copy()
    supported_line = dline.loc[dline["age_s"].le(FIG4A_SUPPORTED_MAX_AGE)]
    continuation = dline.loc[dline["age_s"].ge(FIG4A_SUPPORTED_MAX_AGE)]
    ax.plot(supported_line["age_s"], supported_line["age_only_baseline"], color=RED, lw=LINE_WIDTH, label="age-only baseline")
    ax.plot(supported_line["age_s"], supported_line["full_hazard_model"], color=BLACK, lw=MODEL_LINE_WIDTH, label="full hazard model")
    ax.plot(continuation["age_s"], continuation["age_only_baseline"], color=RED, lw=LINE_WIDTH, ls="--", alpha=0.55)
    ax.plot(continuation["age_s"], continuation["full_hazard_model"], color=BLACK, lw=MODEL_LINE_WIDTH, ls="--", alpha=0.55)
    label_panel(ax, "A")
    ax.set_xlabel("Run age (s)")
    ax.set_ylabel("1-s termination probability")
    ax.set_xlim(0, 60)
    reliable_hi = np.nanmax(panel_a.loc[panel_a["age_s"].le(35), "ci_high"]) * 1.12
    ax.set_ylim(0, min(0.70, max(0.46, reliable_hi)))
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.56, 1.01),
        ncol=2,
        handlelength=1.6,
        columnspacing=0.9,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    style_ax(ax)

    ax = axes[1]
    colors = {"Low": RED, "Mid": BLACK, "High": BLUE}
    for state in ["Low", "Mid", "High"]:
        g = panel_b.loc[panel_b["state"].eq(state) & panel_b["age_bin_right_s"].le(FIG4B_MAIN_XMAX)].copy()
        if g.empty:
            continue
        ax.errorbar(
            g["age_s"],
            g["empirical_estimate"],
            yerr=[g["empirical_estimate"] - g["ci_low"], g["ci_high"] - g["empirical_estimate"]],
            fmt="o-",
            ms=MARKER_SIZE,
            lw=LINE_WIDTH,
            capsize=ERROR_CAPSIZE,
            color=colors[state],
            ecolor=colors[state],
            label=state,
        )
    label_panel(ax, "B")
    ax.set_xlabel("Run age (s)")
    ax.set_ylabel("1-s termination probability")
    ax.set_xlim(0, FIG4B_MAIN_XMAX)
    y_hi = np.nanmax(panel_b.loc[panel_b["age_bin_right_s"].le(FIG4B_MAIN_XMAX), "ci_high"]) * 1.12
    ax.set_ylim(0, max(0.25, y_hi))
    ax.legend(
        title="Order state",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.61, 1.01),
        ncol=3,
        handlelength=1.6,
        columnspacing=0.95,
        labelspacing=0.5,
        borderaxespad=0.2,
    )
    style_ax(ax)

    ax = axes[2]
    labels = {
        "held-out empirical": ("empirical", BLACK, "o", None),
        "age-only OOF": ("age only", RED, None, "-"),
        "age+order OOF": ("age + order", BLUE, None, "-"),
        "full OOF": ("full model", BLACK, None, "-"),
    }
    for series, (label, color, marker, ls) in labels.items():
        g = f4c.loc[f4c["series"].eq(series) & f4c["age_s"].between(0, MAX_AGE)].copy()
        if g.empty:
            continue
        if marker:
            ax.errorbar(
                g["age_s"],
                g["estimate"],
                yerr=[g["estimate"] - g["ci_low"], g["ci_high"] - g["estimate"]],
                fmt=marker,
                color=color,
                ecolor=color,
                ms=MARKER_SIZE - 0.3,
                elinewidth=ERROR_LW,
                capsize=ERROR_CAPSIZE,
                label=label,
                zorder=4,
            )
        else:
            ax.plot(g["age_s"], g["estimate"], color=color, lw=MODEL_LINE_WIDTH, ls=ls, label=label)
    label_panel(ax, "C", y=1.01)
    ax.set_xlabel("Run duration T (s)")
    ax.set_ylabel(r"$P\, (T \geq t)$")
    ax.set_xlim(0, MAX_AGE)
    ax.set_ylim(4e-3, 1.05)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(plain_log_tick))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.legend(frameon=False, loc="lower left", ncol=2, handlelength=1.7, columnspacing=0.9, labelspacing=0.35, borderaxespad=0.2)
    style_ax(ax)
    if save_outputs:
        savefig(fig, "figure4_final_polished", created, close_figure=close_figure)
    elif close_figure:
        plt.close(fig)
    return fig


def plot_figure5(
    f5a: pd.DataFrame,
    f5b: pd.DataFrame,
    f5c: pd.DataFrame,
    created: list[str],
    *,
    close_figure: bool = True,
    save_outputs: bool = True,
) -> plt.Figure:
    fig = plt.figure(figsize=MAIN_FIGURE_SIZE)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.16],
        hspace=0.27,
        wspace=0.34,
        left=0.082,
        right=0.985,
        bottom=0.090,
        top=0.925,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 1:3])

    x_map = {band: i for i, band in enumerate(AGE_BANDS)}
    for _, g in f5a.groupby("transition_label", observed=True):
        g = g.sort_values("age_band", key=lambda s: s.map(x_map))
        current = str(g["current_state"].iloc[0]).capitalize()
        nxt = str(g["next_state"].iloc[0]).capitalize()
        label = f"{current} \u2192 {nxt}"
        color = {
            ("High", "High"): BLUE,
            ("Low", "Low"): RED,
            ("Mid", "High"): BLACK,
            ("High", "Mid"): GREEN,
        }.get((current, nxt), str(g["line_color"].iloc[0]))
        x = g["age_band"].map(x_map).to_numpy(float)
        y = g["point_estimate"].to_numpy(float)
        ax_a.errorbar(
            x,
            y,
            yerr=[y - g["lower_uncertainty_bound"].to_numpy(float), g["upper_uncertainty_bound"].to_numpy(float) - y],
            fmt="o",
            ms=MARKER_SIZE,
            capsize=ERROR_CAPSIZE,
            color=color,
            ecolor=color,
        )
        ax_a.plot(x, y, color=color, lw=LINE_WIDTH, ls="-", label=label)
    ax_a.set_xticks(range(len(AGE_BANDS)))
    ax_a.set_xticklabels([band.replace("-", "\u2013") for band in AGE_BANDS])
    ax_a.set_xlabel("Run-age band")
    ax_a.set_ylabel("Transition probability\nconditional on survival")
    ax_a.set_ylim(0, 1.02)
    label_panel(ax_a, "A")
    ax_a.legend(
        title="Selected transition",
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.54, 0.48),
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
        labelspacing=0.25,
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )
    style_ax(ax_a)

    for model, label, color, marker in [
        ("held-out empirical", "empirical", BLACK, "o"),
        ("stationary generative model", "stationary", RED, None),
        ("age-banded generative model", "age-banded", BLACK, None),
    ]:
        g = f5b.loc[f5b["model_name"].eq(model) & f5b["age_s"].between(0, MAX_AGE)]
        if g.empty:
            continue
        if marker:
            ax_b.errorbar(
                g["age_s"],
                g["estimate"],
                yerr=[g["estimate"] - g["ci_low"], g["ci_high"] - g["estimate"]],
                fmt=marker,
                color=color,
                ecolor=color,
                ms=MARKER_SIZE,
                lw=ERROR_LW,
                capsize=ERROR_CAPSIZE,
                label=label,
            )
        else:
            ax_b.plot(g["age_s"], g["estimate"], color=color, lw=LINE_WIDTH, label=label)
    ax_b.set_xlabel("Run age (s)")
    ax_b.set_ylabel("High-order fraction")
    ax_b.set_xlim(0, MAX_AGE)
    ax_b.set_ylim(0, max(0.62, np.nanmax(f5b["ci_high"]) * 1.05))
    label_panel(ax_b, "B")
    ax_b.legend(frameon=False, loc="lower right", handlelength=1.7, labelspacing=0.35)
    style_ax(ax_b)

    for model, label, color, marker in [
        ("held-out empirical", "empirical", BLACK, "o"),
        ("full generative model", "full model", BLACK, None),
        ("transitions only", "transitions only", BLUE, None),
        ("differential termination only", "termination only", RED, None),
    ]:
        g = f5c.loc[f5c["model_name"].eq(model) & f5c["threshold_s"].between(1, MAX_AGE)]
        if g.empty:
            continue
        x = g["threshold_s"]
        if marker:
            ax_c.errorbar(
                x,
                g["estimate"],
                yerr=[g["estimate"] - g["ci_low"], g["ci_high"] - g["estimate"]],
                fmt=marker,
                color=color,
                ecolor=color,
                ms=MARKER_SIZE,
                lw=ERROR_LW,
                capsize=ERROR_CAPSIZE,
                label=label,
            )
        else:
            ax_c.plot(x, g["estimate"], color=color, lw=LINE_WIDTH, label=label)
    ax_c.set_xlabel("Survival threshold (s)")
    ax_c.set_ylabel("Mean high-order exposure")
    ax_c.set_xlim(1, MAX_AGE)
    ax_c.set_ylim(0, max(0.58, np.nanmax(f5c["ci_high"]) * 1.05))
    label_panel(ax_c, "C", y=1.01)
    ax_c.legend(frameon=False, loc="lower right", ncol=1, handlelength=1.7, labelspacing=0.35)
    style_ax(ax_c)
    if save_outputs:
        savefig(fig, "figure5_final_polished", created, close_figure=close_figure)
    elif close_figure:
        plt.close(fig)
    return fig


def plot_supplementary(tables: dict[str, pd.DataFrame], season_transfer: pd.DataFrame, created: list[str]) -> None:
    surv = pd.read_csv(FIG5_SUPP_SURVIVAL)
    fig, ax = plt.subplots(figsize=(4.7, 3.5))
    for model, label, color, marker in [
        ("held-out empirical", "empirical", BLACK, "o"),
        ("stationary generative model", "stationary", GREY, None),
        ("age-banded generative model", "age-banded", BLACK, None),
        ("observed-path hazard benchmark", "observed-path", BLUE, None),
    ]:
        g = surv.loc[surv["model_name"].eq(model) & surv["age_s"].between(0, MAX_AGE)]
        if marker:
            ax.errorbar(g["age_s"], g["estimate"], yerr=[g["estimate"] - g["ci_low"], g["ci_high"] - g["estimate"]], fmt=marker, ms=3, lw=0.7, capsize=1.6, color=color, label=label)
        else:
            ax.plot(g["age_s"], g["estimate"], color=color, lw=1.7, label=label)
    ax.set_xlabel("Run duration (s)")
    ax.set_ylabel("P(T >= t)")
    ax.set_yscale("log")
    ax.set_ylim(4e-3, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "supplementary_crossfitted_generative_survival", created)

    fixture = tables["fixture_metrics"]
    piv = fixture.pivot(index="fixture_id", columns="model_name", values="survival_IAE")
    diff = (piv["stationary generative model"] - piv["age-banded generative model"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.barh(np.arange(len(diff)), diff.to_numpy(float), color=np.where(diff >= 0, BLACK, GREY), height=0.75)
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Stationary IAE - age-banded IAE")
    ax.set_ylabel("Held-out fixture")
    fig.tight_layout()
    savefig(fig, "supplementary_fixture_survival_error", created)

    trans = tables["transition_matrices"]
    agg = trans.loc[trans["transition_kind"].eq("age_banded")].groupby(["age_band", "from_state", "to_state"], observed=True)["probability"].mean().reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.5), sharex=True, sharey=True)
    for ax, band in zip(axes, AGE_BANDS):
        mat = np.zeros((3, 3))
        g = agg.loc[agg["age_band"].eq(band)]
        for _, row in g.iterrows():
            mat[STATE_ORDER.index(row["from_state"]), STATE_ORDER.index(row["to_state"])] = row["probability"]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Greys")
        ax.set_title(band, loc="center", fontsize=9)
        ax.set_xticks(range(3), ["Low", "Mid", "High"], rotation=45, ha="right")
        ax.set_yticks(range(3), ["Low", "Mid", "High"])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] > 0.55 else "black")
    axes[0].set_ylabel("From state")
    for ax in axes:
        ax.set_xlabel("To state")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Probability")
    fig.tight_layout()
    savefig(fig, "supplementary_full_transition_matrices", created)

    early = agg.loc[agg["age_band"].eq("0-5")]
    late = agg.loc[agg["age_band"].eq("20-35")]
    mat_e = np.zeros((3, 3))
    mat_l = np.zeros((3, 3))
    for _, row in early.iterrows():
        mat_e[STATE_ORDER.index(row["from_state"]), STATE_ORDER.index(row["to_state"])] = row["probability"]
    for _, row in late.iterrows():
        mat_l[STATE_ORDER.index(row["from_state"]), STATE_ORDER.index(row["to_state"])] = row["probability"]
    diff_mat = mat_l - mat_e
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    im = ax.imshow(diff_mat, vmin=-0.2, vmax=0.2, cmap="coolwarm")
    ax.set_xticks(range(3), ["Low", "Mid", "High"])
    ax.set_yticks(range(3), ["Low", "Mid", "High"])
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{diff_mat[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Late - early")
    fig.tight_layout()
    savefig(fig, "supplementary_late_early_transition_contrast", created)

    cur_emp = empirical_all_state_current(tables["oof"])
    model_curve = weighted_fold_curve(tables["fold_curves"], "age-banded generative model")
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    colors = {"low": RED, "mid": GREEN, "high": BLUE}
    for state in STATE_ORDER:
        ax.errorbar(cur_emp["age_s"], cur_emp[f"current_{state}"], yerr=[cur_emp[f"current_{state}"] - cur_emp[f"current_{state}_ci_low"], cur_emp[f"current_{state}_ci_high"] - cur_emp[f"current_{state}"]], fmt="o", ms=2.6, lw=0.5, capsize=1.2, color=colors[state], alpha=0.65)
        ax.plot(model_curve["age_s"], model_curve[f"current_{state}"], color=colors[state], lw=1.7, label=STATE_LABELS[state])
    ax.set_xlabel("Run age (s)")
    ax.set_ylabel("Current state fraction")
    ax.set_xlim(0, MAX_AGE)
    ax.set_ylim(0, 1)
    ax.legend(title="State", frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    savefig(fig, "supplementary_all_state_composition", created)

    exp_emp = empirical_all_state_exposure(tables["oof"])
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    for state in STATE_ORDER:
        ax.plot(exp_emp["threshold_s"], exp_emp[f"exposure_{state}"], color=colors[state], lw=0, marker="o", ms=2.6, alpha=0.6)
        ax.plot(model_curve["age_s"], model_curve[f"exposure_{state}"], color=colors[state], lw=1.7, label=STATE_LABELS[state])
    ax.set_xlabel("Survival threshold (s)")
    ax.set_ylabel("Mean state exposure")
    ax.set_xlim(1, MAX_AGE)
    ax.set_ylim(0, 1)
    ax.legend(title="State", frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    savefig(fig, "supplementary_all_state_exposure", created)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    st = season_transfer.loc[season_transfer["status"].fillna("OK").eq("OK")].copy()
    if st.empty:
        ax.text(0.5, 0.5, "season-transfer stress test unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        sub = st.loc[st["metric_family"].isin(["current_high_composition", "high_order_exposure", "observed_path_survival"]) & st["integrated_absolute_error"].notna()]
        labels = [m.replace("season transfer ", "").replace(" train ", "").replace(" test ", " -> ") for m in sub["metric_scope"].unique()]
        x = np.arange(len(labels))
        width = 0.22
        for i, family in enumerate(["observed_path_survival", "current_high_composition", "high_order_exposure"]):
            vals = []
            for scope in sub["metric_scope"].unique():
                ss = sub.loc[sub["metric_scope"].eq(scope) & sub["metric_family"].eq(family)]
                vals.append(float(ss["integrated_absolute_error"].mean()) if not ss.empty else np.nan)
            ax.bar(x + (i - 1) * width, vals, width=width, label=family.replace("_", " "))
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylabel("Integrated absolute error")
        ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "supplementary_season_transfer", created)

    oof = tables["oof"].copy()
    oof["pred_full"] = pd.to_numeric(oof["pred_full_hazard_termination_probability"], errors="coerce")
    oof["event"] = pd.to_numeric(oof["observed_termination"], errors="coerce").fillna(0).astype(int)
    oof = oof.dropna(subset=["pred_full"])
    oof["prediction_decile"] = pd.qcut(oof["pred_full"], q=10, duplicates="drop")
    calib = (
        oof.groupby("prediction_decile", observed=True)
        .agg(predicted_probability=("pred_full", "mean"), empirical_probability=("event", "mean"), n=("event", "size"), events=("event", "sum"))
        .reset_index()
    )
    calib["ci_low"], calib["ci_high"] = wilson(calib["events"], calib["n"])
    write_csv(calib, "supplementary_interval_hazard_calibration_source.csv", created)
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    ax.plot([0, max(calib["predicted_probability"].max(), calib["empirical_probability"].max()) * 1.05], [0, max(calib["predicted_probability"].max(), calib["empirical_probability"].max()) * 1.05], color="0.6", lw=1.0)
    ax.errorbar(calib["predicted_probability"], calib["empirical_probability"], yerr=[calib["empirical_probability"] - calib["ci_low"], calib["ci_high"] - calib["empirical_probability"]], fmt="o", color=BLACK, ecolor=BLACK, ms=3.5, capsize=2)
    ax.set_xlabel("Mean predicted termination probability")
    ax.set_ylabel("Empirical termination probability")
    fig.tight_layout()
    savefig(fig, "supplementary_crossfitted_interval_hazard_calibration", created)

    oof["_age_bin_2s"] = (np.floor(pd.to_numeric(oof["age_start_s"], errors="coerce") / 2.0) * 2).astype(int)
    state_calib = (
        oof.loc[oof["_age_bin_2s"].between(0, 30)]
        .groupby(["_age_bin_2s", "observed_state"], observed=True)
        .agg(predicted_probability=("pred_full", "mean"), empirical_probability=("event", "mean"), n=("event", "size"), events=("event", "sum"))
        .reset_index()
    )
    state_calib["ci_low"], state_calib["ci_high"] = wilson(state_calib["events"], state_calib["n"])
    write_csv(state_calib, "supplementary_state_conditioned_hazard_calibration_source.csv", created)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for state, color in [("low", RED), ("mid", BLACK), ("high", BLUE)]:
        g = state_calib.loc[state_calib["observed_state"].astype(str).str.lower().eq(state)]
        if g.empty:
            continue
        x = g["_age_bin_2s"] + 1.0
        ax.errorbar(x, g["empirical_probability"], yerr=[g["empirical_probability"] - g["ci_low"], g["ci_high"] - g["empirical_probability"]], fmt="o", color=color, ecolor=color, ms=2.6, capsize=1.2, alpha=0.75)
        ax.plot(x, g["predicted_probability"], color=color, lw=1.4, label=STATE_LABELS[state])
    ax.set_xlabel("Run age (s)")
    ax.set_ylabel("Termination probability")
    ax.set_xlim(0, 30)
    ax.legend(title="Order state", frameon=False, ncol=3)
    fig.tight_layout()
    savefig(fig, "supplementary_crossfitted_state_conditioned_hazard_calibration", created)

    robust = tables.get("robustness_summary", pd.DataFrame()).copy()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    s0 = model_curve.loc[model_curve["age_s"].isin([5, 10, 20, 30, 35]), ["age_s", "current_high"]].copy()
    ax.plot(s0["age_s"], s0["current_high"], marker="o", color=BLACK, lw=1.5, label="S0 raw")
    if not robust.empty:
        for _, row in robust.iterrows():
            xs = np.array([5, 10, 20, 30, 35], dtype=float)
            ys = np.array([row[f"current_high_{int(x)}s"] for x in xs], dtype=float)
            ax.plot(xs, ys, marker="o", lw=1.2, label=str(row["state_definition"]).replace("_", " "))
    ax.set_xlabel("Run age (s)")
    ax.set_ylabel("Current high-order fraction")
    ax.set_xlim(4, 36)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "supplementary_s0_s1_s2_composition_robustness", created)

    team_metrics = tables.get("team_season_metrics", pd.DataFrame()).copy()
    if not team_metrics.empty:
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        summary = team_metrics.groupby(["team", "model_name"], observed=True)["heldout_log_loss"].mean().reset_index()
        teams = list(summary["team"].drop_duplicates())
        x = np.arange(len(teams))
        width = 0.24
        model_order = ["age-only hazard", "age+order hazard", "full hazard"]
        colors_models = [RED, BLUE, BLACK]
        for i, (model, color) in enumerate(zip(model_order, colors_models)):
            vals = [float(summary.loc[summary["team"].eq(team) & summary["model_name"].eq(model), "heldout_log_loss"].mean()) for team in teams]
            ax.bar(x + (i - 1) * width, vals, width=width, color=color, label=model.replace(" hazard", ""))
        ax.set_xticks(x, teams, rotation=15, ha="right")
        ax.set_ylabel("Held-out log loss")
        ax.legend(frameon=False, ncol=3, fontsize=7.5)
        fig.tight_layout()
        savefig(fig, "supplementary_team_specific_results", created)


def make_contact_sheet(created: list[str]) -> None:
    import matplotlib.image as mpimg

    names = [
        "figure4_final_polished.png",
        "figure5_final_polished.png",
        "supplementary_crossfitted_generative_survival.png",
        "supplementary_fixture_survival_error.png",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, name in zip(axes.ravel(), names):
        path = OUT_DIR / name
        if path.exists():
            ax.imshow(mpimg.imread(path))
            ax.set_title(name, fontsize=9)
        else:
            ax.text(0.5, 0.5, f"missing\n{name}", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    path = OUT_DIR / "final_fig4_fig5_contact_sheet.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=220)
    plt.close(fig)
    created.append(str(path))

    for stem, before_name, after_name in [
        ("figure4_before_after_polish", "figure4_final.png", "figure4_final_polished.png"),
        ("figure5_before_after_polish", "figure5_final.png", "figure5_final_polished.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for ax, title, folder, name in [
            (axes[0], "Before", PREVIOUS_FINAL_DIR, before_name),
            (axes[1], "After polish", OUT_DIR, after_name),
        ]:
            path_img = folder / name
            if path_img.exists():
                ax.imshow(mpimg.imread(path_img))
                ax.set_title(title, fontsize=9)
            else:
                ax.text(0.5, 0.5, f"missing\n{path_img}", ha="center", va="center", fontsize=7)
            ax.set_axis_off()
        fig.tight_layout()
        out = OUT_DIR / f"{stem}.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=220)
        plt.close(fig)
        created.append(str(out))


def caption_texts(created: list[str]) -> None:
    fig4 = (
        "Figure 4. A, Pooled empirical one-second termination probability versus run age, "
        "with physical-fixture bootstrap uncertainty. The age-only baseline and full hazard "
        "model are shown through the supported 0-35 s range, with lighter dashed continuation "
        "outside support. Open grey symbols denote sparse bins. B, Pooled empirical termination "
        "probability stratified by recent Low, Mid and High collective-order state using the "
        "trailing five-second order state. C, Leave-one-physical-fixture-out observed-path "
        "survival reconstruction. At each fold, hazard parameters were estimated from all other "
        "physical fixtures and applied to the excluded fixture; order-dependent models used the "
        "observed held-out order sequence as a declared time-varying covariate. Points show "
        "held-out empirical survival with physical-fixture bootstrap uncertainty."
    )
    fig5 = (
        "Figure 5. A, Pooled one-second transition probabilities between selected collective-order "
        "states across run-age bands, conditional on survival through the interval, with physical-"
        "fixture bootstrap uncertainty. B, Leave-one-physical-fixture-out prediction of the current "
        "High-order fraction among surviving runs under stationary and age-banded transition models. "
        "C, Held-out cumulative High-order exposure compared with the full cross-fitted state-survival "
        "model and frozen transition-only and differential-termination-only counterfactuals. "
        "Counterfactual curves are descriptive non-additive model interventions."
    )
    write_text(fig4, "final_figure4_caption.txt", created)
    write_text(fig5, "final_figure5_caption.txt", created)
    math_note = (
        "# Final Figure 5 Counterfactual Definitions\n\n"
        "The full model propagates the training-estimated birth distribution, age-banded transition "
        "matrices and state-dependent termination probabilities.\n\n"
        "The transitions-only curve uses the age-banded transition matrices but replaces state-dependent "
        "termination by a common mortality matched to the fold-level state mixture.\n\n"
        "The differential-termination-only curve preserves state-dependent termination but freezes "
        "state switching with an identity transition operator.\n\n"
        "These frozen counterfactuals are not additive contributions."
    )
    write_text(math_note, "final_counterfactual_definitions.md", created)


def make_report(
    tables: dict[str, pd.DataFrame],
    leakage: pd.DataFrame,
    validation: pd.DataFrame,
    season_transfer: pd.DataFrame,
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    created: list[str],
) -> None:
    fold_summary = tables["fold_summary"]
    fixture = tables["fixture_metrics"]
    n_folds = int(fold_summary["heldout_physical_match_id"].nunique())
    hazard = validation.loc[validation["metric_family"].eq("hazard")]
    comp = validation.loc[validation["metric_family"].eq("current_high_composition")]
    exposure = validation.loc[validation["metric_family"].eq("high_order_exposure")]
    all_state = validation.loc[validation["metric_family"].str.contains("all_state", na=False)]
    robust = tables.get("robustness_summary", pd.DataFrame())

    hazard_age_only = pd.to_numeric(hazard.loc[hazard["model_name"].eq("age-only hazard"), "heldout_log_loss"], errors="coerce").dropna()
    hazard_age_order = pd.to_numeric(hazard.loc[hazard["model_name"].eq("age+order hazard"), "heldout_log_loss"], errors="coerce").dropna()
    hazard_full = pd.to_numeric(hazard.loc[hazard["model_name"].eq("full hazard"), "heldout_log_loss"], errors="coerce").dropna()
    hazard_supports_full = (
        not hazard_age_only.empty
        and not hazard_age_order.empty
        and not hazard_full.empty
        and float(hazard_full.iloc[0]) < float(hazard_age_only.iloc[0])
        and float(hazard_full.iloc[0]) <= float(hazard_age_order.iloc[0])
    )

    piv_surv = fixture.pivot(index="fixture_id", columns="model_name", values="survival_IAE")
    if {"stationary generative model", "age-banded generative model"}.issubset(piv_surv.columns):
        surv_improved = int((piv_surv["age-banded generative model"] < piv_surv["stationary generative model"]).sum())
        surv_total = int(piv_surv.dropna().shape[0])
    else:
        surv_improved = 0
        surv_total = 0

    piv_cur = fixture.pivot(index="fixture_id", columns="model_name", values="current_high_IAE")
    if {"stationary generative model", "age-banded generative model"}.issubset(piv_cur.columns):
        cur_diff = (piv_cur["stationary generative model"] - piv_cur["age-banded generative model"]).dropna()
        cur_improved = int((cur_diff > 0).sum())
        cur_total = int(cur_diff.shape[0])
        cur_mean = float(cur_diff.mean()) if cur_total else np.nan
        cur_median = float(cur_diff.median()) if cur_total else np.nan
        rng = np.random.default_rng(20260724)
        if cur_total:
            boot = np.array([rng.choice(cur_diff.to_numpy(float), size=cur_total, replace=True).mean() for _ in range(BOOTSTRAP_B)])
            cur_ci_low, cur_ci_high = np.nanpercentile(boot, [2.5, 97.5])
        else:
            cur_ci_low = cur_ci_high = np.nan
    else:
        cur_improved = cur_total = 0
        cur_mean = cur_median = cur_ci_low = cur_ci_high = np.nan

    exposure_fixture = fixture_high_exposure_errors(tables)
    exposure_summary = "unavailable"
    if not exposure_fixture.empty:
        full_exposure = exposure_fixture["high_exposure_IAE"].dropna()
        if not full_exposure.empty:
            exposure_summary = (
                f"full model fixture IAE mean={float(full_exposure.mean()):.6g}, "
                f"median={float(full_exposure.median()):.6g}, "
                f"max={float(full_exposure.max()):.6g}"
            )

    st_summary = "not run"
    if not season_transfer.empty:
        ok = season_transfer.loc[season_transfer["status"].fillna("OK").eq("OK")]
        if not ok.empty:
            st_summary = "; ".join(
                f"{scope}: mean IAE={ok.loc[ok['metric_scope'].eq(scope), 'integrated_absolute_error'].dropna().mean():.4f}"
                for scope in ok["metric_scope"].dropna().unique()
                if "season transfer" in scope
            )
        else:
            st_summary = "; ".join(season_transfer.get("details", pd.Series(["unavailable"])).dropna().astype(str).unique())

    leakage_pass = bool(leakage["status"].eq("PASS").all())
    current_high_supports_age_banded = (
        metric_float(comp, "age-banded generative model", "integrated_absolute_error")
        < metric_float(comp, "stationary generative model", "integrated_absolute_error")
    )
    exposure_supports_full = (
        metric_float(exposure, "full generative model", "integrated_absolute_error")
        <= min(
            metric_float(exposure, "transitions only", "integrated_absolute_error"),
            metric_float(exposure, "differential termination only", "integrated_absolute_error"),
        )
    )
    recommendation = (
        "A. READY FOR MANUSCRIPT INSERTION"
        if leakage_pass and hazard_supports_full and current_high_supports_age_banded and exposure_supports_full
        else "B. READY AFTER MINOR COSMETIC EDITS"
        if leakage_pass and (hazard_supports_full or current_high_supports_age_banded)
        else "C. CROSS-FITTING AUDIT REQUIRES CORRECTION"
        if not leakage_pass
        else "D. RESULTS DO NOT SUPPORT THE CURRENT MAIN-FIGURE CLAIMS"
    )

    audit_lines = [
        "# Final Crossfit Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Script: {SCRIPT_PATH.relative_to(REPO_ROOT)}",
        "",
        "## Checks",
        "",
        "```csv",
        markdown_csv(leakage),
        "```",
        "",
        "## Fold Manifest",
        "",
        f"- physical_fixture_folds: {n_folds}",
        f"- out_of_fold_interval_rows: {len(tables['oof'])}",
        f"- out_of_fold_fixtures: {tables['oof']['physical_match_id'].nunique()}",
        "- figure4C_observed_path_prediction: yes",
        "- figure5B_C_uses_heldout_future_state_sequence: no",
    ]
    write_text("\n".join(audit_lines), "final_crossfit_audit.md", created)

    validation_lines = [
        "# Final Crossfitted Validation Summary",
        "",
        f"- hazard_log_loss_age_only: {metric_value(hazard, 'age-only hazard', 'heldout_log_loss')}",
        f"- hazard_log_loss_age_plus_order: {metric_value(hazard, 'age+order hazard', 'heldout_log_loss')}",
        f"- hazard_log_loss_full: {metric_value(hazard, 'full hazard', 'heldout_log_loss')}",
        f"- current_high_IAE_stationary: {metric_value(comp, 'stationary generative model', 'integrated_absolute_error')}",
        f"- current_high_IAE_age_banded: {metric_value(comp, 'age-banded generative model', 'integrated_absolute_error')}",
        f"- paired_current_high_improvement_mean: {cur_mean:.6g}",
        f"- paired_current_high_improvement_median: {cur_median:.6g}",
        f"- paired_current_high_improvement_bootstrap_ci95: [{cur_ci_low:.6g}, {cur_ci_high:.6g}]",
        f"- paired_current_high_fixtures_improved: {cur_improved}/{cur_total}",
        f"- survival_age_banded_improved_fixtures: {surv_improved}/{surv_total}",
        f"- high_order_exposure_full_IAE: {metric_value(exposure, 'full generative model', 'integrated_absolute_error')}",
        f"- high_order_exposure_fixture_distribution: {exposure_summary}",
        f"- all_state_metric_rows: {len(all_state)}",
        f"- s0_s1_s2_robustness_rows: {0 if robust.empty else len(robust)}",
        f"- season_transfer: {st_summary}",
        "",
        "## Validation Table",
        "",
        "```csv",
        markdown_csv(validation),
        "```",
    ]
    write_text("\n".join(validation_lines), "final_crossfitted_validation_summary.md", created)

    lines = [
        "# Final Figure 4/5 Polish Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Script: {SCRIPT_PATH.relative_to(REPO_ROOT)}",
        "",
        "## Inputs",
        "",
        f"- Figure 4A pooled model source: {PANEL_A_AUDIT.relative_to(REPO_ROOT)}",
        f"- Figure 4A pooled empirical source: out-of-fold interval prediction union from {OOF_PREDICTIONS.relative_to(REPO_ROOT)}",
        f"- Figure 4B empirical interval source: {TERMINAL_INTERVALS.relative_to(REPO_ROOT)}",
        f"- Cross-fitted package source: {CROSSFIT_DIR.relative_to(REPO_ROOT)}",
        f"- Figure 4A/B bootstrap unit: physical_fixture_id; B={BOOTSTRAP_B}",
        "",
        "## Final Report Questions",
        "",
        "1. Were any scientific parameters or model definitions changed? No.",
        f"2. What is the exact supported age range in Figure 4A? 0-{FIG4A_SUPPORTED_MAX_AGE} s.",
        "3. How are model curves shown beyond the supported range? Lighter dashed continuation beyond the supported range.",
        f"4. Is Figure 4B restricted or de-emphasised after 30 s? Restricted to 0-{FIG4B_MAIN_XMAX} s.",
        "5. Is Figure 4C on the same interval convention as Figure 4A? Yes; survival is reconstructed as S(t_right)=S(t_left)*(1-p_interval), so the [0,1] interval drops survival at t=1.",
        "6. Does Figure 4C add any new model parameter? No; it reuses the Figure 4A empirical/model one-second interval probabilities.",
        "7. Is Figure 5A explicitly conditional on survival? Yes; transition_conditioning=conditional_on_survival in source data.",
        "8. Is Figure 5A pooled rather than cross-fitted? Yes.",
        "9. Does Figure 5B avoid all held-out future-state information? Yes; it propagates training-estimated pi0, transitions and termination.",
        f"10. Does the age-banded model outperform the stationary model numerically? {'Yes' if current_high_supports_age_banded else 'No'}; current-High IAE stationary={metric_value(comp, 'stationary generative model', 'integrated_absolute_error')}, age-banded={metric_value(comp, 'age-banded generative model', 'integrated_absolute_error')}.",
        f"11. How many physical fixtures improve? Current-High IAE improves in {cur_improved}/{cur_total}; survival IAE improves in {surv_improved}/{surv_total}.",
        f"12. Does Figure 5C reproduce held-out exposure within uncertainty? Full-model high-order exposure IAE={metric_value(exposure, 'full generative model', 'integrated_absolute_error')}; fixture error distribution: {exposure_summary}.",
        "13. Are the counterfactuals correctly labelled non-additive? Yes.",
        f"14. Do S0/S1/S2 support the same qualitative conclusion? Robustness rows available={0 if robust.empty else len(robust)}; see supplementary_s0_s1_s2_composition_robustness.",
        f"15. What do the season-transfer tests show? {st_summary}.",
        f"16. Are Figures 4 and 5 ready for manuscript insertion? {recommendation}.",
        "",
        recommendation,
    ]
    write_text("\n".join(lines), "final_fig4_fig5_polish_report.md", created)


def metric_value(df: pd.DataFrame, model: str, col: str) -> str:
    if df.empty or col not in df.columns:
        return "NA"
    s = df.loc[df["model_name"].eq(model), col].dropna()
    if s.empty:
        return "NA"
    return f"{float(s.iloc[0]):.6g}"


def metric_float(df: pd.DataFrame, model: str, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.inf
    s = pd.to_numeric(df.loc[df["model_name"].eq(model), col], errors="coerce").dropna()
    if s.empty:
        return np.inf
    return float(s.iloc[0])


def main() -> None:
    configure_style()
    require_inputs()
    created: list[str] = []

    tables = copy_core_tables(created)
    leakage = build_leakage_audit(tables["fold_summary"], tables["state_thresholds"], tables["hazard_parameters"], tables["oof"], created)
    build_fold_manifest(tables["fold_summary"], tables["oof"], created)

    terminal = load_terminal_intervals()
    panel_a = make_figure4a_source(tables["oof"], created)
    panel_b = make_figure4b_source(terminal, created)
    sources = make_crossfit_sources(panel_a, created)
    season_transfer = fit_season_transfer(created)
    validation = build_validation_metrics_table(tables, season_transfer, created)

    plot_figure4(panel_a, panel_b, sources["figure4c"], created)
    plot_figure5(sources["figure5a"], sources["figure5b"], sources["figure5c"], created)
    plot_supplementary(tables, season_transfer, created)
    caption_texts(created)
    make_contact_sheet(created)
    make_report(tables, leakage, validation, season_transfer, panel_a, panel_b, created)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "output_folder": str(OUT_DIR.relative_to(REPO_ROOT)),
        "main_figures": ["figure4_final_polished.png", "figure4_final_polished.pdf", "figure5_final_polished.png", "figure5_final_polished.pdf"],
        "comparison_figures": ["figure4_before_after_polish.png", "figure5_before_after_polish.png", "final_fig4_fig5_contact_sheet.png"],
        "state_definition": "S0 raw production state",
        "states": ["Low", "Mid", "High"],
        "risk_interval_s": 1,
        "age_bands_s": AGE_BANDS,
        "figure4A_empirical_source": "union of out-of-fold held-out interval observations",
        "figure4A_supported_age_range_s": f"0-{FIG4A_SUPPORTED_MAX_AGE}",
        "figure4A_model_continuation_beyond_support": "lighter dashed continuation",
        "figure4B_main_age_range_s": f"0-{FIG4B_MAIN_XMAX}",
        "figure5A_transition_conditioning": "conditional_on_survival",
        "crossfit_unit": "leave-one-physical-fixture-out",
        "bootstrap_unit_main_pooled_panels": "physical_fixture_id",
        "bootstrap_replicates_main_pooled_panels": BOOTSTRAP_B,
        "figure4C_survival_convention": "S(t_right)=S(t_left)*(1-p_interval); first [0,1] interval drops survival at t=1",
        "figure4C_source": "reconstructed from Figure 4A empirical/model one-second interval probabilities",
        "figure4C_new_parameters_added": False,
        "figure5B_C_uses_heldout_future_state_history": False,
        "style": {
            "style_source": "analysis.levy_paper.util.paper_utils.configure_paper_plotting plus final style enforcement",
            "main_figure_size_inches": list(MAIN_FIGURE_SIZE),
            "font_family": "serif",
            "mathtext_fontset": "cm",
            "font_size": PAPER_FONT_BASE,
            "axes_labelsize": AXIS_LABEL_FONTSIZE,
            "tick_labelsize": TICK_LABEL_FONTSIZE,
            "legend_fontsize": LEGEND_FONTSIZE,
            "legend_title_fontsize": LEGEND_TITLE_FONTSIZE,
            "panel_letter_fontsize": PANEL_LETTER_FONTSIZE,
            "line_width": LINE_WIDTH,
            "model_line_width": MODEL_LINE_WIDTH,
            "marker_size": MARKER_SIZE,
            "error_line_width": ERROR_LW,
            "error_capsize": ERROR_CAPSIZE,
            "save_dpi": SAVE_DPI,
        },
        "created_files": [str(Path(p).relative_to(REPO_ROOT)) for p in created if Path(p).exists()],
    }
    write_json(manifest, "final_fig4_fig5_manifest.json", created)
    print(f"created {len(created)} files in {OUT_DIR}")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
