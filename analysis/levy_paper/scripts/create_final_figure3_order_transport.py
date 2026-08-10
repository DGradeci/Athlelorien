from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import matplotlib

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAPER_ROOT = SCRIPT_PATH.parents[1]
MPLCONFIG_DIR = PAPER_ROOT / "outputs" / "final" / "figure3_order_transport" / "_mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from analysis.levy_paper.util.paper_utils import cache_path, configure_paper_plotting, draw_panel_letter


OUT_DIR = PAPER_ROOT / "outputs" / "final" / "figure3_order_transport"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = cache_path("centroid_order_runs_2020_2021_all_teams_pitchfix_sticky_active.parquet")

BLACK = "black"
GREY = "0.55"
LOW_COLOR = "#d62728"
MID_COLOR = "black"
HIGH_COLOR = "blue"
STATE_COLORS = {"Low": LOW_COLOR, "Mid": MID_COLOR, "High": HIGH_COLOR}
STATE_ORDER = ["Low", "Mid", "High"]
CCDF_BAND_ALPHA = 0.24
BOOTSTRAP_B = 500
BOOTSTRAP_SEED = 20260718
PAPER_FONT_BASE = 10
PANEL_LETTER_FONTSIZE = 13
AXIS_LABEL_FONTSIZE = PAPER_FONT_BASE
TICK_LABEL_FONTSIZE = PAPER_FONT_BASE * 0.9
LEGEND_FONTSIZE = PAPER_FONT_BASE * 0.85
LEGEND_TITLE_FONTSIZE = PAPER_FONT_BASE * 0.9


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)


def shift_axes_x(axes: list[plt.Axes], dx: float) -> None:
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])


def add_panel_label(ax: plt.Axes, label: str, y: float = 1.04) -> None:
    draw_panel_letter(ax, label.rstrip("."), x=-0.16, y=y, fontsize=PANEL_LETTER_FONTSIZE)


def add_three_panel_labels(ax_a: plt.Axes, ax_b: plt.Axes, ax_c: plt.Axes) -> None:
    add_panel_label(ax_a, "A.")
    add_panel_label(ax_b, "B.")
    add_panel_label(ax_c, "C.", y=1.08)


def add_horizontal_order_legend(fig: plt.Figure, ax_source: plt.Axes, y: float = 0.965) -> None:
    handles, labels = ax_source.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l in STATE_ORDER]
    if not filtered:
        return
    handles, labels = zip(*filtered)
    fig.legend(
        handles,
        labels,
        title="Order state",
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        columnspacing=1.4,
        handlelength=1.8,
        borderaxespad=0.0,
    )


def plain_log_tick(value: float, _pos: int | None = None) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = int(round(np.log10(value)))
    if not np.isclose(value, 10**exponent, rtol=1e-8, atol=0.0):
        return ""
    if exponent == 0:
        return "1"
    return f"{10**exponent:.{abs(exponent)}f}"


def load_runs() -> pd.DataFrame:
    runs = pd.read_parquet(CACHE_PATH)
    if "track_type" in runs.columns:
        runs = runs.loc[runs["track_type"].astype(str).eq("centroid")].copy()
    if "run_uid" in runs.columns:
        runs = runs.drop_duplicates("run_uid").copy()
    needed = ["p_mean", "v_mean_mps", "duration_s", "run_length_m", "p_early_3s"]
    runs = runs.dropna(subset=needed).copy()
    vmax = runs["v_mean_mps"].quantile(0.995)
    runs = runs.loc[
        runs["p_mean"].between(0, 1)
        & runs["p_early_3s"].between(0, 1)
        & runs["v_mean_mps"].between(0, vmax)
        & (runs["duration_s"] > 0)
        & (runs["run_length_m"] > 0)
    ].copy()
    q1, q2 = runs["p_early_3s"].quantile([1 / 3, 2 / 3]).to_numpy(float)
    runs["early_order_state"] = pd.cut(
        runs["p_early_3s"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=STATE_ORDER,
        include_lowest=True,
    )
    return runs


def add_top_state_arrows(ax: plt.Axes, q1: float, q2: float) -> None:
    for q in (q1, q2):
        ax.axvline(q, color=BLACK, lw=1.0, ls="--", alpha=0.85)
    trans = ax.get_xaxis_transform()
    y_arrow = 1.018
    y_text = 1.052
    spans = [
        ("Low", 0.035, q1 - 0.018),
        ("Mid", q1 + 0.018, q2 - 0.018),
        ("High", q2 + 0.018, 0.965),
    ]
    for label, left, right in spans:
        ax.annotate(
            "",
            xy=(right, y_arrow),
            xytext=(left, y_arrow),
            xycoords=trans,
            textcoords=trans,
            arrowprops={"arrowstyle": "<->", "color": BLACK, "lw": 0.85, "shrinkA": 0, "shrinkB": 0},
            annotation_clip=False,
        )
        ax.text(
            0.5 * (left + right),
            y_text,
            label,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=TICK_LABEL_FONTSIZE,
            color=BLACK,
            clip_on=False,
        )


def ccdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return np.array([(values >= g).mean() for g in grid], dtype=float)


def wilson_ci(k: np.ndarray, n: int, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=float)
    if n <= 0:
        nan = np.full_like(k, np.nan, dtype=float)
        return nan, nan
    p = k / float(n)
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denom
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def best_cluster_cols(runs: pd.DataFrame) -> list[str]:
    candidates = [
        ["season", "match_id", "team"],
        ["match_id", "team"],
        ["season", "match_id"],
        ["match_id"],
        ["source_key", "team"],
        ["source_key"],
    ]
    for cols in candidates:
        if all(col in runs.columns for col in cols):
            if runs[cols].drop_duplicates().shape[0] > 1:
                return cols
    return []


def cluster_bootstrap_ci(
    runs: pd.DataFrame,
    value_col: str,
    grid: np.ndarray,
    cluster_cols: list[str],
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    d = runs.dropna(subset=[value_col] + cluster_cols).copy()
    d = d.loc[np.isfinite(d[value_col].to_numpy(float)) & (d[value_col].to_numpy(float) > 0)]
    if d.empty or not cluster_cols:
        nan = np.full(len(grid), np.nan, dtype=float)
        return nan, nan, {"n_clusters": 0, "B": b, "cluster_cols": ",".join(cluster_cols)}

    grouped = list(d.groupby(cluster_cols, observed=True, sort=False))
    n_clusters = len(grouped)
    n_by_cluster = np.zeros(n_clusters, dtype=float)
    k_by_cluster = np.zeros((n_clusters, len(grid)), dtype=float)
    for i, (_key, sub) in enumerate(grouped):
        values = sub[value_col].to_numpy(float)
        n_by_cluster[i] = len(values)
        k_by_cluster[i, :] = [(values >= threshold).sum() for threshold in grid]

    rng = np.random.default_rng(seed)
    boot = np.empty((b, len(grid)), dtype=float)
    for j in range(b):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        denom = n_by_cluster[idx].sum()
        if denom <= 0:
            boot[j, :] = np.nan
        else:
            boot[j, :] = k_by_cluster[idx, :].sum(axis=0) / denom

    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    meta = {
        "n_clusters": int(n_clusters),
        "B": int(b),
        "cluster_cols": ",".join(cluster_cols),
    }
    return lo, hi, meta


def plot_panel_a(
    ax: plt.Axes,
    runs: pd.DataFrame,
    title: str = "A. Order-speed phenotype",
    show_title: bool = False,
) -> None:
    hb = ax.hexbin(
        runs["p_mean"],
        runs["v_mean_mps"],
        C=np.minimum(runs["duration_s"], 60.0),
        reduce_C_function=np.median,
        gridsize=30,
        mincnt=15,
        cmap="viridis",
        linewidths=0.0,
    )
    cbar = ax.figure.colorbar(hb, ax=ax, pad=0.018, fraction=0.055)
    cbar.set_label("Median run duration (s)", fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    q1, q2 = runs["p_mean"].quantile([1 / 3, 2 / 3]).to_numpy(float)
    add_top_state_arrows(ax, q1, q2)
    ax.set(
        xlabel="Mean polarisation",
        ylabel="Mean centroid speed (m/s)",
        xlim=(0, 1),
    )
    ax.set_box_aspect(0.92)
    if show_title:
        ax.set_title(title, pad=24)
    style_ax(ax)


def plot_ccdf_panel(
    ax: plt.Axes,
    runs: pd.DataFrame,
    value_col: str,
    grid: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    uncertainty: str = "errorbar",
    connect_lines: bool = True,
    cluster_cols: list[str] | None = None,
    show_title: bool = False,
) -> pd.DataFrame:
    rows = []
    positive_ci_lows: list[float] = []
    positive_ci_highs: list[float] = []
    cluster_cols = cluster_cols or []
    for state in STATE_ORDER:
        sub = runs.loc[runs["early_order_state"].astype(str).eq(state)]
        values = sub[value_col].to_numpy(float)
        values = values[np.isfinite(values) & (values > 0)]
        n = int(len(values))
        k = np.array([(values >= g).sum() for g in grid], dtype=int)
        y = k / max(n, 1)
        ci_low, ci_high = wilson_ci(k, n)
        ci_type = "wilson_binomial_95"
        if uncertainty == "cluster_bootstrap_band":
            boot_low, boot_high, boot_meta = cluster_bootstrap_ci(sub, value_col, grid, cluster_cols)
            if np.isfinite(boot_low).any() and np.isfinite(boot_high).any():
                ci_low, ci_high = boot_low, boot_high
                ci_type = "cluster_bootstrap_95"
            else:
                boot_meta = {"n_clusters": 0, "B": BOOTSTRAP_B, "cluster_cols": ",".join(cluster_cols)}
        else:
            boot_meta = {"n_clusters": np.nan, "B": np.nan, "cluster_cols": ""}
        if connect_lines:
            ax.plot(grid, y, color=STATE_COLORS[state], lw=1.7, label=state, zorder=3)
        if uncertainty in {"band", "cluster_bootstrap_band"}:
            band_idx = y > 0
            positive_ci_lows.extend(ci_low[band_idx & (ci_low > 0)].astype(float).tolist())
            positive_ci_highs.extend(ci_high[band_idx & (ci_high > 0)].astype(float).tolist())
            ax.fill_between(
                grid[band_idx],
                np.maximum(ci_low[band_idx], 1e-6),
                np.maximum(ci_high[band_idx], 1e-6),
                color=STATE_COLORS[state],
                alpha=CCDF_BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )
        elif uncertainty == "errorbar":
            marker_idx = (np.arange(len(grid)) % max(1, len(grid) // 10) == 0) & (y > 0)
            positive_ci_lows.extend(ci_low[marker_idx & (ci_low > 0)].astype(float).tolist())
            positive_ci_highs.extend(ci_high[marker_idx & (ci_high > 0)].astype(float).tolist())
            lo_display = np.maximum(ci_low[marker_idx], 1e-6)
            yerr_low = np.maximum(y[marker_idx] - lo_display, 0.0)
            yerr_high = np.maximum(ci_high[marker_idx] - y[marker_idx], 0.0)
            ax.errorbar(
                grid[marker_idx],
                y[marker_idx],
                yerr=np.vstack([yerr_low, yerr_high]),
                fmt="o",
                ms=3.0,
                mfc=STATE_COLORS[state],
                mec=STATE_COLORS[state],
                color=STATE_COLORS[state],
                ecolor=STATE_COLORS[state],
                label=state if not connect_lines else "_nolegend_",
                elinewidth=0.65,
                capsize=1.5,
                capthick=0.65,
                lw=0,
                zorder=4,
            )
        else:
            raise ValueError(f"Unknown uncertainty style: {uncertainty}")
        for threshold, survival, count, low, high in zip(grid, y, k, ci_low, ci_high):
            rows.append(
                {
                    "panel": title,
                    "value_col": value_col,
                    "early_order_state": state,
                    "threshold": float(threshold),
                    "survival": float(survival),
                    "n_surviving_at_threshold": int(count),
                    "n_runs": n,
                    "ci_low_95": float(low),
                    "ci_high_95": float(high),
                    "ci_type": ci_type,
                    "bootstrap_B": boot_meta["B"],
                    "bootstrap_n_clusters": boot_meta["n_clusters"],
                    "bootstrap_cluster_cols": boot_meta["cluster_cols"],
                }
            )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(plain_log_tick))
    if positive_ci_lows:
        y_min = max(min(positive_ci_lows) * 0.62, 1e-5)
    else:
        y_min = 8e-4
    if positive_ci_highs:
        y_max = min(max(max(positive_ci_highs) * 1.08, 1.05), 1.18)
    else:
        y_max = 1.05
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(title, pad=7)
    style_ax(ax)
    return pd.DataFrame(rows)


def build_publication_figure(
    runs: pd.DataFrame,
    duration_grid: np.ndarray,
    length_grid: np.ndarray,
    cluster_cols: list[str],
    *,
    close_figure: bool = True,
    save_outputs: bool = True,
) -> tuple[plt.Figure, list[str]]:
    """Build only the selected publication Figure 3 and its audit files."""
    created: list[str] = []
    fig = plt.figure(figsize=(8.4, 6.55))
    grid = fig.add_gridspec(
        2, 4, height_ratios=[1.0, 1.06], hspace=0.42, wspace=0.50,
        left=0.085, right=0.965, bottom=0.095, top=0.925,
    )
    ax_duration = fig.add_subplot(grid[0, 0:2])
    ax_length = fig.add_subplot(grid[0, 2:4])
    ax_phenotype = fig.add_subplot(grid[1, 1:3])
    audit_duration = plot_ccdf_panel(
        ax_duration, runs, "duration_s", duration_grid,
        "A. Duration survivor by order state", "Run duration T (s)", r"$P\, (T \geq t)$",
        uncertainty="cluster_bootstrap_band", cluster_cols=cluster_cols,
    )
    audit_length = plot_ccdf_panel(
        ax_length, runs, "run_length_m", length_grid,
        "B. Length survivor by order state", "Run length L (m)", r"$P\, (L \geq l)$",
        uncertainty="cluster_bootstrap_band", cluster_cols=cluster_cols,
    )
    ax_duration.legend(
        title="Order state", fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
        frameon=False, loc="lower left", ncol=3, handlelength=1.6,
        columnspacing=0.95, borderaxespad=0.2,
    )
    axes_before_panel_c = list(fig.axes)
    plot_panel_a(ax_phenotype, runs, "C. Order-speed phenotype")
    add_three_panel_labels(ax_duration, ax_length, ax_phenotype)
    panel_c_axes = [ax_phenotype] + [ax for ax in fig.axes if ax not in axes_before_panel_c]
    shift_axes_x(panel_c_axes, -0.028)

    stem = "figure3_three_panel_candidate_house_style_cluster_bootstrap"
    audit_path = OUT_DIR / f"{stem}_ccdf_audit.csv"
    manifest = {
        "source_cache": str(CACHE_PATH),
        "n_runs": int(len(runs)),
        "layout": "compact house-style two panels on top, normal-width order-speed phenotype centered below",
        "uncertainty_style": "95% cluster bootstrap confidence bands over match-team clusters",
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_cols": cluster_cols,
        "n_clusters": int(runs[cluster_cols].drop_duplicates().shape[0]) if cluster_cols else 0,
        "style_reference": "Figure 2/4 panel-title grammar with compact spacing and in-panel legend",
        "panel_A": "duration_s CCDF by p_early_3s tercile",
        "panel_B": "run_length_m CCDF by p_early_3s tercile",
        "panel_C": {
            "x": "p_mean, whole-run mean polarisation",
            "y": "v_mean_mps, whole-run mean centroid speed",
            "colour": "median duration_s capped at 60 s",
            "state_arrows": "terciles of p_mean",
        },
        "created_files": [str(OUT_DIR / f"{stem}.png"), str(OUT_DIR / f"{stem}.pdf"), str(audit_path)],
    }
    if save_outputs:
        for ext in ("png", "pdf"):
            path = OUT_DIR / f"{stem}.{ext}"
            fig.savefig(path, dpi=320)
            created.append(str(path))
        pd.concat([audit_duration, audit_length], ignore_index=True).to_csv(audit_path, index=False)
        created.append(str(audit_path))
        manifest_path = OUT_DIR / f"{stem}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        created.append(str(manifest_path))
    if close_figure:
        plt.close(fig)
    return fig, created


def main(*, all_variants: bool = False) -> None:
    configure_paper_plotting(base=PAPER_FONT_BASE)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "axes.titlesize": PAPER_FONT_BASE * 1.15,
            "axes.labelsize": PAPER_FONT_BASE,
            "legend.fontsize": PAPER_FONT_BASE * 0.85,
            "legend.title_fontsize": PAPER_FONT_BASE * 0.9,
            "xtick.labelsize": PAPER_FONT_BASE * 0.9,
            "ytick.labelsize": PAPER_FONT_BASE * 0.9,
            "axes.unicode_minus": False,
        }
    )
    runs = load_runs()
    cluster_cols = best_cluster_cols(runs)
    duration_grid = np.arange(1.0, min(75.0, max(35.0, runs["duration_s"].quantile(0.995))) + 1.0, 2.0)
    length_grid = np.linspace(1.0, min(85.0, max(25.0, runs["run_length_m"].quantile(0.995))), 34)
    if not all_variants:
        _fig, created = build_publication_figure(runs, duration_grid, length_grid, cluster_cols)
        print("OUTPUT_DIR", OUT_DIR)
        print("N_RUNS", len(runs))
        print("BOOTSTRAP_CLUSTER_COLS", cluster_cols)
        print("BOOTSTRAP_B", BOOTSTRAP_B)
        print("CREATED")
        for path in created:
            print(path)
        return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.75),
        gridspec_kw={"width_ratios": [1.24, 1.0, 1.0], "wspace": 0.46},
    )
    plot_panel_a(axes[0], runs, "A. Order-speed phenotype")
    audit_duration = plot_ccdf_panel(
        axes[1],
        runs,
        "duration_s",
        duration_grid,
        "B. Duration survival by order state",
        "Run duration T (s)",
        r"$P\, (T \geq t)$",
    )
    audit_length = plot_ccdf_panel(
        axes[2],
        runs,
        "run_length_m",
        length_grid,
        "C. Length survival by order state",
        "Run length L (m)",
        r"$P\, (L \geq l)$",
    )
    add_three_panel_labels(axes[0], axes[1], axes[2])
    add_horizontal_order_legend(fig, axes[1], y=0.965)
    fig.subplots_adjust(top=0.82, bottom=0.19, left=0.06, right=0.985)

    created = []
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"figure3_three_panel_candidate.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=320)
        created.append(str(path))
    plt.close(fig)

    audit_path = OUT_DIR / "figure3_three_panel_candidate_ccdf_audit.csv"
    pd.concat([audit_duration, audit_length], ignore_index=True).to_csv(audit_path, index=False)
    created.append(str(audit_path))

    manifest = {
        "source_cache": str(CACHE_PATH),
        "n_runs": int(len(runs)),
        "panel_A": {
            "x": "p_mean, whole-run mean polarisation",
            "y": "v_mean_mps, whole-run mean centroid speed",
            "colour": "median duration_s capped at 60 s",
            "state_arrows": "terciles of p_mean",
        },
        "panels_BC": {
            "order_state": "terciles of p_early_3s",
            "B": "duration_s CCDF",
            "C": "run_length_m CCDF",
        },
        "created_files": created,
    }
    manifest_path = OUT_DIR / "figure3_three_panel_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(str(manifest_path))

    fig2 = plt.figure(figsize=(8.4, 7.2))
    gs = fig2.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.12],
        hspace=0.58,
        wspace=0.34,
        left=0.08,
        right=0.965,
        bottom=0.09,
        top=0.88,
    )
    ax_duration = fig2.add_subplot(gs[0, 0])
    ax_length = fig2.add_subplot(gs[0, 1])
    ax_pheno = fig2.add_subplot(gs[1, :])
    audit_duration_2 = plot_ccdf_panel(
        ax_duration,
        runs,
        "duration_s",
        duration_grid,
        "A. Duration survival by order state",
        "Run duration T (s)",
        r"$P\, (T \geq t)$",
    )
    audit_length_2 = plot_ccdf_panel(
        ax_length,
        runs,
        "run_length_m",
        length_grid,
        "B. Length survival by order state",
        "Run length L (m)",
        r"$P\, (L \geq l)$",
    )
    plot_panel_a(ax_pheno, runs, "C. Order-speed phenotype")
    add_three_panel_labels(ax_duration, ax_length, ax_pheno)
    add_horizontal_order_legend(fig2, ax_duration, y=0.965)

    for ext in ("png", "pdf"):
        path = OUT_DIR / f"figure3_three_panel_candidate_2top_1bottom.{ext}"
        fig2.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=320)
        created.append(str(path))
    plt.close(fig2)

    audit_path_2 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_ccdf_audit.csv"
    pd.concat([audit_duration_2, audit_length_2], ignore_index=True).to_csv(audit_path_2, index=False)
    created.append(str(audit_path_2))

    manifest_2 = {
        "source_cache": str(CACHE_PATH),
        "n_runs": int(len(runs)),
        "layout": "two panels on top, order-speed phenotype spanning bottom row",
        "panel_A": "duration_s CCDF by p_early_3s tercile",
        "panel_B": "run_length_m CCDF by p_early_3s tercile",
        "panel_C": {
            "x": "p_mean, whole-run mean polarisation",
            "y": "v_mean_mps, whole-run mean centroid speed",
            "colour": "median duration_s capped at 60 s",
            "state_arrows": "terciles of p_mean",
        },
        "created_files": [
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom.png"),
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom.pdf"),
            str(audit_path_2),
        ],
    }
    manifest_path_2 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_manifest.json"
    manifest_path_2.write_text(json.dumps(manifest_2, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(str(manifest_path_2))

    fig3 = plt.figure(figsize=(8.4, 7.2))
    gs3 = fig3.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.08],
        hspace=0.46,
        wspace=0.52,
        left=0.08,
        right=0.965,
        bottom=0.09,
        top=0.88,
    )
    ax_duration_3 = fig3.add_subplot(gs3[0, 0:2])
    ax_length_3 = fig3.add_subplot(gs3[0, 2:4])
    ax_pheno_3 = fig3.add_subplot(gs3[1, 1:3])
    audit_duration_3 = plot_ccdf_panel(
        ax_duration_3,
        runs,
        "duration_s",
        duration_grid,
        "A. Duration survival by order state",
        "Run duration T (s)",
        r"$P\, (T \geq t)$",
    )
    audit_length_3 = plot_ccdf_panel(
        ax_length_3,
        runs,
        "run_length_m",
        length_grid,
        "B. Length survival by order state",
        "Run length L (m)",
        r"$P\, (L \geq l)$",
    )
    axes_before_panel_c = list(fig3.axes)
    plot_panel_a(ax_pheno_3, runs, "C. Order-speed phenotype")
    panel_c_axes = [ax_pheno_3] + [ax for ax in fig3.axes if ax not in axes_before_panel_c]
    shift_axes_x(panel_c_axes, -0.028)
    add_three_panel_labels(ax_duration_3, ax_length_3, ax_pheno_3)
    add_horizontal_order_legend(fig3, ax_duration_3, y=0.965)

    for ext in ("png", "pdf"):
        path = OUT_DIR / f"figure3_three_panel_candidate_2top_1bottom_centered.{ext}"
        fig3.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=320)
        created.append(str(path))
    plt.close(fig3)

    audit_path_3 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_ccdf_audit.csv"
    pd.concat([audit_duration_3, audit_length_3], ignore_index=True).to_csv(audit_path_3, index=False)
    created.append(str(audit_path_3))

    manifest_3 = {
        "source_cache": str(CACHE_PATH),
        "n_runs": int(len(runs)),
        "layout": "two panels on top, normal-width order-speed phenotype centered on bottom row",
        "panel_A": "duration_s CCDF by p_early_3s tercile",
        "panel_B": "run_length_m CCDF by p_early_3s tercile",
        "panel_C": {
            "x": "p_mean, whole-run mean polarisation",
            "y": "v_mean_mps, whole-run mean centroid speed",
            "colour": "median duration_s capped at 60 s",
            "state_arrows": "terciles of p_mean",
        },
        "created_files": [
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered.png"),
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered.pdf"),
            str(audit_path_3),
        ],
    }
    manifest_path_3 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_manifest.json"
    manifest_path_3.write_text(json.dumps(manifest_3, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(str(manifest_path_3))

    fig4 = plt.figure(figsize=(8.4, 7.2))
    gs4 = fig4.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.08],
        hspace=0.46,
        wspace=0.52,
        left=0.08,
        right=0.965,
        bottom=0.09,
        top=0.88,
    )
    ax_duration_4 = fig4.add_subplot(gs4[0, 0:2])
    ax_length_4 = fig4.add_subplot(gs4[0, 2:4])
    ax_pheno_4 = fig4.add_subplot(gs4[1, 1:3])
    audit_duration_4 = plot_ccdf_panel(
        ax_duration_4,
        runs,
        "duration_s",
        duration_grid,
        "A. Duration survival by order state",
        "Run duration T (s)",
        r"$P\, (T \geq t)$",
        uncertainty="band",
    )
    audit_length_4 = plot_ccdf_panel(
        ax_length_4,
        runs,
        "run_length_m",
        length_grid,
        "B. Length survival by order state",
        "Run length L (m)",
        r"$P\, (L \geq l)$",
        uncertainty="band",
    )
    axes_before_panel_c_4 = list(fig4.axes)
    plot_panel_a(ax_pheno_4, runs, "C. Order-speed phenotype")
    panel_c_axes_4 = [ax_pheno_4] + [ax for ax in fig4.axes if ax not in axes_before_panel_c_4]
    shift_axes_x(panel_c_axes_4, -0.028)
    add_three_panel_labels(ax_duration_4, ax_length_4, ax_pheno_4)
    add_horizontal_order_legend(fig4, ax_duration_4, y=0.965)

    for ext in ("png", "pdf"):
        path = OUT_DIR / f"figure3_three_panel_candidate_2top_1bottom_centered_shaded_uncertainty.{ext}"
        fig4.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=320)
        created.append(str(path))
    plt.close(fig4)

    audit_path_4 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_shaded_uncertainty_ccdf_audit.csv"
    pd.concat([audit_duration_4, audit_length_4], ignore_index=True).to_csv(audit_path_4, index=False)
    created.append(str(audit_path_4))

    manifest_4 = {
        "source_cache": str(CACHE_PATH),
        "n_runs": int(len(runs)),
        "layout": "two panels on top, normal-width order-speed phenotype centered on bottom row",
        "uncertainty_style": "95% Wilson binomial confidence bands; no CCDF markers",
        "panel_A": "duration_s CCDF by p_early_3s tercile",
        "panel_B": "run_length_m CCDF by p_early_3s tercile",
        "panel_C": {
            "x": "p_mean, whole-run mean polarisation",
            "y": "v_mean_mps, whole-run mean centroid speed",
            "colour": "median duration_s capped at 60 s",
            "state_arrows": "terciles of p_mean",
        },
        "created_files": [
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_shaded_uncertainty.png"),
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_shaded_uncertainty.pdf"),
            str(audit_path_4),
        ],
    }
    manifest_path_4 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_shaded_uncertainty_manifest.json"
    manifest_path_4.write_text(json.dumps(manifest_4, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(str(manifest_path_4))

    fig5 = plt.figure(figsize=(8.4, 7.2))
    gs5 = fig5.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.08],
        hspace=0.46,
        wspace=0.52,
        left=0.08,
        right=0.965,
        bottom=0.09,
        top=0.88,
    )
    ax_duration_5 = fig5.add_subplot(gs5[0, 0:2])
    ax_length_5 = fig5.add_subplot(gs5[0, 2:4])
    ax_pheno_5 = fig5.add_subplot(gs5[1, 1:3])
    audit_duration_5 = plot_ccdf_panel(
        ax_duration_5,
        runs,
        "duration_s",
        duration_grid,
        "A. Duration survival by order state",
        "Run duration T (s)",
        r"$P\, (T \geq t)$",
        uncertainty="errorbar",
        connect_lines=False,
    )
    audit_length_5 = plot_ccdf_panel(
        ax_length_5,
        runs,
        "run_length_m",
        length_grid,
        "B. Length survival by order state",
        "Run length L (m)",
        r"$P\, (L \geq l)$",
        uncertainty="errorbar",
        connect_lines=False,
    )
    axes_before_panel_c_5 = list(fig5.axes)
    plot_panel_a(ax_pheno_5, runs, "C. Order-speed phenotype")
    panel_c_axes_5 = [ax_pheno_5] + [ax for ax in fig5.axes if ax not in axes_before_panel_c_5]
    shift_axes_x(panel_c_axes_5, -0.028)
    add_three_panel_labels(ax_duration_5, ax_length_5, ax_pheno_5)
    add_horizontal_order_legend(fig5, ax_duration_5, y=0.965)

    for ext in ("png", "pdf"):
        path = OUT_DIR / f"figure3_three_panel_candidate_2top_1bottom_centered_errorbars_no_lines.{ext}"
        fig5.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=320)
        created.append(str(path))
    plt.close(fig5)

    audit_path_5 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_errorbars_no_lines_ccdf_audit.csv"
    pd.concat([audit_duration_5, audit_length_5], ignore_index=True).to_csv(audit_path_5, index=False)
    created.append(str(audit_path_5))

    manifest_5 = {
        "source_cache": str(CACHE_PATH),
        "n_runs": int(len(runs)),
        "layout": "two panels on top, normal-width order-speed phenotype centered on bottom row",
        "uncertainty_style": "95% Wilson binomial error bars; CCDF markers not connected by lines",
        "panel_A": "duration_s CCDF by p_early_3s tercile",
        "panel_B": "run_length_m CCDF by p_early_3s tercile",
        "panel_C": {
            "x": "p_mean, whole-run mean polarisation",
            "y": "v_mean_mps, whole-run mean centroid speed",
            "colour": "median duration_s capped at 60 s",
            "state_arrows": "terciles of p_mean",
        },
        "created_files": [
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_errorbars_no_lines.png"),
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_errorbars_no_lines.pdf"),
            str(audit_path_5),
        ],
    }
    manifest_path_5 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_errorbars_no_lines_manifest.json"
    manifest_path_5.write_text(json.dumps(manifest_5, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(str(manifest_path_5))

    fig6 = plt.figure(figsize=(8.4, 7.2))
    gs6 = fig6.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.08],
        hspace=0.46,
        wspace=0.52,
        left=0.08,
        right=0.965,
        bottom=0.09,
        top=0.88,
    )
    ax_duration_6 = fig6.add_subplot(gs6[0, 0:2])
    ax_length_6 = fig6.add_subplot(gs6[0, 2:4])
    ax_pheno_6 = fig6.add_subplot(gs6[1, 1:3])
    audit_duration_6 = plot_ccdf_panel(
        ax_duration_6,
        runs,
        "duration_s",
        duration_grid,
        "A. Duration survival by order state",
        "Run duration T (s)",
        r"$P\, (T \geq t)$",
        uncertainty="cluster_bootstrap_band",
        cluster_cols=cluster_cols,
    )
    audit_length_6 = plot_ccdf_panel(
        ax_length_6,
        runs,
        "run_length_m",
        length_grid,
        "B. Length survival by order state",
        "Run length L (m)",
        r"$P\, (L \geq l)$",
        uncertainty="cluster_bootstrap_band",
        cluster_cols=cluster_cols,
    )
    axes_before_panel_c_6 = list(fig6.axes)
    plot_panel_a(ax_pheno_6, runs, "C. Order-speed phenotype")
    panel_c_axes_6 = [ax_pheno_6] + [ax for ax in fig6.axes if ax not in axes_before_panel_c_6]
    shift_axes_x(panel_c_axes_6, -0.028)
    add_three_panel_labels(ax_duration_6, ax_length_6, ax_pheno_6)
    add_horizontal_order_legend(fig6, ax_duration_6, y=0.965)

    for ext in ("png", "pdf"):
        path = OUT_DIR / f"figure3_three_panel_candidate_2top_1bottom_centered_cluster_bootstrap_bands.{ext}"
        fig6.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=320)
        created.append(str(path))
    plt.close(fig6)

    audit_path_6 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_cluster_bootstrap_bands_ccdf_audit.csv"
    pd.concat([audit_duration_6, audit_length_6], ignore_index=True).to_csv(audit_path_6, index=False)
    created.append(str(audit_path_6))

    manifest_6 = {
        "source_cache": str(CACHE_PATH),
        "n_runs": int(len(runs)),
        "layout": "two panels on top, normal-width order-speed phenotype centered on bottom row",
        "uncertainty_style": "95% cluster bootstrap confidence bands over match-team clusters",
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_cols": cluster_cols,
        "n_clusters": int(runs[cluster_cols].drop_duplicates().shape[0]) if cluster_cols else 0,
        "panel_A": "duration_s CCDF by p_early_3s tercile",
        "panel_B": "run_length_m CCDF by p_early_3s tercile",
        "panel_C": {
            "x": "p_mean, whole-run mean polarisation",
            "y": "v_mean_mps, whole-run mean centroid speed",
            "colour": "median duration_s capped at 60 s",
            "state_arrows": "terciles of p_mean",
        },
        "created_files": [
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_cluster_bootstrap_bands.png"),
            str(OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_cluster_bootstrap_bands.pdf"),
            str(audit_path_6),
        ],
    }
    manifest_path_6 = OUT_DIR / "figure3_three_panel_candidate_2top_1bottom_centered_cluster_bootstrap_bands_manifest.json"
    manifest_path_6.write_text(json.dumps(manifest_6, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(str(manifest_path_6))

    fig7 = plt.figure(figsize=(8.4, 6.55))
    gs7 = fig7.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.06],
        hspace=0.42,
        wspace=0.50,
        left=0.085,
        right=0.965,
        bottom=0.095,
        top=0.925,
    )
    ax_duration_7 = fig7.add_subplot(gs7[0, 0:2])
    ax_length_7 = fig7.add_subplot(gs7[0, 2:4])
    ax_pheno_7 = fig7.add_subplot(gs7[1, 1:3])
    audit_duration_7 = plot_ccdf_panel(
        ax_duration_7,
        runs,
        "duration_s",
        duration_grid,
        "A. Duration survivor by order state",
        "Run duration T (s)",
        r"$P\, (T \geq t)$",
        uncertainty="cluster_bootstrap_band",
        cluster_cols=cluster_cols,
    )
    audit_length_7 = plot_ccdf_panel(
        ax_length_7,
        runs,
        "run_length_m",
        length_grid,
        "B. Length survivor by order state",
        "Run length L (m)",
        r"$P\, (L \geq l)$",
        uncertainty="cluster_bootstrap_band",
        cluster_cols=cluster_cols,
    )
    ax_duration_7.legend(
        title="Order state",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        frameon=False,
        loc="lower left",
        ncol=3,
        handlelength=1.6,
        columnspacing=0.95,
        borderaxespad=0.2,
    )
    axes_before_panel_c_7 = list(fig7.axes)
    plot_panel_a(ax_pheno_7, runs, "C. Order-speed phenotype")
    add_three_panel_labels(ax_duration_7, ax_length_7, ax_pheno_7)
    panel_c_axes_7 = [ax_pheno_7] + [ax for ax in fig7.axes if ax not in axes_before_panel_c_7]
    shift_axes_x(panel_c_axes_7, -0.028)

    for ext in ("png", "pdf"):
        path = OUT_DIR / f"figure3_three_panel_candidate_house_style_cluster_bootstrap.{ext}"
        fig7.savefig(path, dpi=320)
        created.append(str(path))
    plt.close(fig7)

    audit_path_7 = OUT_DIR / "figure3_three_panel_candidate_house_style_cluster_bootstrap_ccdf_audit.csv"
    pd.concat([audit_duration_7, audit_length_7], ignore_index=True).to_csv(audit_path_7, index=False)
    created.append(str(audit_path_7))

    manifest_7 = {
        "source_cache": str(CACHE_PATH),
        "n_runs": int(len(runs)),
        "layout": "compact house-style two panels on top, normal-width order-speed phenotype centered below",
        "uncertainty_style": "95% cluster bootstrap confidence bands over match-team clusters",
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_cols": cluster_cols,
        "n_clusters": int(runs[cluster_cols].drop_duplicates().shape[0]) if cluster_cols else 0,
        "style_reference": "Figure 2/4 panel-title grammar with compact spacing and in-panel legend",
        "panel_A": "duration_s CCDF by p_early_3s tercile",
        "panel_B": "run_length_m CCDF by p_early_3s tercile",
        "panel_C": {
            "x": "p_mean, whole-run mean polarisation",
            "y": "v_mean_mps, whole-run mean centroid speed",
            "colour": "median duration_s capped at 60 s",
            "state_arrows": "terciles of p_mean",
        },
        "created_files": [
            str(OUT_DIR / "figure3_three_panel_candidate_house_style_cluster_bootstrap.png"),
            str(OUT_DIR / "figure3_three_panel_candidate_house_style_cluster_bootstrap.pdf"),
            str(audit_path_7),
        ],
    }
    manifest_path_7 = OUT_DIR / "figure3_three_panel_candidate_house_style_cluster_bootstrap_manifest.json"
    manifest_path_7.write_text(json.dumps(manifest_7, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(str(manifest_path_7))
    print("OUTPUT_DIR", OUT_DIR)
    print("N_RUNS", len(runs))
    print("BOOTSTRAP_CLUSTER_COLS", cluster_cols)
    print("BOOTSTRAP_B", BOOTSTRAP_B)
    print("CREATED")
    for path in created:
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the selected publication Figure 3.")
    parser.add_argument("--all-variants", action="store_true", help="Also rebuild superseded exploratory layout variants")
    main(all_variants=parser.parse_args().all_variants)
