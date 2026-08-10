from __future__ import annotations

import json
import importlib.util
import math
import os
from datetime import datetime
from pathlib import Path

_SCRIPT_PATH_FOR_MPL = Path(__file__).resolve()
_MPLCONFIG_DIR = _SCRIPT_PATH_FOR_MPL.parents[1] / "outputs" / "final" / "figure2_transport_phenotype" / "audits" / ".matplotlib"
_MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

try:
    from analysis.levy_paper.util.paper_utils import cache_path, configure_paper_plotting, draw_panel_letters
except Exception:  # pragma: no cover
    cache_path = None
    configure_paper_plotting = None
    draw_panel_letters = None


SCRIPT_PATH = Path(__file__).resolve()
LEVY_DIR = SCRIPT_PATH.parents[1]
DATA_DIR = LEVY_DIR / "data"
FIGURES_DIR = LEVY_DIR / "figures"
OUT_DIR = LEVY_DIR / "outputs" / "final" / "figure2_transport_phenotype"
MAIN_DIR = OUT_DIR / "main"
UNCERTAINTY_DIR = OUT_DIR / "uncertainty"
AUDIT_DIR = OUT_DIR / "audits"
NOTES_DIR = OUT_DIR / "notes"
MANIFEST_DIR = OUT_DIR / "manifests"

for _dir in [OUT_DIR, MAIN_DIR, UNCERTAINTY_DIR, AUDIT_DIR, NOTES_DIR, MANIFEST_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

OUTPUT_VERSION = "v2"
STYLE = {
    "black": "black",
    "red": "#d62728",
    "blue": "blue",
    "grey": "0.55",
    "light_grey": "0.82",
    "shade_grey": "0.93",
    "line_width": 2.0,
    "fit_line_width": 2.2,
    "marker_size": 4.5,
    "error_lw": 1.0,
    "error_capsize": 2.5,
    "sparse_error_lw": 0.7,
    "sparse_alpha": 0.45,
    "font_size": 10,
    "title_size": 13,
    "label_size": 10,
    "legend_size": 8.5,
}
FIGURE2_COLORS = {
    "centroid": STYLE["black"],
    "player": STYLE["red"],
    "exp_null": STYLE["grey"],
    "player_abs": STYLE["red"],
    "player_rel": STYLE["blue"],
    "relative": STYLE["red"],
    "cross_term": STYLE["blue"],
    "fit_window": STYLE["shade_grey"],
    "zero_line": STYLE["grey"],
    "half_line": STYLE["light_grey"],
}
FIGURE2_TEXT = {
    "panel_a_title": "A. Duration survivor",
    "panel_a_xlabel": "Run duration T (s)",
    "panel_a_ylabel": r"$P(T \geq t)$",
    "panel_b_title": "B. Length survivor",
    "panel_b_xlabel": "Run length L (m)",
    "panel_b_ylabel": r"$P(L \geq l)$",
    "panel_c_title": "C. MSD scaling",
    "panel_c_xlabel": r"Lag $\tau$ (s)",
    "panel_c_ylabel": r"MSD (m$^2$)",
    "panel_d_title": "D. MSD decomposition",
    "panel_d_xlabel": r"Lag $\tau$ (s)",
    "panel_d_ylabel": "Fraction of player MSD",
    "centroid_label": "centroid",
    "player_label": "player",
    "exp_null_label": "exp. null, centroid mean",
    "exp_centroid_label": "exp. null, centroid mean",
    "exp_player_label": "exp. null, player mean",
    "player_abs_label": "player (pitch FOR)",
    "player_rel_label": "player (centroid FOR)",
    "relative_label": "motion within formation",
    "cross_term_label": "cross term",
    "centroid_translation_label": "centroid translation",
}
FIGURE2_PLOT = {
    "figure_size": (8.4, 6.55),
    "tight_layout": True,
    "subplots_adjust": {},
    "panel_a": {
        "series_draw": "dots",
        "reference_draw": "line",
        "xscale": "log",
        "yscale": "log",
        "xlim": (0.4, 140.0),
        "ylim": None,
        "show_exp_reference": True,
        "reference_line_style": "-",
        "legend_loc": "lower left",
        "legend_frameon": False,
        "max_marker_points": 115,
    },
    "panel_b": {
        "series_draw": "dots",
        "reference_draw": "line",
        "xscale": "log",
        "yscale": "log",
        "xlim": None,
        "ylim": None,
        "show_exp_reference": True,
        "reference_line_style": "-",
        "legend_loc": "lower left",
        "legend_frameon": False,
        "max_marker_points": 115,
    },
    "panel_c": {
        "series_draw": "dots",
        "fit_draw": "line",
        "xscale": "log",
        "yscale": "log",
        "xlim": None,
        "ylim": None,
        "show_fit_window": False,
        "show_fit_lines": True,
        "fit_window_alpha": 1.0,
        "fit_line_style": "-",
        "legend_loc": "upper left",
        "legend_frameon": False,
        "max_marker_points": 70,
    },
    "panel_d": {
        "series_draw": "line",
        "xscale": "linear",
        "yscale": "linear",
        "xlim": None,
        "ylim": None,
        "show_reference_lines": True,
        "legend_loc": "upper right",
        "legend_frameon": False,
        "legend_ncol": 1,
        "max_marker_points": 80,
    },
}
ORDER_COLORS = {
    "Low": STYLE["red"],
    "Mid": STYLE["black"],
    "High": STYLE["blue"],
}
STYLE_HELPER_USED = "analysis.levy_paper.util.paper_utils.configure_paper_plotting"
TACTICAL_LAG_WINDOW_S = (5.0, 30.0)
EARLY_LAG_WINDOW_S = (1.0, 4.0)
DECOMPOSITION_LAGS_S = np.arange(1, 61, dtype=int)
BOOTSTRAP_B = 200
BOOTSTRAP_UNIT = "match_id+match_phase+team+source_key"
CLUSTER_COLS = ["match_id", "match_phase", "team", "source_key"]
RNG = np.random.default_rng(20260630)


def _required_cache_path(filename: str) -> Path:
    if cache_path is not None:
        return cache_path(filename)
    return DATA_DIR / "processed" / "all_team_2020_2021_sticky_active" / filename


RUNS_PATH = _required_cache_path("runs_long_2020_2021_all_teams_pitchfix_sticky_active.parquet")
MSD_PATH = _required_cache_path("msd_long_2020_2021_all_teams_pitchfix_sticky_active.parquet")
TRAJECTORY_PATH = _required_cache_path("trajectory_long_2020_2021_all_teams_pitchfix_sticky_active.parquet")

if configure_paper_plotting is None:
    style_path = LEVY_DIR / "util" / "paper_utils.py"
    if style_path.exists():
        spec = importlib.util.spec_from_file_location("levy_paper_paper_utils", style_path)
        if spec is not None and spec.loader is not None:
            style_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(style_module)
            configure_paper_plotting = getattr(style_module, "configure_paper_plotting", None)
            draw_panel_letters = getattr(style_module, "draw_panel_letters", draw_panel_letters)


def _deep_update(base: dict, updates: dict | None) -> None:
    if not updates:
        return
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _sync_figure2_aesthetics(aesthetics: dict | None = None) -> None:
    if not aesthetics:
        return
    has_sections = any(key in aesthetics for key in ["style", "colors", "colours", "text", "plot"])
    if not has_sections:
        STYLE.update(aesthetics)
        return
    STYLE.update(aesthetics.get("style", {}))
    FIGURE2_COLORS.update(aesthetics.get("colors", {}))
    FIGURE2_COLORS.update(aesthetics.get("colours", {}))
    FIGURE2_TEXT.update(aesthetics.get("text", {}))
    _deep_update(FIGURE2_PLOT, aesthetics.get("plot", {}))


def _sync_style(style: dict | None = None) -> None:
    if style:
        _sync_figure2_aesthetics(style)


def apply_project_style(style: dict | None = None) -> None:
    global STYLE_HELPER_USED
    _sync_style(style)
    if configure_paper_plotting is not None:
        configure_paper_plotting(base=float(STYLE["font_size"]))
        STYLE_HELPER_USED = "analysis.levy_paper.util.paper_utils.configure_paper_plotting"
    else:
        STYLE_HELPER_USED = "none found"
        plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.size": STYLE["font_size"],
            "axes.titlesize": STYLE["title_size"],
            "axes.labelsize": STYLE["label_size"],
            "legend.fontsize": STYLE["legend_size"],
            "xtick.labelsize": STYLE["font_size"] * 0.9,
            "ytick.labelsize": STYLE["font_size"] * 0.9,
            "axes.unicode_minus": False,
        }
    )


def set_figure2_style(style: dict | None = None) -> None:
    apply_figure2_aesthetics(style)


def apply_figure2_aesthetics(aesthetics: dict | None = None) -> None:
    _sync_figure2_aesthetics(aesthetics)
    apply_project_style()


apply_project_style(STYLE)


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def save_figure(fig: plt.Figure, outdir: Path, basename: str, formats: tuple[str, ...] = ("png", "pdf"), dpi: int = 320) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in formats:
        path = outdir / f"{basename}.{ext}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    return paths


def _style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=3)
    _apply_plain_log_tick_labels(ax)


def _plain_log_tick(value: float, _pos: int | None = None) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = int(round(np.log10(value)))
    if not np.isclose(value, 10**exponent, rtol=1e-9, atol=0.0):
        return ""
    if exponent == 0:
        return "1"
    if -3 <= exponent < 0:
        return f"{10**exponent:.{abs(exponent)}f}"
    if 0 < exponent <= 3:
        return f"{int(10**exponent)}"
    return f"1e{exponent}"


def _apply_plain_log_tick_labels(ax) -> None:
    formatter = FuncFormatter(_plain_log_tick)
    if ax.get_xscale() == "log":
        ax.xaxis.set_major_formatter(formatter)
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(formatter)


def _load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "runs": pd.read_parquet(RUNS_PATH),
        "msd": pd.read_parquet(MSD_PATH),
        "trajectory": pd.read_parquet(TRAJECTORY_PATH),
    }


def _ccdf(values: pd.Series | np.ndarray) -> pd.DataFrame:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n == 0:
        return pd.DataFrame({"x": [], "ccdf": [], "n": []})
    return pd.DataFrame({"x": x, "ccdf": (n - np.arange(n, dtype=float)) / n, "n": n})


def _text(key: str, default: str | None = None) -> str:
    return str(FIGURE2_TEXT.get(key, key if default is None else default))


def _plot_opt(panel: str, key: str, default=None):
    panel_cfg = FIGURE2_PLOT.get(panel, {})
    if isinstance(panel_cfg, dict) and key in panel_cfg:
        return panel_cfg[key]
    return FIGURE2_PLOT.get(key, default)


def _sample_marker_points(x, y, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    ok = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[ok]
    y_arr = y_arr[ok]
    if max_points is None or max_points <= 0 or len(x_arr) <= max_points:
        return x_arr, y_arr
    idx = np.unique(np.linspace(0, len(x_arr) - 1, int(max_points)).astype(int))
    return x_arr[idx], y_arr[idx]


def _draw_xy(
    ax,
    x,
    y,
    *,
    label: str | None,
    color: str,
    panel: str,
    draw_key: str = "series_draw",
    lw: float | None = None,
    ls: str = "-",
    alpha: float = 1.0,
    zorder: int = 3,
):
    draw = str(_plot_opt(panel, draw_key, "line")).lower().replace("_", "+")
    line_requested = draw in {"line", "line+dots", "line+points", "both"}
    dots_requested = draw in {"dots", "points", "line+dots", "line+points", "both"}
    if not line_requested and not dots_requested:
        return None
    handle = None
    if line_requested:
        handle = ax.plot(
            x,
            y,
            color=color,
            lw=STYLE["line_width"] if lw is None else lw,
            ls=ls,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )[0]
    if dots_requested:
        xs, ys = _sample_marker_points(x, y, _plot_opt(panel, "max_marker_points", 80))
        dot_label = label if not line_requested else None
        dots = ax.plot(
            xs,
            ys,
            color=color,
            marker="o",
            ms=float(_plot_opt(panel, "marker_size", STYLE["marker_size"])),
            mec=color,
            mfc=color,
            mew=0.0,
            ls="None",
            alpha=alpha,
            label=dot_label,
            zorder=zorder + 1,
        )[0]
        if handle is None:
            handle = dots
    return handle


def _legend(ax, panel: str) -> None:
    loc = _plot_opt(panel, "legend_loc", None)
    if loc in [None, "", False]:
        return
    bbox = _plot_opt(panel, "legend_bbox_to_anchor", None)
    kwargs = {}
    if bbox not in [None, "", False]:
        kwargs["bbox_to_anchor"] = bbox
    ax.legend(
        loc=loc,
        frameon=bool(_plot_opt(panel, "legend_frameon", False)),
        fontsize=STYLE["legend_size"],
        ncol=int(_plot_opt(panel, "legend_ncol", 1)),
        columnspacing=float(_plot_opt(panel, "legend_columnspacing", 1.0)),
        handlelength=float(_plot_opt(panel, "legend_handlelength", 1.8)),
        **kwargs,
    )


def _legend_allowed(ax, panel: str, allowed_labels: list[str]) -> None:
    loc = _plot_opt(panel, "legend_loc", None)
    if loc in [None, "", False]:
        return
    handles, labels = ax.get_legend_handles_labels()
    keep_handles = []
    keep_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label not in allowed_labels or label in seen:
            continue
        keep_handles.append(handle)
        keep_labels.append(label)
        seen.add(label)
    if not keep_handles:
        return
    bbox = _plot_opt(panel, "legend_bbox_to_anchor", None)
    kwargs = {}
    if bbox not in [None, "", False]:
        kwargs["bbox_to_anchor"] = bbox
    ax.legend(
        keep_handles,
        keep_labels,
        loc=loc,
        frameon=bool(_plot_opt(panel, "legend_frameon", False)),
        fontsize=STYLE["legend_size"],
        ncol=int(_plot_opt(panel, "legend_ncol", 1)),
        columnspacing=float(_plot_opt(panel, "legend_columnspacing", 1.0)),
        handlelength=float(_plot_opt(panel, "legend_handlelength", 1.8)),
        **kwargs,
    )


def _apply_axis_options(ax, panel: str) -> None:
    ax.set_xscale(str(_plot_opt(panel, "xscale", ax.get_xscale())))
    ax.set_yscale(str(_plot_opt(panel, "yscale", ax.get_yscale())))
    xlim = _plot_opt(panel, "xlim", None)
    ylim = _plot_opt(panel, "ylim", None)
    if xlim not in [None, "", False]:
        ax.set_xlim(*xlim)
    if ylim not in [None, "", False]:
        ax.set_ylim(*ylim)


def _plot_ccdf(ax, values: pd.Series, *, label: str, color: str, panel: str, lw: float | None = None) -> pd.DataFrame:
    d = _ccdf(values)
    if not d.empty:
        _draw_xy(ax, d["x"], d["ccdf"], color=color, lw=lw, label=label, panel=panel)
    return d


def _plot_exp_reference(ax, x_max: float, mean_value: float, *, label: str, panel: str, color: str) -> pd.DataFrame:
    if not np.isfinite(x_max) or not np.isfinite(mean_value) or mean_value <= 0:
        return pd.DataFrame({"x": [], "ccdf": []})
    x = np.geomspace(max(1e-3, x_max / 1000.0), x_max, 240)
    y = np.exp(-x / mean_value)
    _draw_xy(
        ax,
        x,
        y,
        color=color,
        lw=STYLE["line_width"] - 0.4,
        ls=str(_plot_opt(panel, "reference_line_style", "--")),
        label=label,
        panel=panel,
        draw_key="reference_draw",
    )
    return pd.DataFrame({"x": x, "ccdf": y})


def _set_ccdf_empirical_ylim(ax, *ccdfs: pd.DataFrame) -> None:
    values = []
    for d in ccdfs:
        if d is not None and not d.empty and "ccdf" in d:
            arr = pd.to_numeric(d["ccdf"], errors="coerce").to_numpy(float)
            values.extend(arr[np.isfinite(arr) & (arr > 0)].tolist())
    if not values:
        return
    y_min = max(1.0e-4, min(values) * 0.45)
    ax.set_ylim(y_min, 1.25)


def _draw_power_law_segment(
    ax,
    *,
    alpha: float,
    intercept_log: float,
    x0: float,
    x1: float,
    y_scale: float,
    text_x_frac: float | None,
    text_dx_points: float = 0.0,
    text_dy_points: float = 0.0,
    text_ha: str = "center",
    text_va: str = "bottom",
    color: str,
    label: str | None = None,
) -> None:
    if not np.isfinite(alpha) or not np.isfinite(intercept_log):
        return
    xx = np.geomspace(x0, x1, 40)
    yy = np.exp(float(intercept_log)) * xx ** float(alpha) * y_scale
    ax.plot(xx, yy, color=color, lw=STYLE["fit_line_width"] - 0.4, ls="-", solid_capstyle="round", zorder=5, label=label)
    if text_x_frac is None:
        return
    text_idx = min(len(xx) - 1, max(0, int(round(text_x_frac * (len(xx) - 1)))))
    ax.annotate(
        rf"$\tau^{{{float(alpha):.2f}}}$",
        xy=(xx[text_idx], yy[text_idx]),
        xytext=(text_dx_points, text_dy_points),
        textcoords="offset points",
        color=color,
        fontsize=STYLE["legend_size"],
        ha=text_ha,
        va=text_va,
        zorder=8,
    )


def _plot_scaling_guide(ax, row: pd.Series, data: pd.DataFrame, *, color: str, track: str) -> None:
    if not np.isfinite(row.get("alpha", np.nan)) or not np.isfinite(row.get("intercept_log", np.nan)):
        return
    early_specs = {
        "player_abs": (1.05, 2.75, 0.88, 0.05, 5.0, 28.5, "left", "center"),
        "centroid": (1.15, 3.05, 0.82, None, 0.0, 0.0, "center", "bottom"),
        "player_rel": (1.25, 3.35, 0.78, 0.45, 4.0, -7.0, "left", "center"),
    }
    early = data.loc[data["tau_s"].between(*EARLY_LAG_WINDOW_S)].copy()
    early_alpha = np.nan
    if len(early) >= 3 and track in early_specs:
        early_fit = _weighted_log_fit(
            early["tau_s"].to_numpy(float),
            early["msd_m2"].to_numpy(float),
            early["n_pairs"].to_numpy(float),
        )
        early_alpha = float(early_fit["alpha"])
        _draw_power_law_segment(
            ax,
            alpha=early_alpha,
            intercept_log=float(early_fit["intercept"]),
            x0=early_specs[track][0],
            x1=early_specs[track][1],
            y_scale=early_specs[track][2],
            text_x_frac=early_specs[track][3],
            text_dx_points=early_specs[track][4],
            text_dy_points=early_specs[track][5],
            text_ha=early_specs[track][6],
            text_va=early_specs[track][7],
            color=color,
        )
    late_specs = {
        "player_abs": (4.2, 15.0, 1.30, 0.43, 0.0, 5.9, "center", "center"),
        "centroid": (6.5, 19.0, 0.70, None, 0.0, 0.0, "center", "bottom"),
        "player_rel": (8.0, 24.0, 0.68, 0.52, 0.0, -5.0, "center", "top"),
    }
    if track in late_specs:
        late_alpha = float(row["alpha"])
        _draw_power_law_segment(
            ax,
            alpha=late_alpha,
            intercept_log=float(row["intercept_log"]),
            x0=late_specs[track][0],
            x1=late_specs[track][1],
            y_scale=late_specs[track][2],
            text_x_frac=late_specs[track][3],
            text_dx_points=late_specs[track][4],
            text_dy_points=late_specs[track][5],
            text_ha=late_specs[track][6],
            text_va=late_specs[track][7],
            color=color,
            label=None,
        )


def _cluster_key(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in CLUSTER_COLS if c not in df.columns]
    if missing:
        return pd.Series(["all_data"] * len(df), index=df.index)
    return df[CLUSTER_COLS].astype(str).agg("__".join, axis=1)


def _ccdf_on_grid(values: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float), np.zeros_like(grid, dtype=int)
    x.sort()
    idx = np.searchsorted(x, grid, side="left")
    at_risk = len(x) - idx
    return at_risk.astype(float) / len(x), at_risk.astype(int)


def _survivor_audit(
    runs: pd.DataFrame,
    *,
    value_col: str,
    grid_col: str,
    centroid_col_prefix: str,
    player_col_prefix: str,
    null_col: str,
    B: int = BOOTSTRAP_B,
) -> pd.DataFrame:
    centroid = runs.loc[runs["track_type"].astype(str).eq("centroid")].copy()
    player = runs.loc[runs["track_type"].astype(str).eq("player_abs")].copy()
    centroid[value_col] = pd.to_numeric(centroid[value_col], errors="coerce")
    player[value_col] = pd.to_numeric(player[value_col], errors="coerce")
    all_values = pd.concat([centroid[value_col], player[value_col]], ignore_index=True).dropna()
    all_values = all_values.loc[all_values > 0]
    if all_values.empty:
        return pd.DataFrame()
    grid = np.geomspace(float(all_values.min()), float(all_values.max()), 180)
    centroid_values = centroid[value_col].dropna().to_numpy(float)
    player_values = player[value_col].dropna().to_numpy(float)
    centroid_ccdf, centroid_risk = _ccdf_on_grid(centroid_values, grid)
    player_ccdf, player_risk = _ccdf_on_grid(player_values, grid)
    mean_centroid = float(np.nanmean(centroid_values))
    null_ccdf = np.exp(-grid / mean_centroid) if mean_centroid > 0 else np.full_like(grid, np.nan)

    runs_boot = runs.loc[runs["track_type"].astype(str).isin(["centroid", "player_abs"])].copy()
    runs_boot["_cluster"] = _cluster_key(runs_boot)
    clusters = sorted(runs_boot["_cluster"].dropna().unique().tolist())
    by_cluster = {}
    for cluster, g in runs_boot.groupby("_cluster", sort=False):
        by_cluster[cluster] = {
            "centroid": pd.to_numeric(g.loc[g["track_type"].astype(str).eq("centroid"), value_col], errors="coerce").dropna().to_numpy(float),
            "player_abs": pd.to_numeric(g.loc[g["track_type"].astype(str).eq("player_abs"), value_col], errors="coerce").dropna().to_numpy(float),
        }
    boot_centroid = np.full((B, len(grid)), np.nan)
    boot_player = np.full((B, len(grid)), np.nan)
    for b in range(B):
        sample = RNG.choice(clusters, size=len(clusters), replace=True)
        c_vals = [by_cluster[c]["centroid"] for c in sample if len(by_cluster[c]["centroid"])]
        p_vals = [by_cluster[c]["player_abs"] for c in sample if len(by_cluster[c]["player_abs"])]
        if c_vals:
            boot_centroid[b], _ = _ccdf_on_grid(np.concatenate(c_vals), grid)
        if p_vals:
            boot_player[b], _ = _ccdf_on_grid(np.concatenate(p_vals), grid)
    c_lo, c_hi = np.nanpercentile(boot_centroid, [2.5, 97.5], axis=0)
    p_lo, p_hi = np.nanpercentile(boot_player, [2.5, 97.5], axis=0)
    out = pd.DataFrame(
        {
            grid_col: grid,
            f"{centroid_col_prefix}_ccdf": centroid_ccdf,
            f"{centroid_col_prefix}_ci_low": c_lo,
            f"{centroid_col_prefix}_ci_high": c_hi,
            f"{player_col_prefix}_ccdf": player_ccdf,
            f"{player_col_prefix}_ci_low": p_lo,
            f"{player_col_prefix}_ci_high": p_hi,
            null_col: null_ccdf,
            "n_centroid_at_risk": centroid_risk,
            "n_player_at_risk": player_risk,
            "reliable_support_flag": (centroid_risk >= 20) & (player_risk >= 20),
            "n_bootstrap": B,
            "bootstrap_unit": BOOTSTRAP_UNIT,
        }
    )
    return out


def _fill_band(ax, x, lo, hi, *, color: str, alpha: float = 0.13) -> None:
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    if np.any(ok):
        ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=alpha, lw=0, zorder=1)


def _aggregate_msd(msd: pd.DataFrame) -> pd.DataFrame:
    d = msd.dropna(subset=["track_type", "k", "tau_s", "msd_m2", "n_pairs"]).copy()
    d["weighted_msd"] = d["msd_m2"].astype(float) * d["n_pairs"].astype(float)
    g = (
        d.groupby(["track_type", "k"], observed=False)
        .agg(
            tau_s=("tau_s", "mean"),
            weighted_msd=("weighted_msd", "sum"),
            n_pairs=("n_pairs", "sum"),
            n_entities=("track_entity_uid", "nunique"),
        )
        .reset_index()
    )
    g["msd_m2"] = g["weighted_msd"] / g["n_pairs"].clip(lower=1)
    return g.sort_values(["track_type", "tau_s"]).reset_index(drop=True)


def _aggregate_msd_by_cluster(msd: pd.DataFrame) -> pd.DataFrame:
    d = msd.dropna(subset=["track_type", "k", "tau_s", "msd_m2", "n_pairs"]).copy()
    d["_cluster"] = _cluster_key(d)
    d["weighted_msd"] = d["msd_m2"].astype(float) * d["n_pairs"].astype(float)
    g = (
        d.groupby(["_cluster", "track_type", "k"], observed=False)
        .agg(tau_s=("tau_s", "mean"), weighted_msd=("weighted_msd", "sum"), n_pairs=("n_pairs", "sum"))
        .reset_index()
    )
    return g


def _msd_bootstrap_summary(msd: pd.DataFrame, agg: pd.DataFrame, B: int = BOOTSTRAP_B) -> pd.DataFrame:
    cluster_agg = _aggregate_msd_by_cluster(msd)
    clusters = sorted(cluster_agg["_cluster"].dropna().unique().tolist())
    if not clusters:
        out = agg.copy()
        out["ci_low"] = np.nan
        out["ci_high"] = np.nan
        out["n_bootstrap"] = 0
        out["bootstrap_unit"] = "unavailable"
        return out
    code = {c: i for i, c in enumerate(clusters)}
    cluster_agg["_cluster_code"] = cluster_agg["_cluster"].map(code).astype(int)
    keys = agg[["track_type", "k", "tau_s", "msd_m2"]].copy()
    boot_records = []
    for b in range(B):
        counts = np.bincount(RNG.choice(len(clusters), size=len(clusters), replace=True), minlength=len(clusters)).astype(float)
        d = cluster_agg.copy()
        d["_w"] = counts[d["_cluster_code"].to_numpy(int)]
        d = d.loc[d["_w"] > 0].copy()
        d["weighted_msd_b"] = d["weighted_msd"] * d["_w"]
        d["n_pairs_b"] = d["n_pairs"] * d["_w"]
        g = (
            d.groupby(["track_type", "k"], observed=False)
            .agg(weighted_msd=("weighted_msd_b", "sum"), n_pairs=("n_pairs_b", "sum"))
            .reset_index()
        )
        g["msd_boot"] = g["weighted_msd"] / g["n_pairs"].clip(lower=1)
        g["bootstrap_id"] = b
        boot_records.append(g[["bootstrap_id", "track_type", "k", "msd_boot"]])
    boot = pd.concat(boot_records, ignore_index=True)
    ci = (
        boot.groupby(["track_type", "k"], observed=False)["msd_boot"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_low", 0.975: "ci_high"})
    )
    out = keys.merge(ci, on=["track_type", "k"], how="left")
    out["n_bootstrap"] = B
    out["bootstrap_unit"] = BOOTSTRAP_UNIT
    return out.sort_values(["track_type", "tau_s"]).reset_index(drop=True)


def _local_slopes_for_track(tau: np.ndarray, msd: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    msd = np.asarray(msd, dtype=float)
    ok = np.isfinite(tau) & np.isfinite(msd) & (tau > 0) & (msd > 0)
    out = np.full_like(tau, np.nan, dtype=float)
    if np.sum(ok) < 3:
        return out
    lx = np.log(tau[ok])
    ly = np.log(msd[ok])
    out[np.where(ok)[0]] = np.gradient(ly, lx)
    return out


def _local_slope_audit(msd_boot: pd.DataFrame) -> pd.DataFrame:
    tracks = ["centroid", "player_abs", "player_rel"]
    base = None
    for track in tracks:
        sub = msd_boot.loc[msd_boot["track_type"].eq(track)].sort_values("tau_s").copy()
        sub[f"alpha_local_{track}"] = _local_slopes_for_track(sub["tau_s"].to_numpy(float), sub["msd_m2"].to_numpy(float))
        cols = ["tau_s", "msd_m2", "ci_low", "ci_high", f"alpha_local_{track}"]
        rename = {
            "tau_s": "lag_s",
            "msd_m2": f"msd_{track}",
            "ci_low": f"msd_{track}_ci_low",
            "ci_high": f"msd_{track}_ci_high",
        }
        cur = sub[cols].rename(columns=rename)
        if base is None:
            base = cur
        else:
            base = base.merge(cur, on="lag_s", how="outer")
    if base is None:
        return pd.DataFrame()
    base = base.sort_values("lag_s").reset_index(drop=True)
    base = base.rename(
        columns={
            "msd_player_abs": "msd_player_abs",
            "msd_player_abs_ci_low": "msd_player_abs_ci_low",
            "msd_player_abs_ci_high": "msd_player_abs_ci_high",
            "msd_player_rel": "msd_player_rel",
            "msd_player_rel_ci_low": "msd_player_rel_ci_low",
            "msd_player_rel_ci_high": "msd_player_rel_ci_high",
        }
    )
    base["fit_window_flag"] = base["lag_s"].between(*TACTICAL_LAG_WINDOW_S)
    base["early_window_flag"] = base["lag_s"].between(*EARLY_LAG_WINDOW_S)
    base["n_bootstrap"] = BOOTSTRAP_B
    base["bootstrap_unit"] = BOOTSTRAP_UNIT
    return base


def _weighted_log_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (x > 0) & (y > 0) & (w > 0)
    x = np.log(x[ok])
    y = np.log(y[ok])
    w = w[ok].astype(float)
    if len(x) < 3:
        return {"alpha": np.nan, "alpha_se": np.nan, "intercept": np.nan, "n_points": int(len(x)), "r2": np.nan}
    X = np.column_stack([np.ones_like(x), x])
    sw = np.sqrt(w / np.nanmax(w))
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    dof = max(len(y) - 2, 1)
    sigma2 = float(np.sum((sw * resid) ** 2) / dof)
    cov = sigma2 * np.linalg.pinv(Xw.T @ Xw)
    alpha_se = math.sqrt(max(float(cov[1, 1]), 0.0))
    ybar = float(np.average(y, weights=w))
    ss_res = float(np.sum(w * (y - yhat) ** 2))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "alpha": float(beta[1]),
        "alpha_se": alpha_se,
        "intercept": float(beta[0]),
        "n_points": int(len(x)),
        "r2": float(r2),
    }


def _msd_scaling_audit(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    left, right = TACTICAL_LAG_WINDOW_S
    for track in ["centroid", "player_abs", "player_rel"]:
        sub = agg.loc[agg["track_type"].eq(track) & agg["tau_s"].between(left, right)].copy()
        fit = _weighted_log_fit(sub["tau_s"].to_numpy(float), sub["msd_m2"].to_numpy(float), sub["n_pairs"].to_numpy(float))
        alpha = fit["alpha"]
        se = fit["alpha_se"]
        rows.append(
            {
                "track_type": track,
                "fit_tau_left_s": left,
                "fit_tau_right_s": right,
                "alpha": alpha,
                "alpha_se": se,
                "alpha_ci_low_approx": alpha - 1.96 * se if np.isfinite(alpha) and np.isfinite(se) else np.nan,
                "alpha_ci_high_approx": alpha + 1.96 * se if np.isfinite(alpha) and np.isfinite(se) else np.nan,
                "n_fit_points": fit["n_points"],
                "weighted_log_r2": fit["r2"],
                "intercept_log": fit["intercept"],
                "superdiffusive_supported": bool(np.isfinite(alpha) and np.isfinite(se) and (alpha - 1.96 * se > 1.0)),
                "uncertainty_type": "weighted log-log regression standard error, not block bootstrap",
            }
        )
    return pd.DataFrame(rows)


def _msd_alpha_bootstrap_summary(msd: pd.DataFrame, B: int = BOOTSTRAP_B) -> pd.DataFrame:
    cluster_agg = _aggregate_msd_by_cluster(msd)
    clusters = sorted(cluster_agg["_cluster"].dropna().unique().tolist())
    if not clusters:
        return pd.DataFrame()
    code = {c: i for i, c in enumerate(clusters)}
    cluster_agg["_cluster_code"] = cluster_agg["_cluster"].map(code).astype(int)
    rows = []
    for b in range(B):
        counts = np.bincount(RNG.choice(len(clusters), size=len(clusters), replace=True), minlength=len(clusters)).astype(float)
        d = cluster_agg.copy()
        d["_w"] = counts[d["_cluster_code"].to_numpy(int)]
        d = d.loc[d["_w"] > 0].copy()
        d["weighted_msd_b"] = d["weighted_msd"] * d["_w"]
        d["n_pairs_b"] = d["n_pairs"] * d["_w"]
        g = (
            d.groupby(["track_type", "k"], observed=False)
            .agg(tau_s=("tau_s", "mean"), weighted_msd=("weighted_msd_b", "sum"), n_pairs=("n_pairs_b", "sum"))
            .reset_index()
        )
        g["msd_m2"] = g["weighted_msd"] / g["n_pairs"].clip(lower=1)
        for track in ["centroid", "player_abs", "player_rel"]:
            sub = g.loc[g["track_type"].eq(track) & g["tau_s"].between(*TACTICAL_LAG_WINDOW_S)].copy()
            fit = _weighted_log_fit(sub["tau_s"].to_numpy(float), sub["msd_m2"].to_numpy(float), sub["n_pairs"].to_numpy(float))
            rows.append({"bootstrap_id": b, "track_type": track, "alpha": fit["alpha"]})
    boot = pd.DataFrame(rows)
    if boot.empty:
        return pd.DataFrame()
    summary = (
        boot.groupby("track_type", observed=False)["alpha"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "alpha_boot_ci_low", 0.5: "alpha_boot_median", 0.975: "alpha_boot_ci_high"})
    )
    summary["n_bootstrap"] = B
    summary["bootstrap_unit"] = BOOTSTRAP_UNIT
    return summary


def _trajectory_with_centroid(traj: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["match_id", "match_phase", "team", "source_key", "t"]
    centroid = (
        traj.loc[traj["track_type"].astype(str).eq("centroid"), key_cols + ["x_m", "y_m"]]
        .rename(columns={"x_m": "R_x", "y_m": "R_y"})
        .drop_duplicates(key_cols)
    )
    players = traj.loc[
        traj["track_type"].astype(str).eq("player_abs"),
        key_cols + ["track_uid", "track_entity_uid", "player_name", "x_m", "y_m"],
    ].copy()
    d = players.merge(centroid, on=key_cols, how="inner", validate="many_to_one")
    d["r_x"] = d["x_m"] - d["R_x"]
    d["r_y"] = d["y_m"] - d["R_y"]
    d["t_ns"] = pd.to_datetime(d["t"]).astype("int64")
    return d.sort_values(["track_uid", "t_ns"]).reset_index(drop=True)


def _compute_msd_decomposition_components(traj: pd.DataFrame, lags_s: np.ndarray = DECOMPOSITION_LAGS_S) -> pd.DataFrame:
    d = _trajectory_with_centroid(traj)
    d["_cluster"] = _cluster_key(d)
    accum: dict[tuple[str, int], dict[str, float | int]] = {}
    one_sec_ns = 1_000_000_000
    cols = ["t_ns", "x_m", "y_m", "R_x", "R_y", "r_x", "r_y"]
    for (cluster, _track_uid), g in d.groupby(["_cluster", "track_uid"], sort=False):
        arr = g[cols].to_numpy(dtype=float)
        if len(arr) <= 1:
            continue
        t_ns = arr[:, 0].astype(np.int64)
        x = arr[:, 1]
        y = arr[:, 2]
        rx = arr[:, 3]
        ry = arr[:, 4]
        relx = arr[:, 5]
        rely = arr[:, 6]
        n = len(arr)
        for lag in lags_s:
            lag = int(lag)
            if n <= lag:
                continue
            ok = (t_ns[lag:] - t_ns[:-lag]) == lag * one_sec_ns
            if not np.any(ok):
                continue
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            dR_x = rx[lag:] - rx[:-lag]
            dR_y = ry[lag:] - ry[:-lag]
            dr_x = relx[lag:] - relx[:-lag]
            dr_y = rely[lag:] - rely[:-lag]
            dx = dx[ok]
            dy = dy[ok]
            dR_x = dR_x[ok]
            dR_y = dR_y[ok]
            dr_x = dr_x[ok]
            dr_y = dr_y[ok]
            key = (str(cluster), lag)
            rec = accum.setdefault(key, {"sum_player": 0.0, "sum_centroid": 0.0, "sum_relative": 0.0, "sum_cross": 0.0, "n_pairs": 0})
            rec["sum_player"] += float(np.sum(dx * dx + dy * dy))
            rec["sum_centroid"] += float(np.sum(dR_x * dR_x + dR_y * dR_y))
            rec["sum_relative"] += float(np.sum(dr_x * dr_x + dr_y * dr_y))
            rec["sum_cross"] += float(np.sum(2.0 * (dR_x * dr_x + dR_y * dr_y)))
            rec["n_pairs"] += int(len(dx))
    rows = []
    for (cluster, lag), rec in accum.items():
        row = {"cluster": cluster, "lag_s": float(lag)}
        row.update(rec)
        rows.append(row)
    return pd.DataFrame(rows)


def _decomposition_from_components(components: pd.DataFrame) -> pd.DataFrame:
    if components.empty:
        return pd.DataFrame()
    g = (
        components.groupby("lag_s", observed=False)
        .agg(
            sum_player=("sum_player", "sum"),
            sum_centroid=("sum_centroid", "sum"),
            sum_relative=("sum_relative", "sum"),
            sum_cross=("sum_cross", "sum"),
            n_pairs=("n_pairs", "sum"),
        )
        .reset_index()
        .sort_values("lag_s")
    )
    rows = []
    for r in g.itertuples():
        n = int(r.n_pairs)
        if n <= 0:
            continue
        msd_player = float(r.sum_player) / n
        msd_centroid = float(r.sum_centroid) / n
        msd_relative = float(r.sum_relative) / n
        cross_term = float(r.sum_cross) / n
        denom = msd_player if msd_player != 0 else np.nan
        rows.append(
            {
                "lag_s": float(r.lag_s),
                "n_pairs": int(n),
                "msd_player": msd_player,
                "msd_centroid": msd_centroid,
                "msd_relative": msd_relative,
                "cross_term": cross_term,
                "f_centroid": msd_centroid / denom,
                "f_relative": msd_relative / denom,
                "f_cross": cross_term / denom,
                "f_sum": (msd_centroid + msd_relative + cross_term) / denom,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_decomposition(components: pd.DataFrame, decomp: pd.DataFrame, B: int = BOOTSTRAP_B) -> pd.DataFrame:
    if components.empty or decomp.empty or "cluster" not in components.columns:
        out = decomp.copy()
        for col in ["f_centroid", "f_relative", "f_cross"]:
            out[f"{col}_ci_low"] = np.nan
            out[f"{col}_ci_high"] = np.nan
        out["n_bootstrap"] = 0
        out["bootstrap_unit"] = "unavailable"
        return out
    clusters = sorted(components["cluster"].dropna().unique().tolist())
    records = []
    for b in range(B):
        sample = pd.Series(RNG.choice(clusters, size=len(clusters), replace=True)).value_counts()
        d = components.merge(sample.rename("_w"), left_on="cluster", right_index=True, how="inner")
        for c in ["sum_player", "sum_centroid", "sum_relative", "sum_cross", "n_pairs"]:
            d[c] = d[c] * d["_w"]
        boot_decomp = _decomposition_from_components(d)
        if boot_decomp.empty:
            continue
        boot_decomp["bootstrap_id"] = b
        records.append(boot_decomp[["bootstrap_id", "lag_s", "f_centroid", "f_relative", "f_cross"]])
    out = decomp.copy()
    if records:
        boot = pd.concat(records, ignore_index=True)
        for col in ["f_centroid", "f_relative", "f_cross"]:
            ci = (
                boot.groupby("lag_s", observed=False)[col]
                .quantile([0.025, 0.975])
                .unstack()
                .reset_index()
                .rename(columns={0.025: f"{col}_ci_low", 0.975: f"{col}_ci_high"})
            )
            out = out.merge(ci, on="lag_s", how="left")
    for col in ["f_centroid", "f_relative", "f_cross"]:
        if f"{col}_ci_low" not in out:
            out[f"{col}_ci_low"] = np.nan
        if f"{col}_ci_high" not in out:
            out[f"{col}_ci_high"] = np.nan
    out["n_bootstrap"] = B if records else 0
    out["bootstrap_unit"] = BOOTSTRAP_UNIT if records else "unavailable"
    return out


def _compute_msd_decomposition(traj: pd.DataFrame, lags_s: np.ndarray = DECOMPOSITION_LAGS_S, B: int = BOOTSTRAP_B) -> tuple[pd.DataFrame, pd.DataFrame]:
    components = _compute_msd_decomposition_components(traj, lags_s=lags_s)
    decomp = _decomposition_from_components(components)
    decomp = _bootstrap_decomposition(components, decomp, B=B)
    return decomp, components


def prepare_figure2_context(
    style: dict | None = None,
    *,
    write_audits: bool = True,
    inputs: dict[str, pd.DataFrame] | None = None,
) -> dict:
    if style is not None:
        apply_project_style(style)
    inputs = _load_inputs() if inputs is None else {k: v.copy() for k, v in inputs.items()}
    missing = {"runs", "msd", "trajectory"} - set(inputs)
    if missing:
        raise KeyError(f"Figure 2 inputs missing required keys: {sorted(missing)}")
    runs = inputs["runs"].copy()
    msd_agg = _aggregate_msd(inputs["msd"])
    msd_boot = _msd_bootstrap_summary(inputs["msd"], msd_agg, B=BOOTSTRAP_B)
    scaling_audit = _msd_scaling_audit(msd_agg)
    alpha_boot = _msd_alpha_bootstrap_summary(inputs["msd"], B=BOOTSTRAP_B)
    if not alpha_boot.empty:
        scaling_audit = scaling_audit.merge(alpha_boot, on="track_type", how="left", suffixes=("", "_boot"))
        scaling_audit["alpha_ci_low"] = scaling_audit["alpha_boot_ci_low"]
        scaling_audit["alpha_ci_high"] = scaling_audit["alpha_boot_ci_high"]
        scaling_audit["uncertainty_type"] = f"{BOOTSTRAP_UNIT} bootstrap percentile interval"
        scaling_audit["superdiffusive_supported"] = scaling_audit["alpha_ci_low"].astype(float) > 1.0
    else:
        scaling_audit["alpha_ci_low"] = scaling_audit["alpha_ci_low_approx"]
        scaling_audit["alpha_ci_high"] = scaling_audit["alpha_ci_high_approx"]
    local_slope_audit = _local_slope_audit(msd_boot)
    decomp, decomp_components = _compute_msd_decomposition(inputs["trajectory"], B=BOOTSTRAP_B)

    centroid_runs = runs.loc[runs["track_type"].astype(str).eq("centroid")].copy()
    player_runs = runs.loc[runs["track_type"].astype(str).eq("player_abs")].copy()
    duration_centroid = _ccdf(centroid_runs["duration_s"])
    duration_player = _ccdf(player_runs["duration_s"])
    length_centroid = _ccdf(centroid_runs["run_length_m"])
    length_player = _ccdf(player_runs["run_length_m"])
    duration_audit = _survivor_audit(
        runs,
        value_col="duration_s",
        grid_col="duration_t",
        centroid_col_prefix="centroid",
        player_col_prefix="player",
        null_col="null_ccdf",
        B=BOOTSTRAP_B,
    )
    length_audit = _survivor_audit(
        runs,
        value_col="run_length_m",
        grid_col="length_l",
        centroid_col_prefix="centroid",
        player_col_prefix="player",
        null_col="null_ccdf",
        B=BOOTSTRAP_B,
    )

    context = {
        "inputs": inputs,
        "runs": runs,
        "centroid_runs": centroid_runs,
        "player_runs": player_runs,
        "msd_agg": msd_agg,
        "msd_boot": msd_boot,
        "scaling_audit": scaling_audit,
        "local_slope_audit": local_slope_audit,
        "decomp": decomp,
        "decomp_components": decomp_components,
        "duration_centroid": duration_centroid,
        "duration_player": duration_player,
        "length_centroid": length_centroid,
        "length_player": length_player,
        "duration_audit": duration_audit,
        "length_audit": length_audit,
    }
    if write_audits:
        duration_audit.to_csv(AUDIT_DIR / "figure2_panelA_duration_survivor_audit.csv", index=False)
        length_audit.to_csv(AUDIT_DIR / "figure2_panelB_length_survivor_audit.csv", index=False)
        msd_agg.to_csv(AUDIT_DIR / "figure2_panelC_msd_aggregate_audit.csv", index=False)
        msd_boot.to_csv(AUDIT_DIR / "figure2_panelC_msd_bootstrap_audit.csv", index=False)
        scaling_audit.to_csv(AUDIT_DIR / "figure2_panelC_msd_scaling_audit.csv", index=False)
        local_slope_audit.to_csv(AUDIT_DIR / "figure2_panelC_local_slope_audit.csv", index=False)
        decomp.to_csv(AUDIT_DIR / "figure2_panelD_msd_decomposition_audit.csv", index=False)
        decomp_components.to_csv(AUDIT_DIR / "figure2_panelD_msd_decomposition_components.csv", index=False)
    return context


def _plot_panel_a(ax, context: dict, *, show_uncertainty: bool = False) -> None:
    centroid = context["centroid_runs"]
    player = context["player_runs"]
    audit = context.get("duration_audit", pd.DataFrame())
    if show_uncertainty and not audit.empty:
        _fill_band(ax, audit["duration_t"], audit["centroid_ci_low"], audit["centroid_ci_high"], color=FIGURE2_COLORS["centroid"], alpha=0.10)
        _fill_band(ax, audit["duration_t"], audit["player_ci_low"], audit["player_ci_high"], color=FIGURE2_COLORS["player"], alpha=0.12)
    centroid_ccdf = _plot_ccdf(ax, centroid["duration_s"], label=_text("centroid_label"), color=FIGURE2_COLORS["centroid"], panel="panel_a")
    player_ccdf = _plot_ccdf(ax, player["duration_s"], label=_text("player_label"), color=FIGURE2_COLORS["player"], panel="panel_a")
    if bool(_plot_opt("panel_a", "show_exp_reference", True)):
        centroid_values = pd.to_numeric(centroid["duration_s"], errors="coerce")
        player_values = pd.to_numeric(player["duration_s"], errors="coerce")
        x_max = float(np.nanmax(pd.concat([centroid_values, player_values], ignore_index=True)))
        _plot_exp_reference(
            ax,
            x_max,
            float(centroid_values.mean()),
            label=_text("exp_centroid_label"),
            panel="panel_a",
            color=FIGURE2_COLORS["centroid"],
        )
        _plot_exp_reference(
            ax,
            x_max,
            float(player_values.mean()),
            label=_text("exp_player_label"),
            panel="panel_a",
            color=FIGURE2_COLORS["player"],
        )
    ax.set(xscale="log", yscale="log", xlabel=_text("panel_a_xlabel"), ylabel=_text("panel_a_ylabel"))
    _apply_axis_options(ax, "panel_a")
    _set_ccdf_empirical_ylim(ax, centroid_ccdf, player_ccdf)
    _legend(ax, "panel_a")
    _style_ax(ax)


def _plot_panel_b(ax, context: dict, *, show_uncertainty: bool = False) -> None:
    centroid = context["centroid_runs"]
    player = context["player_runs"]
    audit = context.get("length_audit", pd.DataFrame())
    if show_uncertainty and not audit.empty:
        _fill_band(ax, audit["length_l"], audit["centroid_ci_low"], audit["centroid_ci_high"], color=FIGURE2_COLORS["centroid"], alpha=0.10)
        _fill_band(ax, audit["length_l"], audit["player_ci_low"], audit["player_ci_high"], color=FIGURE2_COLORS["player"], alpha=0.12)
    centroid_ccdf = _plot_ccdf(ax, centroid["run_length_m"], label=_text("centroid_label"), color=FIGURE2_COLORS["centroid"], panel="panel_b")
    player_ccdf = _plot_ccdf(ax, player["run_length_m"], label=_text("player_label"), color=FIGURE2_COLORS["player"], panel="panel_b")
    if bool(_plot_opt("panel_b", "show_exp_reference", True)):
        centroid_values = pd.to_numeric(centroid["run_length_m"], errors="coerce")
        player_values = pd.to_numeric(player["run_length_m"], errors="coerce")
        x_max = float(np.nanmax(pd.concat([centroid_values, player_values], ignore_index=True)))
        _plot_exp_reference(
            ax,
            x_max,
            float(centroid_values.mean()),
            label=_text("exp_centroid_label"),
            panel="panel_b",
            color=FIGURE2_COLORS["centroid"],
        )
        _plot_exp_reference(
            ax,
            x_max,
            float(player_values.mean()),
            label=_text("exp_player_label"),
            panel="panel_b",
            color=FIGURE2_COLORS["player"],
        )
    ax.set(xscale="log", yscale="log", xlabel=_text("panel_b_xlabel"), ylabel=_text("panel_b_ylabel"))
    _apply_axis_options(ax, "panel_b")
    _set_ccdf_empirical_ylim(ax, centroid_ccdf, player_ccdf)
    _legend(ax, "panel_b")
    _style_ax(ax)


def _plot_panel_c(ax, context: dict, *, show_uncertainty: bool = False) -> None:
    agg = context["msd_agg"]
    msd_boot = context.get("msd_boot", pd.DataFrame())
    scaling = context["scaling_audit"].set_index("track_type")
    colors = {"centroid": FIGURE2_COLORS["centroid"], "player_abs": FIGURE2_COLORS["player_abs"], "player_rel": FIGURE2_COLORS["player_rel"]}
    labels = {"centroid": _text("centroid_label"), "player_abs": _text("player_abs_label"), "player_rel": _text("player_rel_label")}
    left, right = TACTICAL_LAG_WINDOW_S
    if bool(_plot_opt("panel_c", "show_fit_window", True)):
        ax.axvspan(left, right, color=FIGURE2_COLORS["fit_window"], alpha=float(_plot_opt("panel_c", "fit_window_alpha", 1.0)), lw=0, zorder=0)
    for track in ["centroid", "player_abs", "player_rel"]:
        sub = agg.loc[agg["track_type"].eq(track)].sort_values("tau_s")
        if sub.empty:
            continue
        if show_uncertainty and not msd_boot.empty:
            b = msd_boot.loc[msd_boot["track_type"].eq(track)].sort_values("tau_s")
            _fill_band(ax, b["tau_s"], b["ci_low"], b["ci_high"], color=colors[track], alpha=0.11)
        _draw_xy(ax, sub["tau_s"], sub["msd_m2"], color=colors[track], lw=STYLE["line_width"], label=labels[track], panel="panel_c")
        if bool(_plot_opt("panel_c", "show_fit_lines", True)) and track in scaling.index:
            _plot_scaling_guide(ax, scaling.loc[track], sub, color=colors[track], track=track)
    ax.set(xscale="log", yscale="log", xlabel=_text("panel_c_xlabel"), ylabel=_text("panel_c_ylabel"))
    _apply_axis_options(ax, "panel_c")
    _legend_allowed(ax, "panel_c", [_text("centroid_label"), _text("player_abs_label"), _text("player_rel_label")])
    _style_ax(ax)


def _plot_panel_d(ax, context: dict, *, show_uncertainty: bool = False) -> None:
    d = context["decomp"]
    if bool(_plot_opt("panel_d", "show_reference_lines", True)):
        ax.axhline(0.0, color=FIGURE2_COLORS["zero_line"], lw=0.9, ls="-")
    if show_uncertainty:
        for col, color in [("f_centroid", FIGURE2_COLORS["centroid"]), ("f_relative", FIGURE2_COLORS["relative"]), ("f_cross", FIGURE2_COLORS["cross_term"])]:
            lo = f"{col}_ci_low"
            hi = f"{col}_ci_high"
            if lo in d and hi in d:
                _fill_band(ax, d["lag_s"], d[lo], d[hi], color=color, alpha=0.12)
    _draw_xy(ax, d["lag_s"], d["f_centroid"], color=FIGURE2_COLORS["centroid"], lw=STYLE["line_width"], label=_text("centroid_translation_label"), panel="panel_d")
    _draw_xy(ax, d["lag_s"], d["f_relative"], color=FIGURE2_COLORS["relative"], lw=STYLE["line_width"], label=_text("relative_label"), panel="panel_d")
    _draw_xy(ax, d["lag_s"], d["f_cross"], color=FIGURE2_COLORS["cross_term"], lw=STYLE["line_width"], label=_text("cross_term_label"), panel="panel_d")
    ax.set(xlabel=_text("panel_d_xlabel"), ylabel=_text("panel_d_ylabel"))
    y_min = float(np.nanmin(d[["f_centroid", "f_relative", "f_cross"]].to_numpy())) - 0.08
    y_max = float(np.nanmax(d[["f_centroid", "f_relative", "f_cross"]].to_numpy())) + 0.08
    ax.set_ylim(min(-0.15, y_min), max(1.05, y_max))
    _apply_axis_options(ax, "panel_d")
    _legend(ax, "panel_d")
    _style_ax(ax)


def plot_figure2_transport_phenotype(
    style: dict | None = None,
    context: dict | None = None,
    *,
    show_uncertainty: bool = False,
    aesthetics: dict | None = None,
) -> plt.Figure:
    if aesthetics is not None:
        apply_figure2_aesthetics(aesthetics)
    if context is None:
        context = prepare_figure2_context(style=style, write_audits=True)
    elif style is not None:
        apply_project_style(style)
    fig, axes = plt.subplots(2, 2, figsize=tuple(FIGURE2_PLOT.get("figure_size", (9.6, 7.1))))
    _plot_panel_a(axes[0, 0], context, show_uncertainty=show_uncertainty)
    _plot_panel_b(axes[0, 1], context, show_uncertainty=show_uncertainty)
    _plot_panel_c(axes[1, 0], context, show_uncertainty=show_uncertainty)
    _plot_panel_d(axes[1, 1], context, show_uncertainty=show_uncertainty)
    if draw_panel_letters is not None:
        draw_panel_letters(list(axes.flat), x=-0.12, y=1.03, fontsize=STYLE["title_size"])
    if bool(FIGURE2_PLOT.get("tight_layout", True)):
        fig.tight_layout()
    subplots_adjust = FIGURE2_PLOT.get("subplots_adjust", {})
    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)
    fig._figure2_context = context
    return fig


def _fit_window_decision(local_slope: pd.DataFrame, scaling_audit: pd.DataFrame) -> dict:
    out = {
        "fit_window_s": list(TACTICAL_LAG_WINDOW_S),
        "early_window_s": list(EARLY_LAG_WINDOW_S),
        "decision_code_only": "retain_5_30s_pre_saturation_window",
        "early_lag_exclusion_code_only": "early_local_slopes_are_audited_but_not_used_for_main_alpha_fit",
    }
    if local_slope.empty:
        out["local_slope_audit_available"] = False
        return out
    out["local_slope_audit_available"] = True
    for track in ["centroid", "player_abs", "player_rel"]:
        col = f"alpha_local_{track}"
        if col not in local_slope:
            continue
        early = local_slope.loc[local_slope["early_window_flag"], col].dropna()
        fit = local_slope.loc[local_slope["fit_window_flag"], col].dropna()
        out[f"{track}_early_alpha_local_median"] = float(early.median()) if len(early) else np.nan
        out[f"{track}_early_alpha_local_iqr"] = float(early.quantile(0.75) - early.quantile(0.25)) if len(early) else np.nan
        out[f"{track}_fit_alpha_local_median"] = float(fit.median()) if len(fit) else np.nan
        out[f"{track}_fit_alpha_local_iqr"] = float(fit.quantile(0.75) - fit.quantile(0.25)) if len(fit) else np.nan
    if not scaling_audit.empty:
        out["superdiffusion_supported_by_alpha_ci"] = bool(scaling_audit["superdiffusive_supported"].fillna(False).all())
    return out


def _write_uncertainty_notes(context: dict, created: list[str]) -> Path:
    text = "\n".join(
        [
            "# Figure 2 uncertainty notes",
            "",
            f"- Bootstrap unit: {BOOTSTRAP_UNIT}.",
            f"- n_bootstrap: {BOOTSTRAP_B}.",
            "- Panel A/B: bootstrap bands are shown in the uncertainty version and saved in audit CSVs.",
            "- Panel C: MSD bootstrap bands and local log-log slopes are shown/saved for the uncertainty version.",
            "- Panel D: decomposition bootstrap bands are shown in the uncertainty version and saved in the audit CSV.",
            "- Main figure keeps uncertainty bands off for CCDF/MSD readability; audit CSVs retain intervals.",
        ]
    )
    path = NOTES_DIR / "figure2_uncertainty_notes.md"
    path.write_text(text + "\n", encoding="utf-8")
    created.append(str(path))
    return path


def write_figure2_outputs_from_context(context: dict, fig: plt.Figure | None = None, *, save_main_figure: bool = True) -> dict:
    if fig is None:
        fig = plot_figure2_transport_phenotype(context=context)
    saved = []
    if save_main_figure:
        saved = save_figure(fig, MAIN_DIR, "figure2_transport_phenotype_v2")
        plt.close(fig)
    fig_unc = plot_figure2_transport_phenotype(context=context, show_uncertainty=True)
    uncertainty_saved = save_figure(fig_unc, UNCERTAINTY_DIR, "figure2_transport_phenotype_v2_uncertainty")
    plt.close(fig_unc)

    decomp = context["decomp"]
    max_closure_error = float(np.nanmax(np.abs(decomp["f_sum"].to_numpy(float) - 1.0))) if not decomp.empty else np.nan
    tactical = decomp.loc[decomp["lag_s"].between(*TACTICAL_LAG_WINDOW_S)].copy()
    mean_f_centroid = float(tactical["f_centroid"].mean()) if not tactical.empty else np.nan
    scaling_records = context["scaling_audit"].to_dict(orient="records")
    created = [str(p) for p in saved + uncertainty_saved]
    notes_path = _write_uncertainty_notes(context, created)
    fit_window_decision = _fit_window_decision(context["local_slope_audit"], context["scaling_audit"])
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_version": OUTPUT_VERSION,
        "style": {
            "repo_style_checked": True,
            "style_source": STYLE_HELPER_USED,
            "style": STYLE,
            "figure2_colors": FIGURE2_COLORS,
            "figure2_text": FIGURE2_TEXT,
            "figure2_plot": FIGURE2_PLOT,
            "order_colors_defined_for_consistency": ORDER_COLORS,
            "forbidden_legacy_palette_used": False,
        },
        "outputs": {
            "revision_folder": str(OUT_DIR),
            "main_figure_png": str(MAIN_DIR / "figure2_transport_phenotype_v2.png"),
            "main_figure_pdf": str(MAIN_DIR / "figure2_transport_phenotype_v2.pdf"),
            "uncertainty_figure_png": str(UNCERTAINTY_DIR / "figure2_transport_phenotype_v2_uncertainty.png"),
            "uncertainty_figure_pdf": str(UNCERTAINTY_DIR / "figure2_transport_phenotype_v2_uncertainty.pdf"),
            "manifest_json": str(MANIFEST_DIR / "figure2_transport_phenotype_v2_manifest.json"),
            "panelA_duration_survivor_audit_csv": str(AUDIT_DIR / "figure2_panelA_duration_survivor_audit.csv"),
            "panelB_length_survivor_audit_csv": str(AUDIT_DIR / "figure2_panelB_length_survivor_audit.csv"),
            "panelD_msd_decomposition_audit_csv": str(AUDIT_DIR / "figure2_panelD_msd_decomposition_audit.csv"),
            "panelC_msd_scaling_audit_csv": str(AUDIT_DIR / "figure2_panelC_msd_scaling_audit.csv"),
            "panelC_local_slope_audit_csv": str(AUDIT_DIR / "figure2_panelC_local_slope_audit.csv"),
            "uncertainty_notes_md": str(notes_path),
        },
        "required_inputs": {
            "runs_long": str(RUNS_PATH),
            "msd_long": str(MSD_PATH),
            "trajectory_long": str(TRAJECTORY_PATH),
        },
        "panel_A": {
            "quantity": "duration survivor P(T >= t)",
            "centroid_color": "black",
            "player_color": STYLE["red"],
            "reference_definition": "exponential CCDF exp(-t / mean_centroid_duration_s), matched to centroid mean duration",
            "centroid_n": int(len(context["centroid_runs"])),
            "player_n": int(len(context["player_runs"])),
            "uncertainty_visible_main": False,
            "uncertainty_visible_uncertainty_version": True,
            "uncertainty_method": f"{BOOTSTRAP_UNIT} bootstrap, B={BOOTSTRAP_B}",
        },
        "panel_B": {
            "quantity": "length survivor P(L >= l)",
            "centroid_color": "black",
            "player_color": STYLE["red"],
            "reference_definition": "exponential CCDF exp(-l / mean_centroid_length_m), matched to centroid mean run length",
            "uncertainty_visible_main": False,
            "uncertainty_visible_uncertainty_version": True,
            "uncertainty_method": f"{BOOTSTRAP_UNIT} bootstrap, B={BOOTSTRAP_B}",
        },
        "panel_C": {
            "quantity": "MSD versus lag",
            "fit_window_s": list(TACTICAL_LAG_WINDOW_S),
            "fit_window_decision": fit_window_decision,
            "alpha_fits": scaling_records,
            "superdiffusive_supported_rule": "alpha_ci_low > 1",
            "uncertainty_type": f"{BOOTSTRAP_UNIT} bootstrap percentile interval where available",
            "local_slope_audit_csv": str(AUDIT_DIR / "figure2_panelC_local_slope_audit.csv"),
            "uncertainty_visible_main": False,
            "uncertainty_visible_uncertainty_version": True,
        },
        "panel_D": {
            "formula": "MSD_player(tau)=MSD_centroid(tau)+MSD_relative(tau)+2<DeltaR(tau).Deltar_i(tau)>",
            "display": "signed lag-resolved fractional contributions",
            "f_centroid": "MSD_centroid / MSD_player",
            "f_relative": "MSD_relative / MSD_player",
            "f_cross": "cross_term / MSD_player",
            "decomposition_lag_grid_s": [int(x) for x in DECOMPOSITION_LAGS_S.tolist()],
            "tactical_lag_window_s": list(TACTICAL_LAG_WINDOW_S),
            "mean_f_centroid_tactical_window": mean_f_centroid,
            "max_abs_f_sum_minus_one": max_closure_error,
            "uncertainty_bootstrap_or_block_bootstrap": True,
            "bootstrap_unit": BOOTSTRAP_UNIT,
            "n_bootstrap": BOOTSTRAP_B,
            "uncertainty_visible_main": False,
            "uncertainty_visible_uncertainty_version": True,
            "old_bar_ratio_plot_removed": True,
        },
        "qa": {
            "all_declared_outputs_exist": {
                str(MAIN_DIR / "figure2_transport_phenotype_v2.png"): Path(MAIN_DIR / "figure2_transport_phenotype_v2.png").exists(),
                str(MAIN_DIR / "figure2_transport_phenotype_v2.pdf"): Path(MAIN_DIR / "figure2_transport_phenotype_v2.pdf").exists(),
                str(UNCERTAINTY_DIR / "figure2_transport_phenotype_v2_uncertainty.png"): Path(UNCERTAINTY_DIR / "figure2_transport_phenotype_v2_uncertainty.png").exists(),
                str(UNCERTAINTY_DIR / "figure2_transport_phenotype_v2_uncertainty.pdf"): Path(UNCERTAINTY_DIR / "figure2_transport_phenotype_v2_uncertainty.pdf").exists(),
                str(AUDIT_DIR / "figure2_panelC_local_slope_audit.csv"): Path(AUDIT_DIR / "figure2_panelC_local_slope_audit.csv").exists(),
                str(AUDIT_DIR / "figure2_panelD_msd_decomposition_audit.csv"): Path(AUDIT_DIR / "figure2_panelD_msd_decomposition_audit.csv").exists(),
            },
            "panel_D_fraction_closure_max_abs_error": max_closure_error,
            "panel_D_old_bar_plot_removed": True,
        },
        "created_files_this_run": created
        + [
            str(AUDIT_DIR / "figure2_panelA_duration_survivor_audit.csv"),
            str(AUDIT_DIR / "figure2_panelB_length_survivor_audit.csv"),
            str(AUDIT_DIR / "figure2_panelC_msd_aggregate_audit.csv"),
            str(AUDIT_DIR / "figure2_panelC_msd_bootstrap_audit.csv"),
            str(AUDIT_DIR / "figure2_panelC_msd_scaling_audit.csv"),
            str(AUDIT_DIR / "figure2_panelC_local_slope_audit.csv"),
            str(AUDIT_DIR / "figure2_panelD_msd_decomposition_audit.csv"),
            str(AUDIT_DIR / "figure2_panelD_msd_decomposition_components.csv"),
        ],
    }
    _save_json(manifest, MANIFEST_DIR / "figure2_transport_phenotype_v2_manifest.json")
    return {"manifest": manifest, "created": manifest["created_files_this_run"]}


def generate_figure2_outputs(style: dict | None = None) -> dict:
    context = prepare_figure2_context(style=style, write_audits=True)
    fig = plot_figure2_transport_phenotype(style=style, context=context, show_uncertainty=False)
    return write_figure2_outputs_from_context(context, fig=fig)


def main() -> None:
    result = generate_figure2_outputs()
    manifest = result["manifest"]
    print("STYLE_HELPER_USED", STYLE_HELPER_USED)
    print("OUTPUT_FOLDER", str(OUT_DIR))
    print("MAIN_FIGURE_SAVED", manifest["outputs"]["main_figure_png"], manifest["outputs"]["main_figure_pdf"])
    print("PANEL_D_MAX_ABS_F_SUM_MINUS_ONE", manifest["panel_D"]["max_abs_f_sum_minus_one"])
    print("PANEL_C_FIT_WINDOW_S", manifest["panel_C"]["fit_window_s"])
    print("MANIFEST_SAVED", manifest["outputs"]["manifest_json"])


if __name__ == "__main__":
    main()
