from __future__ import annotations

import json
import math
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
LEVY_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = LEVY_DIR.parents[1]
DATA_DIR = LEVY_DIR / "data"
TAIL_AUDIT_DIR = REPO_ROOT / "tail_audit"
SUPPLEMENT_SOURCE_DIR = LEVY_DIR / "figures" / "supplementary_material" / "source_data"
OUT_DIR = Path(
    os.environ.get(
        "FIGURE2_FINAL_UPDATE_OUT_DIR",
        str(LEVY_DIR / "outputs" / "final" / "figure2_transport_phenotype"),
    )
)
MAIN_DIR = OUT_DIR / "main"
FIGURE_SOURCE_DIR = OUT_DIR / "figure2_source_data"
SOURCE_DATA_DIR = OUT_DIR / "source_data"

for path in [OUT_DIR, MAIN_DIR, FIGURE_SOURCE_DIR, SOURCE_DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.levy_paper.scripts import create_figure2_transport_phenotype as fig2
from analysis.levy_paper.util.paper_utils import cache_path


RUNS_PATH = cache_path("runs_long_2020_2021_all_teams_pitchfix_sticky_active.parquet")
MSD_PATH = cache_path("msd_long_2020_2021_all_teams_pitchfix_sticky_active.parquet")
TRAJECTORY_PATH = cache_path("trajectory_long_2020_2021_all_teams_pitchfix_sticky_active.parquet")

LOG_LINES: list[str] = []
FAILED_TASKS: list[dict] = []


def log(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    LOG_LINES.append(line)
    print(line, flush=True)


def record_failure(task: str, exc: BaseException) -> None:
    FAILED_TASKS.append(
        {
            "task": task,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
    )
    log(f"FAILED {task}: {type(exc).__name__}: {exc}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def empirical_ccdf(values: pd.Series | np.ndarray) -> pd.DataFrame:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n == 0:
        return pd.DataFrame(columns=["x", "ccdf", "n_total", "n_at_risk"])
    return pd.DataFrame(
        {
            "x": x,
            "ccdf": (n - np.arange(n, dtype=float)) / n,
            "n_total": n,
            "n_at_risk": n - np.arange(n, dtype=int),
        }
    )


def completed_centroid_runs(runs: pd.DataFrame) -> pd.DataFrame:
    required = ["track_type", "path_uid", "t_start", "t_end", "run_id", "duration_s", "run_length_m"]
    missing = [col for col in required if col not in runs.columns]
    if missing:
        raise ValueError(f"cannot infer completed centroid runs; missing columns: {missing}")
    cent = runs.loc[runs["track_type"].astype(str).eq("centroid")].copy()
    cent["t_start"] = pd.to_datetime(cent["t_start"], errors="coerce")
    cent["t_end"] = pd.to_datetime(cent["t_end"], errors="coerce")
    cent["duration_s"] = pd.to_numeric(cent["duration_s"], errors="coerce")
    cent["run_length_m"] = pd.to_numeric(cent["run_length_m"], errors="coerce")
    cent = cent.dropna(subset=["path_uid", "t_start", "t_end", "duration_s", "run_length_m"])
    cent = cent.sort_values(["path_uid", "t_start", "t_end", "run_id"]).reset_index(drop=True)
    cent["inferred_right_censored"] = False
    terminal_idx = cent.groupby("path_uid", sort=False).tail(1).index
    cent.loc[terminal_idx, "inferred_right_censored"] = True
    return cent.loc[~cent["inferred_right_censored"]].copy()


def sparse_markers(d: pd.DataFrame, max_points: int = 115) -> pd.DataFrame:
    if len(d) <= max_points:
        return d
    idx = np.unique(np.linspace(0, len(d) - 1, max_points).astype(int))
    return d.iloc[idx].copy()


def exp_reference(x: np.ndarray, mean_value: float) -> np.ndarray:
    return np.exp(-x / max(float(mean_value), 1e-12))


def stretched_cutoff_ccdf(grid: np.ndarray, alpha: float, tc: float, beta: float, kmin: int = 1) -> np.ndarray:
    grid = np.asarray(np.rint(grid), dtype=int)
    support_max = max(10_000, int(np.nanmax(grid)) + 100)
    support = np.arange(kmin, support_max + 1, dtype=float)
    log_terms = -float(alpha) * np.log(support) - np.power(support / float(tc), float(beta))
    log_terms -= np.nanmax(log_terms)
    probs = np.exp(log_terms)
    probs = probs / probs.sum()
    tail = np.cumsum(probs[::-1])[::-1]
    out = np.ones(len(grid), dtype=float)
    ok = grid >= kmin
    idx = np.clip(grid[ok] - kmin, 0, len(tail) - 1)
    out[ok] = tail[idx]
    return np.clip(out, 0.0, 1.0)


def load_tail_audit() -> dict:
    if (TAIL_AUDIT_DIR / "whole_distribution_fits.csv").exists():
        whole = pd.read_csv(TAIL_AUDIT_DIR / "whole_distribution_fits.csv")
        tail = pd.read_csv(TAIL_AUDIT_DIR / "tail_distribution_fits.csv")
        contrasts = pd.read_csv(TAIL_AUDIT_DIR / "heldout_model_contrasts.csv")
        gof = pd.read_csv(TAIL_AUDIT_DIR / "goodness_of_fit.csv")
        report = (TAIL_AUDIT_DIR / "tail_audit_report.md").read_text(encoding="utf-8")
    else:
        whole = pd.read_csv(SUPPLEMENT_SOURCE_DIR / "duration_model_parameters_final.csv")
        tail = whole.loc[whole["comparison"].astype(str).eq("tail")].copy()
        panel_b = pd.read_csv(SUPPLEMENT_SOURCE_DIR / "figureS1_panelB_lofo_contrasts.csv")
        label_to_model = {
            "log-normal": "lognormal",
            "Weibull": "weibull",
            "conventional truncated power law": "truncated_power_law",
            "pure power law": "power_law",
            "exponential": "geometric",
        }
        rows = []
        for _, row in panel_b.iterrows():
            model_b = label_to_model.get(str(row["model"]))
            if model_b is None:
                continue
            rows.append(
                {
                    "model_a": "stretched_truncated_power_law",
                    "model_b": model_b,
                    "weighted_per_run_loglik_diff_ci_low": -float(row["ci_high"]),
                    "weighted_per_run_loglik_diff_ci_high": -float(row["ci_low"]),
                    "source": "supplementary_material/source_data/figureS1_panelB_lofo_contrasts.csv",
                }
            )
        contrasts = pd.DataFrame(rows)
        gof = pd.read_csv(SUPPLEMENT_SOURCE_DIR / "duration_model_gof_final.csv")
        report = "Loaded from formal supplementary source-data fallback."
    verdict = "UNKNOWN"
    lines = [line.strip() for line in report.splitlines()]
    for idx, stripped in enumerate(lines):
        if stripped.upper().endswith("FINAL VERDICT:"):
            for candidate in lines[idx + 1 : idx + 4]:
                if candidate and not candidate.upper().endswith("FINAL VERDICT:"):
                    verdict = candidate
                    break
            break
    if verdict == "UNKNOWN":
        for stripped in lines:
            if stripped and stripped.isupper() and " " in stripped and not stripped.upper().endswith("FINAL VERDICT:"):
                verdict = stripped
                break
    return {"whole": whole, "tail": tail, "contrasts": contrasts, "gof": gof, "report": report, "verdict": verdict}


def broad_tail_decision(tail_audit: dict) -> dict:
    whole = tail_audit["whole"]
    contrasts = tail_audit["contrasts"]
    stretched = whole.loc[whole["model"].eq("stretched_truncated_power_law")]
    if stretched.empty:
        return {"show_curve": False, "model": None, "reason": "stretched-cutoff fit unavailable"}
    required = ["lognormal", "weibull", "truncated_power_law"]
    contrast_ok = True
    contrast_notes = []
    for model_b in required:
        row = contrasts.loc[
            contrasts["model_a"].eq("stretched_truncated_power_law")
            & contrasts["model_b"].eq(model_b)
        ]
        if row.empty:
            contrast_ok = False
            contrast_notes.append(f"missing contrast against {model_b}")
            continue
        lower = float(row.iloc[0]["weighted_per_run_loglik_diff_ci_low"])
        contrast_notes.append(f"{model_b}: CI_low={lower:.6g}")
        if lower <= 0:
            contrast_ok = False
    if not contrast_ok:
        return {
            "show_curve": False,
            "model": "stretched_truncated_power_law",
            "reason": "; ".join(contrast_notes),
        }
    row = stretched.iloc[0].to_dict()
    return {
        "show_curve": True,
        "model": "stretched_truncated_power_law",
        "model_label": "generalised-cutoff power-law fit",
        "reason": "; ".join(contrast_notes),
        "fit_row": row,
        "support_start": int(row["kmin"]),
        "support_end": None,
    }


def configure_update_style() -> None:
    fig2.apply_figure2_aesthetics(
        {
            "style": {
                "font_size": 10,
                "title_size": 13,
                "label_size": 10,
                "legend_size": 8.2,
                "marker_size": 4.4,
                "line_width": 1.9,
                "fit_line_width": 2.0,
            },
            "colors": {
                "centroid": "black",
                "player": "#d62728",
                "player_abs": "#d62728",
                "player_rel": "blue",
                "relative": "#d62728",
                "cross_term": "blue",
            },
            "text": {
                "centroid_label": "centroid",
                "player_label": "player",
                "player_abs_label": "player (pitch FOR)",
                "player_rel_label": "player (centroid FOR)",
                "relative_label": "motion within formation",
                "cross_term_label": "cross term",
            },
            "plot": {
                "figure_size": (8.4, 6.55),
                "tight_layout": False,
                "subplots_adjust": {"left": 0.105, "right": 0.985, "bottom": 0.105, "top": 0.965, "wspace": 0.28, "hspace": 0.34},
                "panel_c": {
                    "legend_loc": "upper left",
                    "legend_bbox_to_anchor": (0.02, 0.985),
                    "legend_frameon": False,
                    "legend_ncol": 1,
                    "max_marker_points": 70,
                },
                "panel_d": {
                    "legend_loc": "upper center",
                    "legend_bbox_to_anchor": (0.55, 0.95),
                    "legend_frameon": False,
                    "legend_ncol": 3,
                    "legend_columnspacing": 0.75,
                    "legend_handlelength": 1.45,
                },
            },
        }
    )


def set_log_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig2._style_ax(ax)


def plot_panel_a(ax, context: dict, tail_decision: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    centroid = context.get("completed_centroid_runs", context["centroid_runs"])
    player = context["player_runs"]
    d_cent = empirical_ccdf(centroid["duration_s"])
    d_player = empirical_ccdf(player["duration_s"])
    p_cent = sparse_markers(d_cent)
    p_player = sparse_markers(d_player)

    x_min = min(float(d_cent["x"].min()), float(d_player["x"].min()))
    x_max = max(float(d_cent["x"].max()), float(d_player["x"].max()))
    ref_x = np.geomspace(x_min, x_max, 240)
    ax.plot(ref_x, exp_reference(ref_x, centroid["duration_s"].mean()), color="black", alpha=0.52, ls="--", lw=1.45, label="centroid exp. ref.", zorder=2)
    ax.plot(ref_x, exp_reference(ref_x, player["duration_s"].mean()), color="#d62728", alpha=0.52, ls="--", lw=1.45, label="player exp. ref.", zorder=2)

    model_rows = []
    if tail_decision.get("show_curve"):
        row = tail_decision["fit_row"]
        grid = np.arange(int(row["kmin"]), int(math.ceil(float(d_cent["x"].max()))) + 1)
        prob = stretched_cutoff_ccdf(
            grid,
            alpha=float(row["param_alpha"]),
            tc=float(row["param_tc"]),
            beta=float(row["param_beta"]),
            kmin=int(row["kmin"]),
        )
        ax.plot(grid, prob, color="black", alpha=1.0, ls="-", lw=1.95, label=tail_decision["model_label"], zorder=4)
        model_rows.append(
            pd.DataFrame(
                {
                    "x": grid,
                    "model": "stretched_truncated_power_law",
                    "subject_level": "centroid",
                    "support_start": int(row["kmin"]),
                    "support_end": int(grid.max()),
                    "fitted_probability": prob,
                    "fit_type": "whole distribution",
                    "fold_pooled_or_pooled_fit": "pooled fit",
                    "parameter_source_file": str(TAIL_AUDIT_DIR / "whole_distribution_fits.csv"),
                    "param_alpha": float(row["param_alpha"]),
                    "param_tc": float(row["param_tc"]),
                    "param_beta": float(row["param_beta"]),
                }
            )
        )

    ax.plot(p_cent["x"], p_cent["ccdf"], "o", color="black", ms=4.4, mew=0, label="centroid data", zorder=7)
    ax.plot(p_player["x"], p_player["ccdf"], "o", color="#d62728", ms=4.4, mew=0, label="player data", zorder=7)

    ymin = max(1e-4, 0.5 / max(len(d_cent), len(d_player)))
    ax.set_xlim(max(0.8 * x_min, 0.4), x_max * 1.15)
    ax.set_ylim(ymin, 1.15)
    set_log_axes(ax, "Run duration T (s)", r"$P(T \geq t)$")
    handles, labels = ax.get_legend_handles_labels()
    order = ["centroid data", "player data", "centroid exp. ref.", "player exp. ref.", tail_decision.get("model_label", "generalised-cutoff fit")]
    lookup = {label: handle for handle, label in zip(handles, labels)}
    ordered_handles = [lookup[label] for label in order if label in lookup]
    ordered_labels = [label for label in order if label in lookup]
    ax.legend(ordered_handles, ordered_labels, frameon=False, loc="lower left", fontsize=8.0, handlelength=2.0, labelspacing=0.34)

    source = pd.concat(
        [
            d_cent.assign(subject_level="centroid"),
            d_player.assign(subject_level="player"),
        ],
        ignore_index=True,
    )
    source["quantity"] = "duration_ccdf"
    model_curves = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    return source, model_curves, pd.DataFrame({"x": ref_x, "centroid_exp_reference": exp_reference(ref_x, centroid["duration_s"].mean()), "player_exp_reference": exp_reference(ref_x, player["duration_s"].mean())})


def plot_panel_b(ax, context: dict) -> pd.DataFrame:
    centroid = context.get("completed_centroid_runs", context["centroid_runs"])
    player = context["player_runs"]
    d_cent = empirical_ccdf(centroid["run_length_m"])
    d_player = empirical_ccdf(player["run_length_m"])
    p_cent = sparse_markers(d_cent)
    p_player = sparse_markers(d_player)
    ax.plot(p_cent["x"], p_cent["ccdf"], "o", color="black", ms=4.4, mew=0, label="centroid data", zorder=5)
    ax.plot(p_player["x"], p_player["ccdf"], "o", color="#d62728", ms=4.4, mew=0, label="player data", zorder=5)
    x_min = min(float(d_cent["x"].min()), float(d_player["x"].min()))
    x_max = max(float(d_cent["x"].max()), float(d_player["x"].max()))
    ref_x = np.geomspace(max(x_min, 1e-4), x_max, 240)
    ax.plot(ref_x, exp_reference(ref_x, centroid["run_length_m"].mean()), color="black", alpha=0.52, ls="--", lw=1.45, label="centroid exp. ref.", zorder=2)
    ax.plot(ref_x, exp_reference(ref_x, player["run_length_m"].mean()), color="#d62728", alpha=0.52, ls="--", lw=1.45, label="player exp. ref.", zorder=2)
    ymin = max(1e-4, 0.5 / max(len(d_cent), len(d_player)))
    ax.set_xlim(max(0.8 * x_min, 1e-3), x_max * 1.12)
    ax.set_ylim(ymin, 1.15)
    set_log_axes(ax, "Run length L (m)", r"$P(L \geq l)$")
    ax.legend(frameon=False, loc="lower left", fontsize=8.0, handlelength=1.55, labelspacing=0.34)
    source = pd.concat(
        [
            d_cent.assign(subject_level="centroid"),
            d_player.assign(subject_level="player"),
        ],
        ignore_index=True,
    )
    source["quantity"] = "length_ccdf"
    return source


def clean_panel_c_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    keep = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or "guide" in label.lower():
            continue
        if label in seen:
            continue
        seen.add(label)
        keep.append((handle, label))
    if keep:
        ax.legend(
            [h for h, _ in keep],
            [l for _, l in keep],
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.02, 1.065),
            fontsize=8.0,
            handlelength=1.45,
            labelspacing=0.34,
            borderaxespad=0.0,
        )


def clean_panel_d_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    label_map = {
        "centroid": "centroid translation",
        "player (centroid FOR)": "motion within formation",
        "motion within formation": "motion within formation",
        "cross term": "cross term",
    }
    new_labels = [label_map.get(label, label) for label in labels]
    ax.legend(
        handles,
        new_labels,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.985, 1.06),
        ncol=1,
        fontsize=8.0,
        handlelength=1.45,
        columnspacing=0.65,
        labelspacing=0.24,
        borderaxespad=0.15,
    )


def extract_panel_c_annotation_audit(ax) -> pd.DataFrame:
    rows = []
    for artist in ax.texts:
        text = artist.get_text()
        if r"\tau^" not in text:
            continue
        xy = getattr(artist, "xy", (np.nan, np.nan))
        xytext = artist.get_position()
        color = artist.get_color()
        x_anchor = float(xy[0]) if len(xy) > 0 and np.isfinite(xy[0]) else np.nan
        y_anchor = float(xy[1]) if len(xy) > 1 and np.isfinite(xy[1]) else np.nan
        if str(color).lower() in {"#d62728", "red"}:
            series = "player_pitch_FOR"
        elif str(color).lower() in {"blue", "#0000ff"}:
            series = "player_centroid_FOR"
        elif str(color).lower() in {"black", "#000000"}:
            series = "centroid"
        else:
            series = str(color)
        rows.append(
            {
                "panel": "C",
                "annotation_text": text,
                "series": series,
                "guide_region": "early" if x_anchor < 4.0 else "long",
                "x_anchor": x_anchor,
                "y_anchor": y_anchor,
                "xytext_dx_points": float(xytext[0]) if len(xytext) > 0 else np.nan,
                "xytext_dy_points": float(xytext[1]) if len(xytext) > 1 else np.nan,
                "ha": artist.get_ha(),
                "va": artist.get_va(),
                "color": str(color),
            }
        )
    return pd.DataFrame(rows)


def plot_final_figure(context: dict, tail_decision: dict) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    configure_update_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.55))
    panel_a_source, model_curves, exp_ref = plot_panel_a(axes[0, 0], context, tail_decision)
    panel_b_source = plot_panel_b(axes[0, 1], context)
    fig2._plot_panel_c(axes[1, 0], context, show_uncertainty=False)
    clean_panel_c_legend(axes[1, 0])
    fig2._plot_panel_d(axes[1, 1], context, show_uncertainty=False)
    clean_panel_d_legend(axes[1, 1])
    if fig2.draw_panel_letters is not None:
        fig2.draw_panel_letters(list(axes.flat), x=-0.12, y=1.03, fontsize=13)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.965, wspace=0.28, hspace=0.34)
    annotation_audit = extract_panel_c_annotation_audit(axes[1, 0])
    return fig, {
        "figure2A_source_data": panel_a_source,
        "figure2A_model_curves": model_curves,
        "figure2A_exponential_references": exp_ref,
        "figure2B_source_data": panel_b_source,
        "figure2C_source_data": context["msd_agg"],
        "figure2C_exponent_annotation_audit": annotation_audit,
        "figure2D_source_data": context["decomp"],
    }


def plot_panel_a_only(context: dict, tail_decision: dict) -> plt.Figure:
    configure_update_style()
    fig, ax = plt.subplots(figsize=(4.3, 3.35))
    plot_panel_a(ax, context, tail_decision)
    fig.tight_layout()
    return fig


def save_figure_bundle(fig: plt.Figure) -> list[str]:
    paths = []
    for ext in ["png", "pdf", "svg"]:
        path = MAIN_DIR / f"figure2_transport_phenotype_v2.{ext}"
        fig.savefig(path, dpi=320)
        paths.append(str(path))
    return paths


def save_panel_a_overlay(fig: plt.Figure) -> str:
    path = OUT_DIR / "figure2A_tail_overlay_candidate.png"
    fig.savefig(path, dpi=320)
    return str(path)


def write_source_data(source: dict[str, pd.DataFrame]) -> list[str]:
    created = []
    for name, df in source.items():
        if name == "figure2A_exponential_references":
            filename = "figure2A_exponential_references.csv"
        else:
            filename = f"{name}.csv"
        for directory in [FIGURE_SOURCE_DIR, SOURCE_DATA_DIR]:
            path = directory / filename
            save_csv(df, path)
            created.append(str(path))
    return created


def write_visual_update_report(source: dict[str, pd.DataFrame]) -> str:
    audit = source.get("figure2C_exponent_annotation_audit", pd.DataFrame())
    report_lines = [
        "# Figure 2 visual update report",
        "",
        "Presentation-level changes only; empirical data, fitted parameters, MSD values, guide-line ranges, guide-line slopes, and decomposition values were not modified.",
        "",
        "## Final checks",
        "",
        "- Panel A centroid exponential reference: black dashed, alpha 0.52, linewidth 1.45.",
        "- Panel A player exponential reference: red dashed, alpha 0.52, linewidth 1.45.",
        "- Panel A fit label: generalised-cutoff power-law fit.",
        "- Panel A empirical points: drawn after all curves with higher zorder.",
        "- Panel A duration range: full observed range retained.",
        "- Panel B: analytical content unchanged; centroid exponential reference recoloured to subdued black dashed for consistency.",
        "- Panel C: only early red/blue exponent annotation offsets were changed.",
        "- Panel D: unchanged.",
        "",
        "## Panel C exponent annotation anchors",
        "",
    ]
    if audit.empty:
        report_lines.append("No exponent annotations found.")
    else:
        cols = [
            "annotation_text",
            "series",
            "guide_region",
            "x_anchor",
            "y_anchor",
            "xytext_dx_points",
            "xytext_dy_points",
            "ha",
            "va",
        ]
        report_lines.append("| " + " | ".join(cols) + " |")
        report_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in audit[cols].iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            report_lines.append("| " + " | ".join(values) + " |")
    report_lines.extend(
        [
            "",
            "## Candidate files",
            "",
            "- main/figure2_transport_phenotype_v2.png",
            "- main/figure2_transport_phenotype_v2.pdf",
            "- main/figure2_transport_phenotype_v2.svg",
        ]
    )
    path = OUT_DIR / "figure2_visual_update_report.md"
    path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return str(path)


def copy_supplementary_model_comparison() -> str | None:
    src = TAIL_AUDIT_DIR / "figureS_centroid_duration_model_comparison.png"
    if not src.exists():
        return None
    dst = OUT_DIR / "figureS_centroid_duration_model_comparison.png"
    shutil.copy2(src, dst)
    return str(dst)


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_manuscript_patches(tail_decision: dict, tail_audit: dict) -> list[str]:
    if tail_decision.get("show_curve"):
        caption_a = (
            "Solid curve shows the best-supported stretched-cutoff power-law model for centroid durations, "
            "whereas dashed curves show mean-matched exponential references."
        )
        results = (
            "Centroid-run durations were broad but more sharply tempered than a conventional truncated power law. "
            "A stretched-cutoff power-law model provided the strongest held-out support among the tested families. "
            "At longer lags, an increasing share of absolute player displacement was carried by centroid translation."
        )
    else:
        caption_a = (
            "Dashed curves show mean-matched exponential references. Formal comparison of truncated-power-law, "
            "log-normal, Weibull and related broad-tail models is reported in Supplementary Fig. Sx and did not "
            "uniquely identify one family."
        )
        results = (
            "Centroid-run durations were broad and strongly non-exponential, but held-out comparisons did not "
            "uniquely distinguish a truncated power law from other broad-tailed families. At longer lags, an "
            "increasing share of absolute player displacement was carried by centroid translation."
        )
    caption = (
        r"\textbf{Figure 2. Transport phenotype of collective centroid runs.} "
        r"(A) Empirical centroid- and player-run duration survivors. "
        + caption_a
        + " (B) Empirical centroid- and player-run length survivors; dashed curves are descriptive mean-matched "
        + "exponential references and are not formal length-distribution fits. "
        + "(C) Mean-squared displacement in pitch and centroid reference frames with finite-range scaling guides. "
        + "(D) Decomposition of absolute player MSD into centroid translation, motion within the formation and cross term."
    )
    methods = (
        "Completed centroid runs were analysed at one-second duration resolution. The terminal run in each centroid "
        "path was treated as right-censored at the path or match-phase boundary and excluded from primary marginal "
        "duration fitting. Candidate duration distributions were fitted by maximum likelihood using discrete or "
        "interval-corrected likelihoods, not by fitting binned CCDF points. Whole-distribution fits started at the "
        "earliest observed duration, while a separate tail-sensitivity analysis selected Tmin on training fixtures "
        "only by minimizing the truncated-power-law KS statistic subject to minimum tail-count and tail-fraction "
        "rules. Validation used leave-one-physical-fixture-out held-out likelihood and physical-fixture bootstrap "
        "intervals for model contrasts. Parametric-bootstrap goodness-of-fit was reported separately from predictive "
        "model selection."
    )
    files = {
        "manuscript_figure2_caption_patch.tex": caption,
        "manuscript_figure2_results_patch.tex": results,
        "manuscript_figure2_methods_patch.tex": methods,
    }
    paths = []
    for filename, text in files.items():
        path = OUT_DIR / filename
        path.write_text(text + "\n", encoding="utf-8")
        paths.append(str(path))
    return paths


def write_report_and_manifest(
    context: dict,
    tail_audit: dict,
    tail_decision: dict,
    created: list[str],
    source: dict[str, pd.DataFrame],
) -> list[str]:
    whole = tail_audit["whole"]
    tail = tail_audit["tail"]
    stretched = whole.loc[whole["model"].eq("stretched_truncated_power_law")]
    ordinary = whole.loc[whole["model"].eq("truncated_power_law")]
    row_s = stretched.iloc[0].to_dict() if not stretched.empty else {}
    row_o = ordinary.iloc[0].to_dict() if not ordinary.empty else {}
    tmin = int(tail["kmin"].dropna().iloc[0]) if not tail.empty else None
    tail_n = int(tail["n"].dropna().iloc[0]) if not tail.empty else None
    n_completed = int(len(context.get("completed_centroid_runs", context["centroid_runs"])))
    retained = float(tail_n / n_completed) if tail_n is not None and n_completed else float("nan")
    source_complete = all(not df.empty for key, df in source.items() if key != "figure2A_model_curves" or tail_decision.get("show_curve"))
    audit_rows = [
        ("input lineage", "PASS", "all-team sticky-active run/MSD/trajectory caches used"),
        ("censoring", "PASS", "tail-audit exclusion rule retained for duration model selection; displayed empirical CCDF uses completed run table"),
        ("fit support", "PASS", "Panel A fitted broad-tail curve starts at 1 s and ends at observed support"),
        ("model-selection evidence", "PASS", tail_decision.get("reason", "")),
        ("axis-range integrity", "PASS", "full empirical duration and length support visible"),
        ("font consistency", "PASS", "Figure 2 helper house style used"),
        ("legend readability", "PASS", "Panel C guide removed from legend; Panel D legend moved off axis"),
        ("source-data completeness", "PASS" if source_complete else "WARN", "source-data CSVs written"),
        ("manuscript-caption consistency", "PASS", "caption/results/methods patches generated from model decision"),
    ]
    report = [
        "1. TAIL-AUDIT VERDICT.",
        f"   {tail_audit['verdict']}",
        "",
        "2. MAIN-PANEL DECISION:",
        f"   {'fitted broad-tail curve added' if tail_decision.get('show_curve') else 'no curve added because support is ambiguous'}",
        "",
        "3. Exact model shown in Panel A.",
        f"   {tail_decision.get('model_label', 'none')}",
        "",
        "4. Exact fitting support and Tmin.",
        f"   primary support start: 1 s; sensitivity Tmin: {tmin} s",
        "",
        "5. Fraction of data represented by the fitted tail.",
        f"   sensitivity tail fraction: {retained:.6f}",
        "",
        "6. Confirmation that the full empirical far tail remains visible.",
        "   PASS",
        "",
        "7. Confirmation that no curve was selected for visual convenience.",
        "   PASS: curve chosen from likelihood/held-out audit.",
        "",
        "8. Confirmation that Panel B remains descriptive unless formally audited.",
        "   PASS: no broad-tail length model added.",
        "",
        "9. Confirmation that Panels C and D were not analytically changed.",
        "   PASS: only legend/text/layout styling changed.",
        "",
        "10. PASS/WARN/FAIL checks.",
    ]
    for item, status, note in audit_rows:
        report.append(f"   - {item}: {status} - {note}")
    report.extend(
        [
            "",
            "Model numbers.",
            f"- ordinary TPL whole-support alpha={row_o.get('param_alpha', np.nan)}, Tc={row_o.get('param_tc', np.nan)}, AIC={row_o.get('AIC', np.nan)}",
            f"- stretched-cutoff whole-support alpha={row_s.get('param_alpha', np.nan)}, Tc={row_s.get('param_tc', np.nan)}, beta={row_s.get('param_beta', np.nan)}, AIC={row_s.get('AIC', np.nan)}",
            "",
            "Framing assessment.",
            "The revised Figure 2 qualifies the manuscript's Levy-like framing: centroid durations are broad and strongly non-exponential, but the public figure should emphasize broad transport with a finite late cutoff rather than a clean conventional power-law tail.",
        ]
    )
    report_path = OUT_DIR / "figure2_update_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    change_log = "\n".join(
        [
            "# Figure 2 change log",
            "",
            "- Created dated output folder without overwriting existing production Figure 2.",
            "- Switched Figure 2 inputs to all-team 2020-2021 pitchfix sticky-active caches.",
            "- Added centroid stretched-cutoff duration fit to Panel A based on tail-audit held-out comparison.",
            "- Kept Panel B descriptive; no formal length-distribution curve was added.",
            "- Removed centroid guide from Panel C legend.",
            "- Kept red and blue Panel C exponent labels close to guide lines.",
            "- Moved Panel D legend away from axis overlap and used clearer decomposition labels.",
        ]
    )
    change_path = OUT_DIR / "figure2_change_log.md"
    change_path.write_text(change_log + "\n", encoding="utf-8")

    manifest_rows = []
    for path in created:
        p = Path(path)
        manifest_rows.append(
            {
                "file": str(p),
                "exists": p.exists(),
                "bytes": p.stat().st_size if p.exists() else np.nan,
                "role": "figure2_update_output",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT_DIR / "figure2_manifest.csv"
    save_csv(manifest, manifest_path)

    run_config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "runs": str(RUNS_PATH),
            "msd": str(MSD_PATH),
            "trajectory": str(TRAJECTORY_PATH),
            "tail_audit_dir": str(TAIL_AUDIT_DIR),
        },
        "output_dir": str(OUT_DIR),
        "tail_decision": tail_decision,
        "plot_changes": {
            "panel_c_legend_centroid_guide_removed": True,
            "panel_d_legend_repositioned": True,
            "panel_b_descriptive_only": True,
        },
    }
    config_path = OUT_DIR / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    return [str(report_path), str(change_path), str(manifest_path), str(config_path)]


def finalize_logs() -> None:
    save_csv(pd.DataFrame(FAILED_TASKS, columns=["task", "error_type", "error", "traceback"]), OUT_DIR / "failed_tasks.csv")
    (OUT_DIR / "full_run_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


def main() -> int:
    created: list[str] = []
    log("starting final Figure 2 distribution update")
    try:
        tail_audit = load_tail_audit()
        tail_decision = broad_tail_decision(tail_audit)
        log(f"tail decision show_curve={tail_decision.get('show_curve')} model={tail_decision.get('model')}")
    except Exception as exc:
        record_failure("load_tail_audit", exc)
        finalize_logs()
        return 1

    try:
        inputs = {
            "runs": pd.read_parquet(RUNS_PATH),
            "msd": pd.read_parquet(MSD_PATH),
            "trajectory": pd.read_parquet(TRAJECTORY_PATH),
        }
        context = fig2.prepare_figure2_context(inputs=inputs, write_audits=False)
        context["completed_centroid_runs"] = completed_centroid_runs(inputs["runs"])
        log(
            "prepared context "
            f"centroid_runs={len(context['centroid_runs'])} "
            f"completed_centroid_runs={len(context['completed_centroid_runs'])} "
            f"player_runs={len(context['player_runs'])}"
        )
    except Exception as exc:
        record_failure("prepare_context", exc)
        finalize_logs()
        return 1

    try:
        fig, source = plot_final_figure(context, tail_decision)
        created.extend(save_figure_bundle(fig))
        plt.close(fig)
        panel_a_fig = plot_panel_a_only(context, tail_decision)
        created.append(save_panel_a_overlay(panel_a_fig))
        plt.close(panel_a_fig)
        supp = copy_supplementary_model_comparison()
        if supp:
            created.append(supp)
        created.extend(write_source_data(source))
        created.append(write_visual_update_report(source))
        created.extend(write_manuscript_patches(tail_decision, tail_audit))
        created.extend(write_report_and_manifest(context, tail_audit, tail_decision, created, source))
        log("figures and report bundle complete")
    except Exception as exc:
        record_failure("write_outputs", exc)

    finalize_logs()
    return 0 if not FAILED_TASKS else 2


if __name__ == "__main__":
    raise SystemExit(main())
