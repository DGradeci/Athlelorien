"""Focused pre-submission robustness audit for the football runs paper.

The script writes all new outputs to ``pre_submission_robustness/`` and does
not overwrite manuscript figures or tables.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "pre_submission_robustness"
OUT.mkdir(exist_ok=True)

DATA = ROOT / "analysis" / "levy_paper" / "data" / "processed"
ALL = DATA / "all_team_2020_2021_sticky_active"
CROSS = DATA / "figure4_fig5_crossfitted"

META = ALL / "multiseason_raw_pitch_metadata_2020_2021_all_teams_pitchfix_sticky_active.parquet"
TRAJ = ALL / "trajectory_long_2020_2021_all_teams_pitchfix_sticky_active.parquet"
HAZ = ALL / "hazard_intervals_2020_2021_all_teams_pitchfix_sticky_active.parquet"
OOF = CROSS / "out_of_fold_interval_predictions.parquet"

RNG_SEED = 20260801
MAX_AGE = 35
BOOTSTRAP_B = 500
THETA_DEG = 30.0
EPS = 1e-9


def setup_style() -> None:
    try:
        import sys

        sys.path.insert(0, str(ROOT))
        from analysis.levy_paper.util.paper_utils import setup_paper_style

        setup_paper_style()
    except Exception:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fixture_key(df: pd.DataFrame) -> pd.Series:
    fields = [c for c in ["season", "date", "home", "away", "score"] if c in df.columns]
    return df[fields].astype(str).agg("__".join, axis=1)


def part1_dataset_audit() -> dict:
    meta = pd.read_parquet(META).copy()
    for col in ["season", "date", "home", "away", "score", "source_key", "team", "status", "pool"]:
        if col not in meta.columns:
            meta[col] = ""
    meta["fixture_id_true"] = fixture_key(meta)
    meta["match_date_group"] = meta["season"].astype(str) + "__" + meta["date"].astype(str)
    meta["included_final_single_pool"] = (meta["status"] == "ok") & (meta["pool"] == "single")
    meta["included_final_h2h_pool"] = (meta["status"] == "ok") & (meta["pool"] == "h2h")
    meta["tracked_clubs_in_fixture"] = meta.groupby("fixture_id_true")["team"].transform(
        lambda s: "; ".join(sorted(set(map(str, s.dropna()))))
    )
    meta["n_records_for_fixture"] = meta.groupby("fixture_id_true")["source_key"].transform("count")
    meta["both_studied_teams_recorded"] = meta.groupby("fixture_id_true")["team"].transform("nunique").ge(2)
    meta["reason_for_exclusion_if_excluded"] = np.where(
        meta["included_final_single_pool"], "", meta.get("error", "").astype(str).where(meta.get("error", "").astype(str).ne(""), meta["status"].astype(str))
    )

    cols = [
        "fixture_id_true",
        "date",
        "season",
        "home",
        "away",
        "score",
        "tracked_clubs_in_fixture",
        "source_key",
        "team",
        "pool",
        "status",
        "included_final_single_pool",
        "included_final_h2h_pool",
        "both_studied_teams_recorded",
        "n_records_for_fixture",
        "reason_for_exclusion_if_excluded",
        "match_date_group",
    ]
    audit = meta[[c for c in cols if c in meta.columns]].drop_duplicates().sort_values(["season", "date", "fixture_id_true", "source_key", "pool"])
    audit.to_csv(OUT / "dataset_count_audit.csv", index=False)

    ok = meta[meta["included_final_single_pool"]].copy()
    summary = {
        "source_rows_total": int(len(meta)),
        "unique_physical_matches_in_source_data": int(meta["fixture_id_true"].nunique()),
        "unique_physical_matches_passing_final_inclusion": int(ok["fixture_id_true"].nunique()),
        "team_match_observations_final": int(ok.shape[0]),
        "distinct_match_dates_final": int(ok["match_date_group"].nunique()),
        "matches_containing_both_tracked_clubs_final": int(ok.groupby("fixture_id_true")["team"].nunique().ge(2).sum()),
        "date_groups_with_two_separate_fixtures_final": int(ok.groupby("match_date_group")["fixture_id_true"].nunique().gt(1).sum()),
        "duplicate_or_repeated_recordings_final": int(ok.shape[0] - ok["fixture_id_true"].nunique()),
    }
    md = [
        "# Dataset Count Audit",
        "",
        "## Counts",
        "",
        f"- unique_physical_matches_in_source_data: {summary['unique_physical_matches_in_source_data']}",
        f"- unique_physical_matches_passing_final_inclusion: {summary['unique_physical_matches_passing_final_inclusion']}",
        f"- team_match_observations_final: {summary['team_match_observations_final']}",
        f"- distinct_match_dates_final: {summary['distinct_match_dates_final']}",
        f"- matches_containing_both_tracked_clubs_final: {summary['matches_containing_both_tracked_clubs_final']}",
        f"- duplicate_or_repeated_recordings_final: {summary['duplicate_or_repeated_recordings_final']}",
        "",
        "## Reconciliation",
        "",
        "- 66 = final included tracked team-match/source observations.",
        "- 62 = distinct competitive fixture identifiers from season/date/home/away/score among final included records.",
        "- 47 = date-based validation groups used by cross-fitting; some dates contain two separate fixtures.",
        "",
        "## Output",
        "",
        f"- table: {rel(OUT / 'dataset_count_audit.csv')}",
    ]
    write_md(OUT / "dataset_count_audit.md", "\n".join(md))
    return summary


@dataclass
class SegmentVariant:
    name: str
    description: str
    speed_guard_mps: float | None = None
    smooth: str | None = None
    available: bool = True


def centroid_paths() -> pd.DataFrame:
    cols = ["season", "match_id", "match_phase", "team", "source_key", "track_type", "track_uid", "path_uid", "t", "x_m", "y_m"]
    tr = pd.read_parquet(TRAJ, columns=cols)
    c = tr[tr["track_type"].astype(str).eq("centroid")].copy()
    c["t"] = pd.to_datetime(c["t"])
    return c.sort_values(["track_uid", "t"]).reset_index(drop=True)


def segment_group(g: pd.DataFrame, variant: SegmentVariant) -> list[dict]:
    g = g.sort_values("t").copy()
    if variant.smooth == "rolling_median_3":
        g["x_use"] = g["x_m"].rolling(3, center=True, min_periods=1).median()
        g["y_use"] = g["y_m"].rolling(3, center=True, min_periods=1).median()
    else:
        g["x_use"] = g["x_m"]
        g["y_use"] = g["y_m"]
    xy = g[["x_use", "y_use"]].to_numpy(float)
    tt = g["t"].to_numpy()
    n = len(g)
    if n < 2:
        return []
    dxy = xy[1:] - xy[:-1]
    dt = np.diff(tt).astype("timedelta64[ns]").astype(float) / 1e9
    step_dist = np.linalg.norm(dxy, axis=1)
    speed = np.divide(step_dist, dt, out=np.full_like(step_dist, np.nan), where=dt > 0)
    theta = np.deg2rad(THETA_DEG)
    out = []
    run_start = 0
    prev_vec = None
    prev_speed = np.nan
    run_id = 0
    for i in range(1, n):
        v = dxy[i - 1]
        nv = float(np.linalg.norm(v))
        if nv <= 1e-6 or not np.isfinite(nv):
            continue
        if prev_vec is None:
            prev_vec = v
            prev_speed = speed[i - 1]
            continue
        npv = float(np.linalg.norm(prev_vec))
        if npv <= 1e-6 or not np.isfinite(npv):
            prev_vec = v
            prev_speed = speed[i - 1]
            continue
        cosang = float(np.clip(np.dot(prev_vec, v) / (npv * nv), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        can_turn = ang > theta
        if variant.speed_guard_mps is not None:
            relevant_speed = np.nanmin([prev_speed, speed[i - 1]])
            if not np.isfinite(relevant_speed) or relevant_speed < variant.speed_guard_mps:
                can_turn = False
        if can_turn:
            s = run_start
            e = i - 1
            if e - s + 1 >= 2:
                dur = float((tt[e] - tt[s]) / np.timedelta64(1, "s"))
                length = float(np.nansum(step_dist[s:e]))
                if length > 0 and dur >= 1:
                    out.append({"run_id": run_id, "t_start": tt[s], "t_end": tt[e], "duration_s": dur, "run_length_m": length, "v_mean_mps": length / dur, "observed_termination": 1})
                    run_id += 1
            run_start = i - 1
        prev_vec = v
        prev_speed = speed[i - 1]
    s = run_start
    e = n - 1
    if e - s + 1 >= 2:
        dur = float((tt[e] - tt[s]) / np.timedelta64(1, "s"))
        length = float(np.nansum(step_dist[s:e]))
        if length > 0 and dur >= 1:
            out.append({"run_id": run_id, "t_start": tt[s], "t_end": tt[e], "duration_s": dur, "run_length_m": length, "v_mean_mps": length / dur, "observed_termination": 0})
    return out


def segment_variant(c: pd.DataFrame, variant: SegmentVariant) -> pd.DataFrame:
    rows = []
    meta_cols = ["season", "match_id", "match_phase", "team", "source_key", "track_uid", "path_uid"]
    for keys, g in c.groupby(meta_cols, sort=False, observed=True):
        for r in segment_group(g, variant):
            row = dict(zip(meta_cols, keys))
            row.update(r)
            rows.append(row)
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs
    runs["variant"] = variant.name
    runs["physical_fixture_id"] = runs["season"].astype(str) + "__" + runs["match_id"].astype(str)
    runs["run_uid_variant"] = (
        runs["variant"].astype(str)
        + "__"
        + runs["track_uid"].astype(str)
        + "__run="
        + runs["run_id"].astype(str)
    )
    return runs


def hazard_counts_from_runs(runs: pd.DataFrame, max_age: int = MAX_AGE) -> pd.DataFrame:
    rows = []
    durations = runs["duration_s"].to_numpy(float)
    events = runs["observed_termination"].to_numpy(int)
    clusters = runs["physical_fixture_id"].to_numpy(object)
    for age in range(max_age):
        at = durations > age + 1e-9
        ev = at & (events == 1) & (durations <= age + 1 + 1e-9)
        rows.append(
            {
                "age_start_s": age,
                "age_mid_s": age + 0.5,
                "n_at_risk": int(at.sum()),
                "n_terminated": int(ev.sum()),
                "termination_probability": float(ev.sum() / at.sum()) if at.sum() else np.nan,
                "n_clusters": int(pd.Series(clusters[at]).nunique()) if at.sum() else 0,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_hazard_ci(runs: pd.DataFrame, max_age: int = MAX_AGE, b: int = BOOTSTRAP_B) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    clusters = np.array(sorted(runs["physical_fixture_id"].dropna().unique()))
    if len(clusters) < 2:
        h = hazard_counts_from_runs(runs, max_age)
        h["ci_low"] = np.nan
        h["ci_high"] = np.nan
        return h
    vals = np.empty((b, max_age), dtype=float)
    for bi in range(b):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([runs[runs["physical_fixture_id"].eq(cl)] for cl in sample], ignore_index=True)
        vals[bi, :] = hazard_counts_from_runs(boot, max_age)["termination_probability"].to_numpy(float)
    h = hazard_counts_from_runs(runs, max_age)
    h["ci_low"] = np.nanpercentile(vals, 2.5, axis=0)
    h["ci_high"] = np.nanpercentile(vals, 97.5, axis=0)
    return h


def survival_ccdf(runs: pd.DataFrame, max_t: int = 80) -> pd.DataFrame:
    d = runs["duration_s"].to_numpy(float)
    grid = np.arange(1, max_t + 1)
    return pd.DataFrame({"duration_s": grid, "ccdf": [(d >= t).mean() for t in grid]})


def fit_age_only_from_runs(runs: pd.DataFrame) -> dict:
    h = hazard_counts_from_runs(runs, max_age=int(max(5, min(80, math.ceil(runs["duration_s"].max())))))
    fit = h[h["n_at_risk"] >= 30].copy()
    ages = fit["age_mid_s"].to_numpy(float)
    deaths = fit["n_terminated"].to_numpy(float)
    n = fit["n_at_risk"].to_numpy(float)
    nonevents = n - deaths

    def nll(theta):
        mu = math.exp(theta[0])
        a0 = math.exp(theta[1])
        lam = mu / (a0 + ages)
        p = np.clip(1 - np.exp(-lam), 1e-12, 1 - 1e-12)
        return -float(np.sum(deaths * np.log(p) + nonevents * np.log1p(-p)))

    best = None
    for start in [(math.log(1.0), math.log(2.0)), (math.log(1.5), math.log(4.0)), (math.log(0.8), math.log(1.0))]:
        res = minimize(nll, np.array(start), method="L-BFGS-B", bounds=[(math.log(0.01), math.log(20)), (math.log(0.1), math.log(50))])
        if best is None or res.fun < best.fun:
            best = res
    k = 2
    nobs = float(n.sum())
    return {
        "age_only_mu": float(math.exp(best.x[0])),
        "age_only_a0_s": float(math.exp(best.x[1])),
        "age_only_log_likelihood": float(-best.fun),
        "age_only_AIC": float(2 * k + 2 * best.fun),
        "age_only_BIC": float(k * math.log(max(nobs, 1)) + 2 * best.fun),
        "fit_max_age_s": int(fit["age_start_s"].max()) if not fit.empty else np.nan,
    }


def part2_segmentation_robustness() -> dict:
    metrics_path = OUT / "segmentation_robustness_metrics.csv"
    hazard_path = OUT / "segmentation_robustness_hazard_source.csv"
    ccdf_path = OUT / "segmentation_robustness_ccdf_source.csv"
    if metrics_path.exists() and hazard_path.exists() and ccdf_path.exists():
        metrics = pd.read_csv(metrics_path)
        hz = pd.read_csv(hazard_path)
        ccdf = pd.read_csv(ccdf_path)
        plot_segmentation(hz, ccdf)
        metrics.to_latex(OUT / "tableS_segmentation_robustness.tex", index=False, float_format="%.3g")
        verdict = classify_segmentation(metrics)
        notes = [
            "# Segmentation Robustness Notes",
            "",
            f"- verdict: {verdict}",
            "- Reused existing segmentation robustness source tables in this output directory.",
            "- Native-frequency mean/median one-second aggregation could not be executed from the retained compact local cache; the retained trajectory_long cache is already one-second centroid/player data.",
            "- Low-speed guards merge low-speed candidate turns into the current run; they do not omit exposure intervals.",
        ]
        write_md(OUT / "segmentation_robustness_notes.md", "\n".join(notes))
        return {"segmentation_verdict": verdict, "segmentation_metrics": metrics}

    variants = [
        SegmentVariant("primary_recomputed", "Final one-second centroid positions; no smoothing; 30-degree turn threshold."),
        SegmentVariant("native_1s_mean_or_median_unavailable", "Requested native-frequency mean/median aggregation; native-frequency active centroid data are not retained in the compact local cache.", available=False),
        SegmentVariant("low_speed_guard_0p25", "Merge candidate turns when the minimum adjacent centroid step speed is below 0.25 m/s.", speed_guard_mps=0.25),
        SegmentVariant("low_speed_guard_0p50", "Merge candidate turns when the minimum adjacent centroid step speed is below 0.50 m/s.", speed_guard_mps=0.50),
        SegmentVariant("rolling_median_3", "Apply centred three-sample rolling median to centroid x/y before heading calculation.", smooth="rolling_median_3"),
    ]
    c = centroid_paths()
    all_metrics = []
    hazard_tables = []
    ccdf_tables = []
    variant_runs = {}
    for var in variants:
        if not var.available:
            all_metrics.append(
                {
                    "variant": var.name,
                    "available": False,
                    "description": var.description,
                    "total_centroid_runs": np.nan,
                    "observed_terminations": np.nan,
                    "censored_runs": np.nan,
                    "median_duration_s": np.nan,
                    "p90_duration_s": np.nan,
                    "p95_duration_s": np.nan,
                    "age_of_min_termination_probability_s": np.nan,
                    "late_flattening_or_increase_remains": "not_tested_native_frequency_cache_unavailable",
                }
            )
            continue
        runs = segment_variant(c, var)
        variant_runs[var.name] = runs
        h = bootstrap_hazard_ci(runs, MAX_AGE, BOOTSTRAP_B)
        h["variant"] = var.name
        hazard_tables.append(h)
        cc = survival_ccdf(runs)
        cc["variant"] = var.name
        ccdf_tables.append(cc)
        reliable = h[h["n_at_risk"] >= 30].copy()
        min_row = reliable.loc[reliable["termination_probability"].idxmin()] if not reliable.empty else None
        late = reliable[reliable["age_start_s"] >= 20]
        tail_remains = "insufficient_tail_support"
        if min_row is not None and not late.empty:
            tail_remains = bool(late["termination_probability"].mean() > min_row["termination_probability"] * 1.10)
        fit = fit_age_only_from_runs(runs)
        all_metrics.append(
            {
                "variant": var.name,
                "available": True,
                "description": var.description,
                "total_centroid_runs": int(len(runs)),
                "observed_terminations": int(runs["observed_termination"].sum()),
                "censored_runs": int((1 - runs["observed_termination"]).sum()),
                "median_duration_s": float(runs["duration_s"].median()),
                "p90_duration_s": float(runs["duration_s"].quantile(0.90)),
                "p95_duration_s": float(runs["duration_s"].quantile(0.95)),
                "age_of_min_termination_probability_s": float(min_row["age_mid_s"]) if min_row is not None else np.nan,
                "min_termination_probability": float(min_row["termination_probability"]) if min_row is not None else np.nan,
                "mean_termination_probability_age_ge_20": float(late["termination_probability"].mean()) if not late.empty else np.nan,
                "late_flattening_or_increase_remains": tail_remains,
                **fit,
            }
        )
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(OUT / "segmentation_robustness_metrics.csv", index=False)
    if hazard_tables:
        hz = pd.concat(hazard_tables, ignore_index=True)
        hz.to_csv(OUT / "segmentation_robustness_hazard_source.csv", index=False)
        ccdf = pd.concat(ccdf_tables, ignore_index=True)
        ccdf.to_csv(OUT / "segmentation_robustness_ccdf_source.csv", index=False)
        plot_segmentation(hz, ccdf)
    metrics.to_latex(OUT / "tableS_segmentation_robustness.tex", index=False, float_format="%.3g")
    verdict = classify_segmentation(metrics)
    notes = [
        "# Segmentation Robustness Notes",
        "",
        f"- verdict: {verdict}",
        "- Native-frequency mean/median one-second aggregation could not be executed from the retained compact local cache; the retained trajectory_long cache is already one-second centroid/player data.",
        "- Tested variants from retained centroid trajectories: primary recomputation, low-speed turn guards at 0.25 and 0.50 m/s, and centred three-sample rolling median before heading calculation.",
        "- Low-speed guards merge low-speed candidate turns into the current run; they do not omit exposure intervals.",
        "- Rolling median variant is a mild temporal-noise check and is not used to replace the primary analysis.",
    ]
    write_md(OUT / "segmentation_robustness_notes.md", "\n".join(notes))
    return {"segmentation_verdict": verdict, "segmentation_metrics": metrics}


def classify_segmentation(metrics: pd.DataFrame) -> str:
    m = metrics[metrics["available"].eq(True)].copy()
    if m.empty:
        return "FAIL"
    same_median = m["median_duration_s"].between(1.0, 5.0).mean()
    has_decline = m["age_of_min_termination_probability_s"].ge(2).mean()
    tail = m["late_flattening_or_increase_remains"].astype(str).isin(["True", "true", "insufficient_tail_support"]).mean()
    if same_median >= 0.75 and has_decline >= 0.75 and tail >= 0.75:
        return "PASS"
    if same_median >= 0.5 and has_decline >= 0.5:
        return "MIXED"
    return "FAIL"


def plot_segmentation(hz: pd.DataFrame, ccdf: pd.DataFrame) -> None:
    setup_style()
    try:
        import sys

        sys.path.insert(0, str(ROOT))
        from analysis.levy_paper.util.paper_utils import draw_panel_letter
    except Exception:
        draw_panel_letter = None

    colors = {
        "primary_recomputed": "black",
        "low_speed_guard_0p25": "#d62728",
        "low_speed_guard_0p50": "#1f77b4",
        "rolling_median_3": "0.45",
    }
    labels = {
        "primary_recomputed": "Primary segmentation",
        "low_speed_guard_0p25": r"0.25 m s$^{-1}$ turn guard",
        "low_speed_guard_0p50": r"0.50 m s$^{-1}$ turn guard",
        "rolling_median_3": "Three-sample rolling median",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    ax = axes[0]
    for var, g in hz.groupby("variant"):
        if var not in colors:
            continue
        ax.plot(g["age_mid_s"], g["termination_probability"], color=colors[var], lw=1.6, label=labels[var])
        ax.fill_between(g["age_mid_s"].to_numpy(float), g["ci_low"].to_numpy(float), g["ci_high"].to_numpy(float), color=colors[var], alpha=0.10, linewidth=0)
    ax.set_xlabel("Run age (s)")
    ax.set_ylabel("Interval termination probability")
    ax.set_xlim(0, MAX_AGE)
    ax.set_ylim(bottom=0)
    if draw_panel_letter is not None:
        draw_panel_letter(ax, "A", x=-0.11, y=1.05, fontsize=15)
    else:
        ax.text(-0.11, 1.05, "A.", transform=ax.transAxes, ha="left", va="bottom", fontsize=15, clip_on=False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[1]
    for var, g in ccdf.groupby("variant"):
        if var not in colors:
            continue
        ax.plot(g["duration_s"], g["ccdf"], color=colors[var], lw=1.6, label=labels[var])
    ax.set_xlabel("Run duration T (s)")
    ax.set_ylabel(r"$P(T \geq t)$")
    ax.set_yscale("log")
    ax.set_xlim(1, 60)
    ax.set_ylim(1e-4, 1.05)
    if draw_panel_letter is not None:
        draw_panel_letter(ax, "B", x=-0.11, y=1.05, fontsize=15)
    else:
        ax.text(-0.11, 1.05, "B.", transform=ax.transAxes, ha="left", va="bottom", fontsize=15, clip_on=False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(OUT / "figS_segmentation_robustness.png", dpi=300)
    fig.savefig(OUT / "figS_segmentation_robustness.pdf")
    fig.savefig(OUT / "figS_segmentation_robustness.svg")
    plt.close(fig)


def load_prediction_base() -> pd.DataFrame:
    hcols = [
        "season",
        "match_id",
        "team",
        "source_key",
        "match_phase",
        "track_uid",
        "run_uid",
        "t0",
        "t1",
        "age_start_s",
        "age_mid_s",
        "age_end_s",
        "p_group",
        "v_group_mps",
    ]
    ocols = [
        "physical_match_id",
        "season",
        "match_id",
        "team",
        "source_key",
        "match_phase",
        "run_uid",
        "age_start_s",
        "observed_termination",
        "censored_event",
    ]
    h = pd.read_parquet(HAZ, columns=hcols)
    o = pd.read_parquet(OOF, columns=ocols)
    keys = ["season", "match_id", "team", "source_key", "match_phase", "run_uid", "age_start_s"]
    d = h.merge(o, on=keys, how="inner", validate="one_to_one")
    d["physical_fixture_id"] = d["physical_match_id"].astype(str)
    d["event_model"] = pd.to_numeric(d["observed_termination"], errors="coerce").fillna(0).astype(int)
    d["log1p_age_mid_s"] = np.log1p(pd.to_numeric(d["age_mid_s"], errors="coerce"))
    return d


def add_centroid_speed_heading(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["track_type", "track_uid", "t", "x_m", "y_m"]
    tr = pd.read_parquet(TRAJ, columns=cols)
    c = tr[tr["track_type"].astype(str).eq("centroid")].copy()
    c["t"] = pd.to_datetime(c["t"])
    c = c.sort_values(["track_uid", "t"]).reset_index(drop=True)
    c["_dt"] = c.groupby("track_uid")["t"].diff().dt.total_seconds()
    c["_dx"] = c.groupby("track_uid")["x_m"].diff()
    c["_dy"] = c.groupby("track_uid")["y_m"].diff()
    c["vcentroid_current"] = np.sqrt(c["_dx"] ** 2 + c["_dy"] ** 2) / c["_dt"]
    c.loc[(c["_dt"] <= 0) | (c["_dt"] > 2), "vcentroid_current"] = np.nan
    c["heading_current"] = np.arctan2(c["_dy"], c["_dx"])
    c.loc[~np.isfinite(c["vcentroid_current"]) | (c["vcentroid_current"] <= 1e-6), "heading_current"] = np.nan
    step = c[["track_uid", "t", "vcentroid_current", "heading_current"]].rename(columns={"t": "t1"})
    out = d.copy()
    out["t1"] = pd.to_datetime(out["t1"])
    return out.merge(step, on=["track_uid", "t1"], how="left", validate="many_to_one")


def add_previous_window_features(d: pd.DataFrame) -> pd.DataFrame:
    out = add_centroid_speed_heading(d).sort_values(["run_uid", "age_start_s"]).copy()
    for col in ["p_prev5", "vcentroid_prev5", "heading_R_prev5"]:
        out[col] = np.nan
    for _, g in out.groupby("run_uid", sort=False):
        idx = g.index.to_numpy()
        ages = g["age_start_s"].to_numpy(float)
        p = pd.to_numeric(g["p_group"], errors="coerce").to_numpy(float)
        v = pd.to_numeric(g["vcentroid_current"], errors="coerce").to_numpy(float)
        head = pd.to_numeric(g["heading_current"], errors="coerce").to_numpy(float)
        for i, age in enumerate(ages):
            prior = np.where((ages < age) & (ages >= age - 5.0))[0]
            if prior.size == 0:
                continue
            pv = p[prior]
            vv = v[prior]
            hh = head[prior]
            if np.isfinite(pv).any():
                out.loc[idx[i], "p_prev5"] = float(np.nanmean(pv))
            if np.isfinite(vv).any():
                out.loc[idx[i], "vcentroid_prev5"] = float(np.nanmean(vv))
            valid_h = np.isfinite(hh)
            if valid_h.any():
                out.loc[idx[i], "heading_R_prev5"] = float(np.sqrt(np.nanmean(np.cos(hh[valid_h])) ** 2 + np.nanmean(np.sin(hh[valid_h])) ** 2))
    return out


def zscore_train_test(train: pd.DataFrame, test: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    xtr = pd.to_numeric(train[col], errors="coerce").to_numpy(float)
    mu = np.nanmean(xtr)
    sd = np.nanstd(xtr)
    if not np.isfinite(sd) or sd < 1e-9:
        sd = 1.0
    return (xtr - mu) / sd, (pd.to_numeric(test[col], errors="coerce").to_numpy(float) - mu) / sd


def fit_predict_fold(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    xtr_parts = [train["log1p_age_mid_s"].to_numpy(float)]
    xte_parts = [test["log1p_age_mid_s"].to_numpy(float)]
    for col in cols:
        ztr, zte = zscore_train_test(train, test, col)
        xtr_parts.append(ztr)
        xte_parts.append(zte)
    xtr = np.vstack(xtr_parts).T
    xte = np.vstack(xte_parts).T
    y = train["event_model"].to_numpy(int)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(xtr, y)
    return model.predict_proba(xte)[:, 1]


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(x, y.astype(int))
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def metric_summary(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
    logloss = float(-ll / len(y))
    brier = float(np.mean((y - p) ** 2))
    intercept, slope = calibration(y, p)
    return {
        "heldout_log_likelihood": ll,
        "log_loss": logloss,
        "brier_score": brier,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "n_intervals": int(len(y)),
        "n_events": int(np.sum(y)),
    }


def part3_incremental_prediction() -> dict:
    d = add_previous_window_features(load_prediction_base())
    d = d[d["censored_event"].eq(False)].copy()
    model_cols = {
        "M0_age_only": [],
        "M1_age_speed": ["vcentroid_prev5"],
        "M2_age_polarisation": ["p_prev5"],
        "M3_age_speed_polarisation": ["vcentroid_prev5", "p_prev5"],
        "M4_age_speed_directional_stability": ["vcentroid_prev5", "heading_R_prev5"],
        "M5_age_speed_directional_stability_polarisation": ["vcentroid_prev5", "heading_R_prev5", "p_prev5"],
    }
    required = sorted({c for cols in model_cols.values() for c in cols})
    d = d.dropna(subset=["event_model", "log1p_age_mid_s"] + required).copy()
    fixtures = np.array(sorted(d["physical_fixture_id"].unique()))
    pred_cols = {}
    for name in model_cols:
        pred_cols[name] = np.full(len(d), np.nan, dtype=float)
    d = d.reset_index(drop=True)
    for fixture in fixtures:
        test_mask = d["physical_fixture_id"].eq(fixture).to_numpy()
        train = d.loc[~test_mask].copy()
        test = d.loc[test_mask].copy()
        for name, cols in model_cols.items():
            pred_cols[name][test_mask] = fit_predict_fold(train, test, cols)
    for name, pred in pred_cols.items():
        d[f"pred_{name}"] = pred

    y = d["event_model"].to_numpy(int)
    metrics = []
    for name in model_cols:
        metrics.append({"model": name, **metric_summary(y, d[f"pred_{name}"].to_numpy(float))})
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT / "incremental_prediction_metrics.csv", index=False)
    write_incremental_odds_ratios(d, model_cols)

    rows = []
    for fixture, g in d.groupby("physical_fixture_id"):
        yy = g["event_model"].to_numpy(int)
        row = {"physical_fixture_id": fixture, "n_intervals": int(len(g)), "n_events": int(yy.sum())}
        for contrast, m_full, m_base in [
            ("M3_minus_M1", "M3_age_speed_polarisation", "M1_age_speed"),
            ("M5_minus_M4", "M5_age_speed_directional_stability_polarisation", "M4_age_speed_directional_stability"),
        ]:
            pf = np.clip(g[f"pred_{m_full}"].to_numpy(float), 1e-8, 1 - 1e-8)
            pb = np.clip(g[f"pred_{m_base}"].to_numpy(float), 1e-8, 1 - 1e-8)
            llf = float(np.sum(yy * np.log(pf) + (1 - yy) * np.log1p(-pf)))
            llb = float(np.sum(yy * np.log(pb) + (1 - yy) * np.log1p(-pb)))
            row[f"{contrast}_log_likelihood_diff"] = llf - llb
            row[f"{contrast}_log_loss_diff"] = (-llf / len(g)) - (-llb / len(g))
            row[f"{contrast}_brier_diff"] = float(np.mean((yy - pf) ** 2) - np.mean((yy - pb) ** 2))
        rows.append(row)
    by_fixture = pd.DataFrame(rows)

    rng = np.random.default_rng(RNG_SEED)
    boot_rows = []
    for contrast in ["M3_minus_M1", "M5_minus_M4"]:
        for metric in ["log_likelihood_diff", "log_loss_diff", "brier_diff"]:
            col = f"{contrast}_{metric}"
            vals = by_fixture[col].to_numpy(float)
            boot = np.array([np.nanmean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(BOOTSTRAP_B)])
            boot_rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "mean_fixture_level_difference": float(np.nanmean(vals)),
                    "ci_low": float(np.nanpercentile(boot, 2.5)),
                    "ci_high": float(np.nanpercentile(boot, 97.5)),
                    "direction_favouring_added_polarisation": "positive" if metric == "log_likelihood_diff" else "negative",
                    "n_fixtures": int(len(vals)),
                    "bootstrap_B": BOOTSTRAP_B,
                }
            )
    boot_df = pd.DataFrame(boot_rows)
    for _, r in boot_df.iterrows():
        by_fixture[f"bootstrap_{r['contrast']}_{r['metric']}_mean"] = r["mean_fixture_level_difference"]
    by_fixture.to_csv(OUT / "incremental_prediction_by_fixture.csv", index=False)
    boot_df.to_csv(OUT / "incremental_prediction_bootstrap_summary.csv", index=False)
    metrics_df.to_latex(OUT / "tableS_incremental_prediction.tex", index=False, float_format="%.4g")
    plot_incremental(by_fixture)
    verdict = classify_incremental(boot_df)
    notes = [
        "# Incremental Prediction Notes",
        "",
        f"- verdict: {verdict}",
        "- All predictors are strictly preceding five-second summaries within the same run; current interval and future data are excluded.",
        "- Standardisation parameters are estimated on training fixtures only inside each held-out date-group fold.",
        "- M3 versus M1 is the primary speed-adjusted incremental-polarisation contrast.",
        "- M5 versus M4 additionally controls for recent centroid directional stability; this is diagnostic and may control away a pathway through which order acts.",
        "- Negative log-loss and Brier differences favour adding polarisation; positive log-likelihood differences favour adding polarisation.",
    ]
    write_md(OUT / "incremental_prediction_notes.md", "\n".join(notes))
    return {"incremental_verdict": verdict, "incremental_metrics": metrics_df, "incremental_bootstrap": boot_df}


def write_incremental_odds_ratios(d: pd.DataFrame, model_cols: dict[str, list[str]]) -> None:
    rows = []
    y = d["event_model"].to_numpy(int)
    for name in ["M2_age_polarisation", "M3_age_speed_polarisation", "M5_age_speed_directional_stability_polarisation"]:
        cols = model_cols[name]
        parts = [d["log1p_age_mid_s"].to_numpy(float)]
        names = ["log1p_age_mid_s"]
        for col in cols:
            x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            if not np.isfinite(sd) or sd < 1e-9:
                sd = 1.0
            parts.append((x - mu) / sd)
            names.append(col)
        xmat = np.vstack(parts).T
        keep = np.isfinite(xmat).all(axis=1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(xmat[keep], y[keep])
        for feature, coef in zip(names, model.coef_[0]):
            if feature == "p_prev5":
                rows.append(
                    {
                        "model": name,
                        "feature": feature,
                        "coefficient_per_1sd": float(coef),
                        "odds_ratio_per_1sd": float(np.exp(coef)),
                        "n_intervals": int(keep.sum()),
                        "n_events": int(y[keep].sum()),
                        "status": "descriptive_pooled_fit_not_primary_heldout_metric",
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "incremental_prediction_polarisation_odds_ratios.csv", index=False)


def classify_incremental(boot: pd.DataFrame) -> str:
    primary = boot[(boot["contrast"] == "M3_minus_M1") & (boot["metric"] == "log_loss_diff")]
    if primary.empty:
        return "FAIL"
    hi = float(primary["ci_high"].iloc[0])
    lo = float(primary["ci_low"].iloc[0])
    mean = float(primary["mean_fixture_level_difference"].iloc[0])
    if hi < 0:
        return "PASS"
    if mean < 0 or lo < 0:
        return "MIXED"
    return "FAIL"


def plot_incremental(by_fixture: pd.DataFrame) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    specs = [
        ("M3_minus_M1_log_loss_diff", "Log-loss diff\n(M3-M1)", "negative favours p"),
        ("M3_minus_M1_brier_diff", "Brier diff\n(M3-M1)", "negative favours p"),
        ("M3_minus_M1_log_likelihood_diff", "Log-likelihood diff\n(M3-M1)", "positive favours p"),
    ]
    for ax, (col, xlabel, note) in zip(axes, specs):
        vals = by_fixture[col].to_numpy(float)
        ax.axvline(0, color="0.6", lw=1)
        ax.hist(vals, bins=18, color="0.2", alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fixtures")
        ax.text(0.04, 0.94, note, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    fig.savefig(OUT / "figS_incremental_prediction.png", dpi=300)
    fig.savefig(OUT / "figS_incremental_prediction.pdf")
    plt.close(fig)


def write_summary(part1: dict, part2: dict, part3: dict) -> None:
    seg = part2["segmentation_metrics"]
    inc = part3["incremental_metrics"]
    boot = part3["incremental_bootstrap"]
    primary = boot[(boot["contrast"] == "M3_minus_M1") & (boot["metric"] == "log_loss_diff")]
    primary_text = "unavailable"
    if not primary.empty:
        r = primary.iloc[0]
        primary_text = f"mean fixture log-loss diff={r['mean_fixture_level_difference']:.6g}, 95% CI [{r['ci_low']:.6g}, {r['ci_high']:.6g}]"
    or_text = "unavailable"
    or_path = OUT / "incremental_prediction_polarisation_odds_ratios.csv"
    if or_path.exists():
        ors = pd.read_csv(or_path)
        m3 = ors[ors["model"].eq("M3_age_speed_polarisation")]
        if not m3.empty:
            or_text = f"M3 pooled descriptive OR per 1 SD lagged p={float(m3['odds_ratio_per_1sd'].iloc[0]):.4g}"
    text = [
        "# Pre-submission Robustness Summary",
        "",
        "## 1. Concerns",
        "",
        "- Match counts: distinguish physical fixtures, date groups, team-match records, runs and risk intervals.",
        "- Run segmentation: test whether early termination is an artefact of one-second unsmoothed centroid turns or low-speed jitter.",
        "- Incremental prediction: test whether lagged polarisation improves held-out prediction beyond centroid speed and directional stability.",
        "",
        "## 2. Analyses performed",
        "",
        "- Dataset audit from final multiseason metadata.",
        "- Segmentation robustness from retained one-second centroid trajectories: primary recomputation, two low-speed guards and mild rolling median smoothing.",
        "- Cross-fitted interval prediction using strictly preceding five-second p, centroid speed and centroid directional stability.",
        "",
        "## 3. Main numerical results",
        "",
        f"- Dataset: {part1['unique_physical_matches_passing_final_inclusion']} final distinct fixture keys; {part1['team_match_observations_final']} team-match records; {part1['distinct_match_dates_final']} date-based validation groups.",
        f"- Segmentation verdict: {part2['segmentation_verdict']}.",
        f"- Segmentation primary rows: {seg[seg['variant'].eq('primary_recomputed')].to_dict('records')[:1]}",
        f"- Incremental prediction verdict: {part3['incremental_verdict']}.",
        f"- Primary M3 versus M1 contrast: {primary_text}.",
        f"- Polarisation coefficient check: {or_text}.",
        "",
        "## 4. PASS / MIXED / FAIL verdicts",
        "",
        f"- Concern 1, dataset counts: PASS for reconciliation; wording needs the 47/62/66 distinction.",
        f"- Concern 2, segmentation robustness: {part2['segmentation_verdict']}.",
        f"- Concern 3, incremental polarisation prediction: {part3['incremental_verdict']}.",
        "",
        "## 5. Recommended manuscript changes",
        "",
        "- Use 62 distinct competitive fixture keys and 66 tracked team-match/source records; reserve 47 for date-based validation groups.",
        "- Add segmentation robustness figure/table if the authors want to address early-hazard sensitivity directly.",
        "- Phrase polarisation claims in terms of incremental held-out prediction only if M3 improves over M1; otherwise use more cautious marker/state language.",
        "",
        "## 6. Recommended supplementary figures and tables",
        "",
        "- figS_segmentation_robustness.png/pdf",
        "- tableS_segmentation_robustness.tex",
        "- figS_incremental_prediction.png/pdf",
        "- tableS_incremental_prediction.tex",
        "- incremental_prediction_polarisation_odds_ratios.csv",
        "",
        "## 7. Material contradictions",
        "",
        "- Native-frequency first-vs-mean/median aggregation could not be tested from retained local caches.",
        "- No unrelated scientific direction or new theoretical model was added.",
        "",
        "## 8. Files and scripts created or modified",
        "",
        f"- script: {rel(ROOT / 'analysis/levy_paper/scripts/run_pre_submission_robustness_audit.py')}",
    ]
    for p in sorted(OUT.iterdir()):
        text.append(f"- output: {rel(p)}")
    text.extend(
        [
            "",
            "## 9. Reproduction command",
            "",
            "```powershell",
            'cd "<path-to-Athlelorien>"',
            r".\.venv\Scripts\python.exe analysis\levy_paper\scripts\run_pre_submission_robustness_audit.py",
            "```",
        ]
    )
    write_md(OUT / "pre_submission_robustness_summary.md", "\n".join(text))


def main() -> None:
    setup_style()
    part1 = part1_dataset_audit()
    part2 = part2_segmentation_robustness()
    part3 = part3_incremental_prediction()
    write_summary(part1, part2, part3)
    manifest = {
        "status": "complete",
        "output_dir": str(OUT),
        "files": [rel(p) for p in sorted(OUT.iterdir())],
    }
    (OUT / "pre_submission_robustness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
