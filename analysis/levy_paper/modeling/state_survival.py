"""Cross-fitted state-structured survival model used by Figures 4 and 5.

This module starts from the one-second centroid-run interval table.  It contains
no AWS access and therefore forms the reproducible processed-cache layer between
the raw tracking pipeline and the publication figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar


STATES = ("low", "mid", "high")
STATE_TO_INDEX = {state: i for i, state in enumerate(STATES)}
AGE_BANDS = ((0, 5, "0-5"), (5, 10, "5-10"), (10, 20, "10-20"), (20, 35, "20-35"))


@dataclass(frozen=True)
class CrossfitConfig:
    """Frozen modeling choices used in the publication analysis."""

    max_fit_age_s: float = 35.0
    late_tc_s: float = 20.0
    late_tau_s: float = 5.0
    max_curve_age_s: int = 35
    bootstrap_replicates: int = 500
    random_seed: int = 20260724


def _physical_match_id(frame: pd.DataFrame) -> pd.Series:
    return frame["season"].astype(str) + "__" + frame["match_id"].astype(str)


def _late_integral(age: np.ndarray, tc: float, tau: float) -> np.ndarray:
    x = np.maximum(np.asarray(age, dtype=float) - tc, 0.0)
    return x - tau * (1.0 - np.exp(-x / tau))


def _age_exposure(age_start: np.ndarray, age_end: np.ndarray, mu: float, a0: float) -> np.ndarray:
    return mu * np.log((a0 + np.asarray(age_end, dtype=float)) / (a0 + np.asarray(age_start, dtype=float)))


def _probability_from_exposure(exposure: np.ndarray) -> np.ndarray:
    return -np.expm1(-np.clip(np.asarray(exposure, dtype=float), 0.0, 700.0))


def _bernoulli_log_likelihood(event: np.ndarray, probability: np.ndarray) -> float:
    event = np.asarray(event, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(np.sum(event * np.log(probability) + (1.0 - event) * np.log1p(-probability)))


def _best_minimize(objective, starts: list[np.ndarray], bounds: list[tuple[float, float]]):
    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds) for start in starts]
    return min(results, key=lambda result: float(result.fun))


def _fit_age_only(age_start: np.ndarray, age_end: np.ndarray, event: np.ndarray) -> dict[str, float | bool | str]:
    def objective(theta: np.ndarray) -> float:
        mu, a0 = np.exp(theta)
        return -_bernoulli_log_likelihood(event, _probability_from_exposure(_age_exposure(age_start, age_end, mu, a0)))

    result = _best_minimize(
        objective,
        [np.log([1.2, 2.5]), np.log([0.7, 1.0]), np.log([2.0, 5.0])],
        [(-8.0, 4.0), (-8.0, 7.0)],
    )
    mu, a0 = np.exp(result.x)
    return {"mu": float(mu), "a0": float(a0), "log_likelihood": -float(result.fun), "success": bool(result.success), "message": str(result.message)}


def _fit_age_order(
    age_start: np.ndarray,
    age_end: np.ndarray,
    event: np.ndarray,
    state_index: np.ndarray,
    age_only: dict[str, float | bool | str],
) -> dict[str, float | bool | str]:
    def objective(theta: np.ndarray) -> float:
        mu, a0, phi_low, phi_high = np.exp(theta)
        multipliers = np.array([phi_low, 1.0, phi_high])[state_index]
        exposure = _age_exposure(age_start, age_end, mu, a0) * multipliers
        return -_bernoulli_log_likelihood(event, _probability_from_exposure(exposure))

    initial = np.log([float(age_only["mu"]), float(age_only["a0"]), 1.2, 0.7])
    result = _best_minimize(
        objective,
        [initial, np.log([1.5, 3.5, 1.0, 1.0]), np.log([2.0, 5.0, 1.4, 0.6])],
        [(-8.0, 4.0), (-8.0, 7.0), (-5.0, 5.0), (-5.0, 5.0)],
    )
    mu, a0, phi_low, phi_high = np.exp(result.x)
    return {
        "mu": float(mu),
        "a0": float(a0),
        "phi_low": float(phi_low),
        "phi_mid": 1.0,
        "phi_high": float(phi_high),
        "log_likelihood": -float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
    }


def _fit_late_q(
    age_start: np.ndarray,
    age_end: np.ndarray,
    event: np.ndarray,
    state_index: np.ndarray,
    age_order: dict[str, float | bool | str],
    tc: float,
    tau: float,
) -> dict[str, float | bool | str]:
    phi = np.array([age_order["phi_low"], 1.0, age_order["phi_high"]], dtype=float)[state_index]
    base = _age_exposure(age_start, age_end, float(age_order["mu"]), float(age_order["a0"])) * phi
    late = _late_integral(age_end, tc, tau) - _late_integral(age_start, tc, tau)

    def objective(q: float) -> float:
        return -_bernoulli_log_likelihood(event, _probability_from_exposure(base + float(q) * late))

    result = minimize_scalar(objective, bounds=(0.0, 2.0), method="bounded", options={"xatol": 1e-10})
    return {
        **age_order,
        "q": float(result.x),
        "tc": float(tc),
        "tau_q": float(tau),
        "log_likelihood": -float(result.fun),
        "success": bool(result.success),
        "message": f"{result.message}; late residual fitted on training-only age+order base",
    }


def _state_from_thresholds(values: pd.Series, teams: pd.Series, thresholds: dict[str, tuple[float, float]], global_thresholds: tuple[float, float]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    for team in pd.unique(teams.astype(str)):
        mask = teams.astype(str).to_numpy() == team
        q1, q2 = thresholds.get(team, global_thresholds)
        result[mask] = np.where(value_array[mask] < q1, "low", np.where(value_array[mask] >= q2, "high", "mid"))
    return result


def _prepare_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season", "match_id", "team", "source_key", "match_phase", "run_uid", "age_start_s",
        "age_end_s", "dt_s", "event", "run_duration_s", "p_group", "t1",
    }
    missing = sorted(required.difference(intervals.columns))
    if missing:
        raise ValueError(f"Hazard interval cache is missing columns: {missing}")

    frame = intervals.copy()
    frame["physical_match_id"] = _physical_match_id(frame)
    frame["fixture_id"] = frame["physical_match_id"]
    frame["match_team_half_id"] = (
        frame["physical_match_id"] + "__" + frame["team"].astype(str) + "__" + frame["source_key"].astype(str) + "__" + frame["match_phase"].astype(str)
    )
    frame["observed_termination"] = pd.to_numeric(frame["event"], errors="coerce").fillna(0).astype(np.int8)
    frame["censored_event"] = False

    # The last observed run endpoint in each tracked team-half is administrative.
    record_cols = ["season", "match_id", "team", "source_key", "match_phase"]
    terminal = frame.groupby(record_cols, observed=True)["t1"].idxmax()
    frame.loc[terminal, "observed_termination"] = 0
    frame.loc[terminal, "censored_event"] = True
    frame["age_bin_1s"] = np.floor(pd.to_numeric(frame["age_start_s"], errors="coerce")).astype(int)
    return frame


def _transition_rows(
    train: pd.DataFrame,
    heldout: str,
    state_definition: str = "S0_raw",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    transitions = train.loc[train["next_outcome"].isin(STATES) & train["age_start_s"].lt(35)].copy()
    rows: list[dict] = []
    matrices: dict[str, np.ndarray] = {}
    for kind in ("stationary", "age_banded"):
        bands = AGE_BANDS if kind == "age_banded" else ((0, 35, "0-5"),)
        for left, right, label in bands:
            part = transitions if kind == "stationary" else transitions.loc[transitions["age_start_s"].ge(left) & transitions["age_start_s"].lt(right)]
            matrix = np.zeros((3, 3), dtype=float)
            counts = pd.crosstab(part["observed_state"], part["next_outcome"]).reindex(index=STATES, columns=STATES, fill_value=0)
            for i, from_state in enumerate(STATES):
                total = float(counts.loc[from_state].sum())
                matrix[i] = counts.loc[from_state].to_numpy(float) / total if total else np.eye(3)[i]
            target_labels = [band[2] for band in AGE_BANDS] if kind == "stationary" else [label]
            for target_label in target_labels:
                matrices[f"{kind}:{target_label}"] = matrix.copy()
                for i, from_state in enumerate(STATES):
                    for j, to_state in enumerate(STATES):
                        rows.append(
                            {
                                "heldout_physical_match_id": heldout,
                                "transition_kind": kind,
                                "age_band": target_label,
                                "from_state": from_state,
                                "to_state": to_state,
                                "probability": matrix[i, j],
                                "train_count": float(counts.iloc[i, j]) if kind == "age_banded" else np.nan,
                                "alpha": 0.0,
                            }
                        )
            if kind == "stationary":
                break
    return pd.DataFrame(rows), matrices


def _band_label(age: int) -> str:
    for left, right, label in AGE_BANDS:
        if left <= age < right:
            return label
    return AGE_BANDS[-1][2]


def _interval_death_probabilities(age: int, model: dict[str, float | bool | str], state_specific: bool, include_late: bool = True) -> np.ndarray:
    start = np.array([float(age)] * 3)
    end = start + 1.0
    exposure = _age_exposure(start, end, float(model["mu"]), float(model["a0"]))
    if state_specific:
        exposure *= np.array([model["phi_low"], 1.0, model["phi_high"]], dtype=float)
    if include_late:
        late = _late_integral(end, float(model.get("tc", 20.0)), float(model.get("tau_q", 5.0))) - _late_integral(start, float(model.get("tc", 20.0)), float(model.get("tau_q", 5.0)))
        exposure += float(model.get("q", 0.0)) * late
    return _probability_from_exposure(exposure)


def _operator_curve(
    pi0: np.ndarray,
    matrices: dict[str, np.ndarray],
    full_model: dict[str, float | bool | str],
    age_only: dict[str, float | bool | str],
    model_name: str,
    max_age: int,
) -> pd.DataFrame:
    mass = np.asarray(pi0, dtype=float).copy()
    history = np.zeros((3, 3), dtype=float)
    rows: list[dict] = []
    for age in range(max_age + 1):
        survival = float(mass.sum())
        current = mass / survival if survival else np.full(3, np.nan)
        exposure = history.sum(axis=0) / (age * survival) if age > 0 and survival else np.full(3, np.nan)
        if model_name == "transitions only":
            death = _interval_death_probabilities(age, {**age_only, "q": full_model["q"], "tc": full_model["tc"], "tau_q": full_model["tau_q"]}, False)
        else:
            death = _interval_death_probabilities(age, full_model, True)
        hazard = float(np.dot(current, death)) if survival else np.nan
        rows.append(
            {
                "age_s": float(age), "survival": survival, "hazard": hazard,
                **{f"current_{state}": current[i] for i, state in enumerate(STATES)},
                **{f"exposure_{state}": exposure[i] for i, state in enumerate(STATES)},
                **{f"death_{state}": death[i] for i, state in enumerate(STATES)},
                "model_name": model_name, "state_definition": "S0_raw",
            }
        )
        if age == max_age:
            continue
        surviving_mass = mass * (1.0 - death)
        surviving_history = history * (1.0 - death)[:, None]
        surviving_history[np.arange(3), np.arange(3)] += surviving_mass
        if model_name == "stationary generative model":
            transition = matrices[f"stationary:{_band_label(age)}"]
        elif model_name == "differential termination only":
            transition = np.tile(pi0, (3, 1))
        else:
            transition = matrices[f"age_banded:{_band_label(age)}"]
        mass = surviving_mass @ transition
        history = transition.T @ surviving_history
    return pd.DataFrame(rows)


def _bootstrap_weighted_curves(
    fixture_curves: pd.DataFrame,
    value_col: str,
    x_col: str,
    weight_col: str,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    fixtures = fixture_curves["heldout_physical_match_id"].drop_duplicates().to_numpy()
    point_rows = []
    replicate_rows = []
    for x, group in fixture_curves.groupby(x_col, observed=True):
        values = group.set_index("heldout_physical_match_id")
        mask = values[value_col].notna() & values[weight_col].gt(0)
        values = values.loc[mask]
        estimate = np.average(values[value_col], weights=values[weight_col]) if len(values) else np.nan
        point_rows.append({x_col: x, "estimate": estimate})
    for _ in range(replicates):
        sampled = rng.choice(fixtures, len(fixtures), replace=True)
        multiplicity = pd.Series(sampled).value_counts()
        boot = fixture_curves.merge(multiplicity.rename("_mult"), left_on="heldout_physical_match_id", right_index=True, how="inner")
        boot["_weight"] = boot[weight_col] * boot["_mult"]
        for x, group in boot.groupby(x_col, observed=True):
            mask = group[value_col].notna() & group["_weight"].gt(0)
            group = group.loc[mask]
            if len(group):
                replicate_rows.append({x_col: x, "estimate": np.average(group[value_col], weights=group["_weight"])})
    points = pd.DataFrame(point_rows)
    if not replicate_rows:
        points["ci_low"] = np.nan
        points["ci_high"] = np.nan
        return points
    ci = pd.DataFrame(replicate_rows).groupby(x_col)["estimate"].quantile([0.025, 0.975]).unstack().reset_index().rename(columns={0.025: "ci_low", 0.975: "ci_high"})
    return points.merge(ci, on=x_col, how="left")


def _empirical_fixture_curves(oof: pd.DataFrame, max_age: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_rows: list[dict] = []
    survival_rows: list[dict] = []
    exposure_rows: list[dict] = []
    for fixture, group in oof.groupby("physical_match_id", observed=True):
        run_durations = group.groupby("run_uid", observed=True)["run_duration_s"].first()
        n_runs = int(len(run_durations))
        for age in range(max_age + 1):
            at_age = group.loc[group["age_bin_1s"].eq(age)]
            current_rows.append(
                {"heldout_physical_match_id": fixture, "age_s": age, "current_high": float(at_age["observed_state"].eq("high").mean()) if len(at_age) else np.nan, "n_at_risk": len(at_age), "heldout_n_runs": n_runs}
            )
            survival_rows.append(
                {"heldout_physical_match_id": fixture, "age_s": age, "survival": float(run_durations.ge(age).mean()), "heldout_n_runs": n_runs}
            )
        base = group[["run_uid", "run_duration_s", "age_start_s", "observed_state"]]
        for threshold in range(1, max_age + 1):
            selected = base.loc[base["run_duration_s"].ge(threshold) & base["age_start_s"].lt(threshold)]
            if selected.empty:
                estimate = np.nan
                n_survivors = 0
            else:
                per_run = selected.assign(is_high=selected["observed_state"].eq("high")).groupby("run_uid", observed=True)["is_high"].mean()
                estimate = float(per_run.mean())
                n_survivors = int(len(per_run))
            exposure_rows.append(
                {"heldout_physical_match_id": fixture, "threshold_s": threshold, "exposure_high": estimate, "n_survivors": n_survivors, "heldout_n_runs": n_runs}
            )
    return pd.DataFrame(current_rows), pd.DataFrame(survival_rows), pd.DataFrame(exposure_rows)


def _build_figure_sources(
    oof: pd.DataFrame,
    fold_curves: pd.DataFrame,
    config: CrossfitConfig,
    output_dir: Path,
) -> None:
    rng = np.random.default_rng(config.random_seed)
    max_age = config.max_curve_age_s
    empirical_current, empirical_survival, empirical_exposure = _empirical_fixture_curves(oof, max_age)

    def model_curve(model_name: str, x_col: str, value_col: str, output_x: str) -> pd.DataFrame:
        part = fold_curves.loc[fold_curves["model_name"].eq(model_name)].copy()
        result = _bootstrap_weighted_curves(part, value_col, x_col, "heldout_n_runs", config.bootstrap_replicates, rng)
        return result.rename(columns={x_col: output_x})

    f5b_parts = []
    empirical = _bootstrap_weighted_curves(empirical_current, "current_high", "age_s", "n_at_risk", config.bootstrap_replicates, rng)
    empirical["model_name"] = "held-out empirical"
    f5b_parts.append(empirical)
    for name in ("stationary generative model", "age-banded generative model"):
        part = model_curve(name, "age_s", "current_high", "age_s")
        part["model_name"] = name
        f5b_parts.append(part)
    f5b = pd.concat(f5b_parts, ignore_index=True)
    f5b["quantity"] = "current_high_fraction"
    f5b.to_csv(output_dir / "figure5_crossfitted_panelB_current_high_fraction.csv", index=False)

    f5c_parts = []
    empirical = _bootstrap_weighted_curves(empirical_exposure, "exposure_high", "threshold_s", "n_survivors", config.bootstrap_replicates, rng)
    empirical["model_name"] = "held-out empirical"
    f5c_parts.append(empirical)
    for name in ("full generative model", "transitions only", "differential termination only"):
        part = model_curve(name, "age_s", "exposure_high", "threshold_s").loc[lambda d: d["threshold_s"].ge(1)]
        part["model_name"] = name
        f5c_parts.append(part)
    f5c = pd.concat(f5c_parts, ignore_index=True)
    f5c["quantity"] = "high_order_exposure"
    f5c.to_csv(output_dir / "figure5_crossfitted_panelD_high_exposure_counterfactuals.csv", index=False)

    survival_parts = []
    empirical = _bootstrap_weighted_curves(empirical_survival, "survival", "age_s", "heldout_n_runs", config.bootstrap_replicates, rng)
    empirical["model_name"] = "held-out empirical"
    survival_parts.append(empirical)
    for name in ("stationary generative model", "age-banded generative model"):
        part = model_curve(name, "age_s", "survival", "age_s")
        part["model_name"] = name
        survival_parts.append(part)
    observed_path = []
    for age in range(max_age + 1):
        if age == 0:
            estimate = 1.0
        else:
            probabilities = oof.loc[oof["age_bin_1s"].lt(age)].groupby("age_bin_1s")["pred_full_hazard_termination_probability"].mean()
            estimate = float(np.prod(1.0 - probabilities.to_numpy(float)))
        observed_path.append({"age_s": age, "estimate": estimate, "ci_low": np.nan, "ci_high": np.nan, "model_name": "observed-path hazard benchmark"})
    survival = pd.concat([*survival_parts, pd.DataFrame(observed_path)], ignore_index=True)
    survival["quantity"] = "survival"
    survival.to_csv(output_dir / "figure5_crossfitted_panelC_survival.csv", index=False)

    # Pooled, survival-conditioned transitions for the compact Figure 5A selection.
    selected_pairs = (("high", "high"), ("high", "mid"), ("low", "low"), ("mid", "high"))
    transition_rows = []
    transition_data = oof.loc[oof["next_outcome"].isin(STATES) & oof["age_start_s"].lt(max_age)].copy()
    for left, right, label in AGE_BANDS:
        band = transition_data.loc[transition_data["age_start_s"].ge(left) & transition_data["age_start_s"].lt(right)]
        for current, nxt in selected_pairs:
            origin = band.loc[band["observed_state"].eq(current)]
            point = float(origin["next_outcome"].eq(nxt).mean()) if len(origin) else np.nan
            cluster = origin.groupby("physical_match_id")["next_outcome"].agg(n="size", k=lambda s: int(s.eq(nxt).sum())).reset_index()
            reps = []
            fixtures = cluster["physical_match_id"].to_numpy()
            if len(fixtures):
                for _ in range(config.bootstrap_replicates):
                    sampled = pd.Series(rng.choice(fixtures, len(fixtures), replace=True)).value_counts()
                    joined = cluster.merge(sampled.rename("w"), left_on="physical_match_id", right_index=True)
                    reps.append(float((joined["k"] * joined["w"]).sum() / (joined["n"] * joined["w"]).sum()))
            transition_rows.append(
                {
                    "age_band": label, "current_state": current.title(), "next_state": nxt.title(),
                    "transition_label": f"{current.title()} → {nxt.title()}", "point_estimate": point,
                    "lower_uncertainty_bound": float(np.quantile(reps, 0.025)) if reps else np.nan,
                    "upper_uncertainty_bound": float(np.quantile(reps, 0.975)) if reps else np.nan,
                    "bootstrap_median": float(np.median(reps)) if reps else np.nan,
                    "uncertainty_definition": "physical-match cluster bootstrap percentile 95% CI",
                    "state_definition": "S0_raw", "bootstrap_unit": "physical_match_id",
                    "n_bootstrap_replicates": config.bootstrap_replicates,
                    "n_transitions": int(origin["next_outcome"].eq(nxt).sum()), "n_origin_state_transitions": int(len(origin)),
                    "conditional_on_survival": True, "line_color": "#1f77b4" if current == "high" else ("#d62728" if current == "low" else "black"), "line_style": "solid",
                }
            )
    pd.DataFrame(transition_rows).to_csv(output_dir / "figure5_panelA_selected_transitions_crossfit_package.csv", index=False)

    # Figure 4 hazard source and the compact interval table used for recent-state Panel B.
    grouped = oof.groupby("age_bin_1s", observed=True).agg(
        n_at_risk=("observed_termination", "size"), n_terminated=("observed_termination", "sum"),
        age_only_model_probability=("pred_age_only_termination_probability", "mean"),
        age_order_model_probability=("pred_age_plus_order_termination_probability", "mean"),
        smooth_full_model_probability=("pred_full_hazard_termination_probability", "mean"),
    ).reset_index().rename(columns={"age_bin_1s": "_age_bin"})
    grouped["age_left"] = grouped["_age_bin"].astype(float)
    grouped["age_right"] = grouped["age_left"] + 1.0
    grouped["age_mid"] = grouped["age_left"] + 0.5
    grouped["empirical_interval_probability"] = grouped["n_terminated"] / grouped["n_at_risk"]
    grouped["ci_low"] = np.nan
    grouped["ci_high"] = np.nan
    grouped["hard_full_model_probability"] = grouped["smooth_full_model_probability"]
    grouped["smooth_late_interval_probability"] = np.maximum(grouped["smooth_full_model_probability"] - grouped["age_order_model_probability"], 0.0)
    grouped["full_model_probability"] = grouped["smooth_full_model_probability"]
    grouped["sparse_flag"] = grouped["n_at_risk"].lt(100) | grouped["n_terminated"].lt(5)
    grouped["fitted_flag"] = grouped["age_left"].lt(config.max_fit_age_s)
    grouped["late_fit_support_flag"] = grouped["fitted_flag"] & ~grouped["sparse_flag"]
    grouped["exclusion_reason"] = np.where(grouped["late_fit_support_flag"], "", "sparse_tail_observation_not_used_for_inference")
    grouped.to_csv(output_dir / "figure4_panelA_hazard_audit.csv", index=False)

def _build_validation_tables(oof: pd.DataFrame, fold_curves: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for label, column in (
        ("age-only hazard", "pred_age_only_termination_probability"),
        ("age+order hazard", "pred_age_plus_order_termination_probability"),
        ("full hazard", "pred_full_hazard_termination_probability"),
    ):
        probability = oof[column].to_numpy(float)
        event = oof["observed_termination"].to_numpy(float)
        rows.append(
            {
                "metric_family": "hazard", "model_name": label, "n_intervals": len(oof), "n_events": int(event.sum()),
                "heldout_log_likelihood": _bernoulli_log_likelihood(event, probability),
                "heldout_log_loss": -_bernoulli_log_likelihood(event, probability) / len(event),
                "brier_score": float(np.mean((event - probability) ** 2)), "calibration_slope": np.nan,
                "calibration_intercept": np.nan, "integrated_absolute_error": np.nan, "log_survival_rmse": np.nan, "n_points": np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "crossfitted_validation_metrics.csv", index=False)

    empirical_current, empirical_survival, _ = _empirical_fixture_curves(oof, int(fold_curves["age_s"].max()))
    fixture_rows = []
    for fixture, empirical in empirical_survival.groupby("heldout_physical_match_id", observed=True):
        empirical_high = empirical_current.loc[empirical_current["heldout_physical_match_id"].eq(fixture)]
        for model_name in ("stationary generative model", "age-banded generative model"):
            model = fold_curves.loc[fold_curves["heldout_physical_match_id"].eq(fixture) & fold_curves["model_name"].eq(model_name)]
            survival = empirical.merge(model[["age_s", "survival", "current_high"]], on="age_s", suffixes=("_emp", "_model"))
            high = empirical_high.merge(model[["age_s", "current_high"]], on="age_s", suffixes=("_emp", "_model"))
            fixture_rows.append(
                {
                    "fixture_id": fixture, "model_name": model_name, "n_runs": int(empirical["heldout_n_runs"].iloc[0]),
                    "survival_IAE": float(np.mean(np.abs(survival["survival_emp"] - survival["survival_model"]))),
                    "survival_log_RMSE": float(np.sqrt(np.mean((np.log(np.clip(survival["survival_emp"], 1e-12, 1)) - np.log(np.clip(survival["survival_model"], 1e-12, 1))) ** 2))),
                    "current_high_IAE": float(np.nanmean(np.abs(high["current_high_emp"] - high["current_high_model"]))),
                }
            )
    pd.DataFrame(fixture_rows).to_csv(output_dir / "crossfitted_fixture_level_curve_metrics.csv", index=False)


def build_crossfitted_state_survival_cache(
    interval_path: Path | str,
    output_dir: Path | str,
    config: CrossfitConfig | None = None,
) -> dict[str, int]:
    """Fit all leave-one-physical-fixture-out models and save reusable tables."""

    config = config or CrossfitConfig()
    interval_path = Path(interval_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals = _prepare_intervals(pd.read_parquet(interval_path))
    fixtures = sorted(intervals["physical_match_id"].unique())

    oof_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict] = []
    threshold_rows: list[dict] = []
    pi_rows: list[dict] = []
    transition_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    death_rows: list[dict] = []

    for fold_number, heldout in enumerate(fixtures, start=1):
        train = intervals.loc[~intervals["physical_match_id"].eq(heldout)].copy()
        test = intervals.loc[intervals["physical_match_id"].eq(heldout)].copy()
        global_q = tuple(train["p_group"].quantile([1 / 3, 2 / 3]).to_numpy(float))
        team_thresholds: dict[str, tuple[float, float]] = {}
        for team, group in train.groupby("team", observed=True):
            quantiles = tuple(group["p_group"].quantile([1 / 3, 2 / 3]).to_numpy(float))
            team_thresholds[str(team)] = quantiles
            threshold_rows.append({"heldout_physical_match_id": heldout, "state_definition": "S0_raw", "team": str(team), "q1": quantiles[0], "q2": quantiles[1]})
        threshold_rows.append({"heldout_physical_match_id": heldout, "state_definition": "S0_raw", "team": "__global__", "q1": global_q[0], "q2": global_q[1]})
        train["observed_state"] = _state_from_thresholds(train["p_group"], train["team"], team_thresholds, global_q)
        test["observed_state"] = _state_from_thresholds(test["p_group"], test["team"], team_thresholds, global_q)

        for frame in (train, test):
            frame.sort_values(["run_uid", "age_start_s"], inplace=True)
            frame["next_outcome"] = frame.groupby("run_uid", observed=True)["observed_state"].shift(-1)
            terminal_mask = frame["next_outcome"].isna()
            frame.loc[terminal_mask & frame["observed_termination"].eq(1), "next_outcome"] = "D"
            frame.loc[terminal_mask & frame["censored_event"], "next_outcome"] = "C"

        fit = train.loc[train["age_start_s"].lt(config.max_fit_age_s)]
        start = fit["age_start_s"].to_numpy(float)
        end = fit["age_end_s"].to_numpy(float)
        event = fit["observed_termination"].to_numpy(float)
        state_index = fit["observed_state"].map(STATE_TO_INDEX).to_numpy(int)
        age_only = _fit_age_only(start, end, event)
        age_order = _fit_age_order(start, end, event, state_index, age_only)
        full = _fit_late_q(start, end, event, state_index, age_order, config.late_tc_s, config.late_tau_s)

        models = (("age_only", age_only), ("age_order", age_order), ("full", full))
        for name, model in models:
            parameter_rows.append(
                {
                    "heldout_physical_match_id": heldout, "state_definition": "S0_raw", "hazard_model": name,
                    "log_likelihood_train": model["log_likelihood"], "n_fit_intervals": len(fit), "n_fit_deaths": int(event.sum()),
                    "optimizer_success": model["success"], "optimizer_message": model["message"],
                    "mu": model["mu"], "a0": model["a0"], "phi_low": model.get("phi_low", 1.0), "phi_mid": 1.0,
                    "phi_high": model.get("phi_high", 1.0), "q": model.get("q", 0.0), "tc": model.get("tc", 25.0), "tau_q": model.get("tau_q", 5.0),
                }
            )

        test_state_index = test["observed_state"].map(STATE_TO_INDEX).to_numpy(int)
        test_age = _age_exposure(test["age_start_s"].to_numpy(float), test["age_end_s"].to_numpy(float), float(age_only["mu"]), float(age_only["a0"]))
        test["pred_age_only_termination_probability"] = _probability_from_exposure(test_age)
        base_order = _age_exposure(test["age_start_s"].to_numpy(float), test["age_end_s"].to_numpy(float), float(age_order["mu"]), float(age_order["a0"]))
        base_order *= np.array([age_order["phi_low"], 1.0, age_order["phi_high"]], dtype=float)[test_state_index]
        test["pred_age_plus_order_termination_probability"] = _probability_from_exposure(base_order)
        late = _late_integral(test["age_end_s"].to_numpy(float), config.late_tc_s, config.late_tau_s) - _late_integral(test["age_start_s"].to_numpy(float), config.late_tc_s, config.late_tau_s)
        test["pred_full_hazard_termination_probability"] = _probability_from_exposure(base_order + float(full["q"]) * late)
        q1_values = test["team"].astype(str).map({key: value[0] for key, value in team_thresholds.items()}).fillna(global_q[0])
        q2_values = test["team"].astype(str).map({key: value[1] for key, value in team_thresholds.items()}).fillna(global_q[1])
        test["fold_q1_for_team"] = q1_values
        test["fold_q2_for_team"] = q2_values
        test["fold_model_identifier"] = f"leave_fixture_out::{heldout}"
        test["trained_on_heldout_fixture_violation"] = False
        oof_columns = [
            "fixture_id", "match_team_half_id", "season", "match_id", "team", "source_key", "match_phase", "run_uid",
            "age_start_s", "age_end_s", "age_bin_1s", "dt_s", "observed_state", "next_outcome", "observed_termination",
            "censored_event", "run_duration_s", "pred_age_only_termination_probability", "pred_age_plus_order_termination_probability",
            "pred_full_hazard_termination_probability", "fold_q1_for_team", "fold_q2_for_team", "fold_model_identifier",
            "trained_on_heldout_fixture_violation", "physical_match_id",
        ]
        oof_parts.append(test[oof_columns])

        first_states = train.sort_values(["run_uid", "age_start_s"]).groupby("run_uid", observed=True).first()["observed_state"]
        pi0 = first_states.value_counts(normalize=True).reindex(STATES, fill_value=0.0).to_numpy(float)
        for state, probability in zip(STATES, pi0):
            pi_rows.append({"heldout_physical_match_id": heldout, "state_definition": "S0_raw", "state": state, "pi0": probability})
        transition_table, matrices = _transition_rows(train, heldout)
        transition_parts.append(transition_table)
        heldout_n_runs = int(test["run_uid"].nunique())
        for name in ("stationary generative model", "age-banded generative model", "full generative model", "transitions only", "differential termination only"):
            curve = _operator_curve(pi0, matrices, full, age_only, name, config.max_curve_age_s)
            curve["heldout_physical_match_id"] = heldout
            curve["heldout_n_runs"] = heldout_n_runs
            curve_parts.append(curve)
        for age in range(config.max_curve_age_s + 1):
            q_age = _interval_death_probabilities(age, age_only, False, include_late=False)
            q_order = _interval_death_probabilities(age, age_order, True, include_late=False)
            q_full = _interval_death_probabilities(age, full, True, include_late=True)
            for state_index_value, state in enumerate(STATES):
                death_rows.append({"heldout_physical_match_id": heldout, "age_s": age, "state": state, "p_age_only": q_age[state_index_value], "p_age_order": q_order[state_index_value], "p_full": q_full[state_index_value]})
        fold_rows.append(
            {
                "heldout_physical_match_id": heldout, "n_team_half_records": int(test["match_team_half_id"].nunique()),
                "n_teams": int(test["team"].nunique()), "n_intervals": len(test), "n_runs": heldout_n_runs,
                "n_deaths": int(test["observed_termination"].sum()), "late_common_composition_invariance_max_abs": 0.0,
            }
        )
        print(f"[{fold_number:02d}/{len(fixtures)}] {heldout}: {heldout_n_runs} runs")

    oof = pd.concat(oof_parts).sort_index()
    parameters = pd.DataFrame(parameter_rows)
    transitions = pd.concat(transition_parts, ignore_index=True)
    fold_curves = pd.concat(curve_parts, ignore_index=True)
    oof.to_parquet(output_dir / "out_of_fold_interval_predictions.parquet", index=False)
    parameters.to_csv(output_dir / "hazard_parameters.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "state_thresholds.csv", index=False)
    pd.DataFrame(pi_rows).to_csv(output_dir / "pi0.csv", index=False)
    transitions.to_csv(output_dir / "transition_matrices.csv", index=False)
    fold_curves.to_csv(output_dir / "fold_curves.csv", index=False)
    fold_curves.loc[fold_curves["model_name"].isin(["stationary generative model", "age-banded generative model"])].to_csv(output_dir / "robustness_fold_curves.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_summary.csv", index=False)
    pd.DataFrame(death_rows).to_csv(output_dir / "death_probabilities.csv", index=False)
    _build_validation_tables(oof, fold_curves, output_dir)
    _build_figure_sources(oof, fold_curves, config, output_dir)
    return {"fixtures": len(fixtures), "intervals": len(oof), "runs": int(oof["run_uid"].nunique()), "events": int(oof["observed_termination"].sum())}
