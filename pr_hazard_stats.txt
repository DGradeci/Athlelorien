# src/utils/hazard_stats.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ============================================================
# Small helpers
# ============================================================

def _as_str_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = d[c].astype(str)
    return d


def _nearest_index(times: np.ndarray, target) -> int | None:
    if len(times) == 0:
        return None

    target = pd.Timestamp(target).to_datetime64()
    pos = np.searchsorted(times, target)

    if pos <= 0:
        return 0
    if pos >= len(times):
        return len(times) - 1

    left = pos - 1
    right = pos

    dl = abs((times[left] - target).astype("timedelta64[ns]").astype(np.int64))
    dr = abs((times[right] - target).astype("timedelta64[ns]").astype(np.int64))

    return left if dl <= dr else right


def _empirical_ccdf(values, min_value=None):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]

    if min_value is not None:
        x = x[x >= min_value]

    if len(x) == 0:
        return np.array([]), np.array([])

    x = np.sort(x)
    n = len(x)
    thresholds, first_idx = np.unique(x, return_index=True)
    ccdf = (n - first_idx) / n

    return thresholds, ccdf


# ============================================================
# State assignment
# ============================================================

def assign_p_states_to_pmv(
    df_pmv: pd.DataFrame,
    *,
    p_col: str = "p_group",
    team_col: str = "team",
    out_col: str = "p_state_current",
    q_low: float = 1 / 3,
    q_high: float = 2 / 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Assign low/mid/high current-polarisation states to df_pmv
    using team-specific terciles of p_col.
    """
    d = df_pmv.copy()

    if p_col not in d.columns:
        raise KeyError(f"{p_col!r} not found in df_pmv.")

    d[team_col] = d[team_col].astype(str)
    d[out_col] = pd.Series(index=d.index, dtype="object")
    d["p_state_q1"] = np.nan
    d["p_state_q2"] = np.nan

    for team, g in d.groupby(team_col, sort=False, dropna=False):
        vals = pd.to_numeric(g[p_col], errors="coerce")
        vals = vals[np.isfinite(vals)].to_numpy(float)

        if verbose:
            print(f"Team={team} | finite {p_col}: {len(vals)} / {len(g)}")

        if len(vals) < 10 or len(np.unique(vals)) < 3:
            if verbose:
                print(f"  WARNING: not enough values to assign p states for {team}")
            continue

        q1, q2 = np.quantile(vals, [q_low, q_high])

        m_team = d[team_col].astype(str).eq(str(team))
        x = pd.to_numeric(d.loc[m_team, p_col], errors="coerce")

        d.loc[m_team & (x <= q1), out_col] = "low"
        d.loc[m_team & (x > q1) & (x <= q2), out_col] = "mid"
        d.loc[m_team & (x > q2), out_col] = "high"

        d.loc[m_team, "p_state_q1"] = q1
        d.loc[m_team, "p_state_q2"] = q2

        if verbose:
            print(f"  q1={q1:.3f}, q2={q2:.3f}")

    d[out_col] = pd.Categorical(
        d[out_col],
        categories=["low", "mid", "high"],
        ordered=True,
    )

    return d


# ============================================================
# Hazard interval construction
# ============================================================

def build_centroid_hazard_intervals(
    transport: dict,
    df_pmv: pd.DataFrame,
    *,
    interval_s: float = 1.0,
    min_duration_s: float = 1.0,
    team_col: str = "team",
    phase_col: str = "match_phase",
    t_col: str = "_t",
    p_col: str = "p_group",
    state_col: str = "p_state_current",
    assign_states: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build person-period / run-age hazard table for centroid runs.

    One row = one exposure interval inside one centroid run.

    Columns include:
      run_uid, team, match_phase, t0, t1, age_start_s, age_mid_s,
      dt_s, event, p_group, m_group, v_group_mps, p_state_current

    event = 1 only for the terminal interval of each run.
    """

    if "runs_long" not in transport:
        raise KeyError("transport must contain 'runs_long'.")

    runs = transport["runs_long"].copy()
    runs = runs[runs["track_type"] == "centroid"].copy()

    runs[team_col] = runs[team_col].astype(str)
    runs[phase_col] = runs[phase_col].astype(str)
    runs["t_start"] = pd.to_datetime(runs["t_start"])
    runs["t_end"] = pd.to_datetime(runs["t_end"])
    runs["duration_s"] = pd.to_numeric(runs["duration_s"], errors="coerce")
    runs["run_length_m"] = pd.to_numeric(runs["run_length_m"], errors="coerce")

    runs = runs[
        np.isfinite(runs["duration_s"]) &
        (runs["duration_s"] >= float(min_duration_s))
    ].copy()

    pmv = df_pmv.copy()
    pmv[t_col] = pd.to_datetime(pmv[t_col])
    pmv[team_col] = pmv[team_col].astype(str)
    pmv[phase_col] = pmv[phase_col].astype(str)

    if assign_states or state_col not in pmv.columns:
        pmv = assign_p_states_to_pmv(
            pmv,
            p_col=p_col,
            team_col=team_col,
            out_col=state_col,
            verbose=verbose,
        )

    pmv = pmv.sort_values([team_col, phase_col, t_col]).reset_index(drop=True)

    # Cache PMV arrays by team/phase.
    lookup = {}
    value_cols = [p_col, "m_group", "v_group_mps", state_col, "p_state_q1", "p_state_q2"]

    for keys, g in pmv.groupby([team_col, phase_col], sort=False, dropna=False):
        team, phase = keys
        gg = g.sort_values(t_col).reset_index(drop=True)

        lookup[(str(team), str(phase))] = {
            "times": gg[t_col].to_numpy(dtype="datetime64[ns]"),
            "frame": gg,
        }

    rows = []

    for _, r in runs.iterrows():
        team = str(r[team_col])
        phase = str(r[phase_col])
        key = (team, phase)

        if key not in lookup:
            continue

        t_start = pd.Timestamp(r["t_start"])
        duration = float(r["duration_s"])

        if not np.isfinite(duration) or duration <= 0:
            continue

        # Interval edges in run age.
        edges = np.arange(0.0, duration, float(interval_s))
        edges = np.r_[edges, duration]
        edges = np.unique(np.round(edges, 10))

        if len(edges) < 2:
            continue

        times = lookup[key]["times"]
        frame = lookup[key]["frame"]

        for j in range(len(edges) - 1):
            age0 = float(edges[j])
            age1 = float(edges[j + 1])
            dt = age1 - age0

            if dt <= 0:
                continue

            age_mid = 0.5 * (age0 + age1)
            t0 = t_start + pd.Timedelta(seconds=age0)
            t1 = t_start + pd.Timedelta(seconds=age1)
            tm = t_start + pd.Timedelta(seconds=age_mid)

            idx = _nearest_index(times, tm)

            if idx is None:
                continue

            pmv_row = frame.iloc[idx]

            row = {
                "match_id": r.get("match_id", "single_match"),
                "team": team,
                "source_key": r.get("source_key", "unknown_source"),
                "match_phase": phase,
                "track_uid": r.get("track_uid", np.nan),
                "run_uid": r["run_uid"],
                "t_run_start": t_start,
                "t_run_end": r["t_end"],
                "t0": t0,
                "t1": t1,
                "t_mid": tm,
                "age_start_s": age0,
                "age_mid_s": age_mid,
                "age_end_s": age1,
                "dt_s": dt,
                "event": int(j == len(edges) - 2),
                "run_duration_s": duration,
                "run_length_m": r.get("run_length_m", np.nan),
                "run_v_mean_mps": r.get("v_mean_mps", np.nan),
                "p_group": pmv_row.get(p_col, np.nan),
                "m_group": pmv_row.get("m_group", np.nan),
                "v_group_mps": pmv_row.get("v_group_mps", np.nan),
                state_col: pmv_row.get(state_col, np.nan),
                "p_state_q1": pmv_row.get("p_state_q1", np.nan),
                "p_state_q2": pmv_row.get("p_state_q2", np.nan),
            }

            rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["event"] = out["event"].astype(int)
    out["dt_s"] = pd.to_numeric(out["dt_s"], errors="coerce")
    out["age_mid_s"] = pd.to_numeric(out["age_mid_s"], errors="coerce")
    out["p_group"] = pd.to_numeric(out["p_group"], errors="coerce")

    out[state_col] = pd.Categorical(
        out[state_col],
        categories=["low", "mid", "high"],
        ordered=True,
    )

    if verbose:
        print("hazard_intervals:", out.shape)
        print("events:", int(out["event"].sum()))
        print("runs:", out["run_uid"].nunique())

    return out.reset_index(drop=True)


# ============================================================
# Empirical binned hazard
# ============================================================

def build_empirical_hazard(
    hazard_intervals: pd.DataFrame,
    *,
    state_col: str = "p_state_current",
    age_bin_width_s: float = 2.0,
    max_age_s: float | None = None,
    min_exposure_s: float = 5.0,
) -> pd.DataFrame:
    """
    Empirical binned hazard:
        h = events / exposure_time
    grouped by team, state, and age bin.
    """
    d = hazard_intervals.copy()
    d = d.dropna(subset=["age_mid_s", "dt_s", "event", state_col]).copy()

    if max_age_s is not None:
        d = d[d["age_mid_s"] <= float(max_age_s)].copy()

    d["age_bin_left_s"] = (
        np.floor(d["age_mid_s"] / float(age_bin_width_s)) * float(age_bin_width_s)
    )
    d["age_bin_mid_s"] = d["age_bin_left_s"] + 0.5 * float(age_bin_width_s)

    rows = []

    group_cols = ["team", state_col, "age_bin_left_s", "age_bin_mid_s"]

    for keys, g in d.groupby(group_cols, observed=True, sort=True):
        team, state, age_left, age_mid = keys
        exposure = float(g["dt_s"].sum())
        events = int(g["event"].sum())

        if exposure < float(min_exposure_s):
            continue

        hazard = events / exposure if exposure > 0 else np.nan

        # Simple Poisson-style uncertainty on rate.
        if events > 0:
            se = np.sqrt(events) / exposure
            lo = max(hazard - 1.96 * se, 0.0)
            hi = hazard + 1.96 * se
        else:
            lo = 0.0
            hi = 1.96 / exposure

        rows.append({
            "team": team,
            state_col: state,
            "age_bin_left_s": float(age_left),
            "age_bin_mid_s": float(age_mid),
            "n_intervals": int(len(g)),
            "events": events,
            "exposure_s": exposure,
            "hazard_emp": hazard,
            "hazard_lo": lo,
            "hazard_hi": hi,
        })

    return pd.DataFrame(rows)


# ============================================================
# Inverse-age hazard model
# ============================================================

def _inverse_age_hazard(age, lambda_inf, mu, a0):
    age = np.asarray(age, dtype=float)
    return lambda_inf + mu / (float(a0) + age)


def _interval_nll_inverse_age(params, age, dt, event, a0):
    """
    Exact interval likelihood assuming constant h over each small interval:
      P(no event) = exp(-h dt)
      P(event)    = 1 - exp(-h dt)
    """
    log_lambda, log_mu = params
    lambda_inf = np.exp(log_lambda)
    mu = np.exp(log_mu)

    h = _inverse_age_hazard(age, lambda_inf, mu, a0)
    x = np.clip(h * dt, 1e-12, 50.0)

    # no event: -log exp(-x) = x
    nll_no = x

    # event: -log(1 - exp(-x))
    nll_ev = -np.log1p(-np.exp(-x))

    nll = np.where(event.astype(bool), nll_ev, nll_no)
    return float(np.nansum(nll))


def fit_inverse_age_hazard_by_group(
    hazard_intervals: pd.DataFrame,
    *,
    state_col: str = "p_state_current",
    group_cols: tuple[str, ...] = ("team", "p_state_current"),
    a0: float = 1.0,
    min_events: int = 20,
    max_age_s: float | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fit:
        h(a) = lambda_inf + mu / (a0 + a)

    separately by group.
    """
    d = hazard_intervals.copy()
    d = d.dropna(subset=["age_mid_s", "dt_s", "event"]).copy()

    if max_age_s is not None:
        d = d[d["age_mid_s"] <= float(max_age_s)].copy()

    rows = []

    for keys, g in d.groupby(list(group_cols), observed=True, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        meta = dict(zip(group_cols, keys))

        g = g.dropna(subset=["age_mid_s", "dt_s", "event"]).copy()
        events = int(g["event"].sum())

        if events < int(min_events):
            row = dict(meta)
            row.update({
                "success": False,
                "message": "too_few_events",
                "n_intervals": int(len(g)),
                "events": events,
                "exposure_s": float(g["dt_s"].sum()),
                "lambda_inf": np.nan,
                "mu": np.nan,
                "a0": float(a0),
                "nll": np.nan,
                "aic": np.nan,
            })
            rows.append(row)
            continue

        age = g["age_mid_s"].to_numpy(float)
        dt = g["dt_s"].to_numpy(float)
        event = g["event"].to_numpy(int)

        crude_rate = max(events / np.sum(dt), 1e-4)
        init_lambda = max(crude_rate * 0.5, 1e-4)
        init_mu = 0.5

        x0 = np.log([init_lambda, init_mu])

        res = minimize(
            _interval_nll_inverse_age,
            x0=x0,
            args=(age, dt, event, float(a0)),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
        )

        lambda_inf, mu = np.exp(res.x)
        nll = float(res.fun)

        row = dict(meta)
        row.update({
            "success": bool(res.success),
            "message": str(res.message),
            "n_intervals": int(len(g)),
            "events": events,
            "exposure_s": float(np.sum(dt)),
            "lambda_inf": float(lambda_inf),
            "mu": float(mu),
            "a0": float(a0),
            "nll": nll,
            "aic": 2 * 2 + 2 * nll,
        })
        rows.append(row)

        if verbose:
            print(
                f"{meta} | events={events}, "
                f"lambda_inf={lambda_inf:.4g}, mu={mu:.4g}, success={res.success}"
            )

    return pd.DataFrame(rows)


def build_hazard_prediction_grid(
    hazard_fit_summary: pd.DataFrame,
    *,
    state_col: str = "p_state_current",
    age_max_s: float = 60.0,
    n_grid: int = 300,
) -> pd.DataFrame:
    """
    Smooth predicted hazard and model-implied survival for each fitted group.

    Survival for fixed state:
        S(a) = exp(-lambda_inf a) * (a0 / (a0+a))^mu
    """
    rows = []
    age = np.linspace(0.0, float(age_max_s), int(n_grid))

    for _, r in hazard_fit_summary.iterrows():
        if not bool(r.get("success", False)):
            continue

        lam = float(r["lambda_inf"])
        mu = float(r["mu"])
        a0 = float(r["a0"])

        h = _inverse_age_hazard(age, lam, mu, a0)
        S = np.exp(-lam * age) * (a0 / (a0 + age)) ** mu

        for a, hh, ss in zip(age, h, S):
            rows.append({
                "team": r.get("team", "pooled"),
                state_col: r.get(state_col, np.nan),
                "age_s": float(a),
                "hazard_pred": float(hh),
                "survival_pred": float(ss),
                "lambda_inf": lam,
                "mu": mu,
                "a0": a0,
            })

    return pd.DataFrame(rows)


# ============================================================
# Empirical survival from run starting state
# ============================================================

def build_run_survival_from_intervals(
    hazard_intervals: pd.DataFrame,
    *,
    state_col: str = "p_state_current",
    min_duration_s: float = 1.0,
) -> pd.DataFrame:
    """
    Empirical run-duration CCDF where each run is labelled by its first interval state.
    This is a predictive-ish comparison, not the same as p_mean.
    """
    d = hazard_intervals.copy()
    d = d.sort_values(["team", "run_uid", "age_start_s"]).copy()

    first = (
        d.groupby(["team", "run_uid"], observed=True, sort=False)
        .first()
        .reset_index()
    )

    first = first.dropna(subset=[state_col, "run_duration_s"]).copy()

    rows = []

    for keys, g in first.groupby(["team", state_col], observed=True, sort=False):
        team, state = keys
        x, y = _empirical_ccdf(g["run_duration_s"], min_value=min_duration_s)

        for xx, yy in zip(x, y):
            rows.append({
                "team": team,
                state_col: state,
                "age_s": float(xx),
                "survival_emp": float(yy),
                "n_runs": int(len(g)),
            })

    return pd.DataFrame(rows)