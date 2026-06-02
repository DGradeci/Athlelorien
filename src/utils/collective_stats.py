# src/utils/collective_stats.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _dedup(cols):
    # keep order, remove duplicates
    return list(dict.fromkeys(cols))


def build_df_polarisation(
    df_paths: pd.DataFrame,
    min_speed_mps: float = 0.0,
    team_col: str = "team",
    phase_col: str = "match_phase",
    player_col: str = "player_name",
    path_col: str = "path_uid",
    t_col: str = "_t",
    x_col: str = "x_m",
    y_col: str = "y_m",
    phases: tuple[str, ...] = ("1H", "2H"),
    status_col: str | None = None,       # optional safety filter
    active_value: str = "active",
) -> pd.DataFrame:
    """
    Polarisation p_group(t) in [0,1] for each (team, match_phase, time).

    Uses per-player finite differences within each continuous path.

    Returns columns (subset depending on what exists):
      team, match_phase, _t, n_players, polarisation
    """
    req = {player_col, t_col, path_col, x_col, y_col}
    missing = req - set(df_paths.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df_paths.copy()

    # Optional: ensure only active players if status is present
    if status_col is not None and status_col in df.columns:
        df = df[df[status_col] == active_value].copy()

    # time
    df[t_col] = pd.to_datetime(df[t_col], errors="coerce")
    df = df.dropna(subset=[t_col, x_col, y_col, player_col, path_col])

    # phases (avoid categorical weirdness by casting to str after filtering)
    if phase_col in df.columns and phases is not None:
        df = df[df[phase_col].isin(phases)].copy()
        df[phase_col] = df[phase_col].astype(str)

    # also cast team to str if present (head-to-head safe)
    if team_col in df.columns:
        df[team_col] = df[team_col].astype(str)

    # sort for diffs
    sort_cols = _dedup([c for c in [team_col, phase_col, player_col, path_col, t_col] if c in df.columns])
    df = df.sort_values(sort_cols)

    # per-player diffs within paths
    gcols = _dedup([c for c in [team_col, phase_col, player_col, path_col] if c in df.columns])
    g = df.groupby(gcols, sort=False, observed=True)

    dt = g[t_col].diff().dt.total_seconds().replace(0, np.nan)
    dx = g[x_col].diff()
    dy = g[y_col].diff()

    vx = dx / dt
    vy = dy / dt
    speed = np.sqrt(vx**2 + vy**2)

    df["ux"] = vx / speed
    df["uy"] = vy / speed
    df["speed_pitch_mps"] = speed

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ux", "uy", "speed_pitch_mps"])

    if min_speed_mps > 0:
        df = df[df["speed_pitch_mps"] >= float(min_speed_mps)].copy()

    agg_cols = _dedup([c for c in [team_col, phase_col, t_col] if c in df.columns])

    out = (
        df.groupby(agg_cols, observed=True)
          .agg(ux_mean=("ux", "mean"), uy_mean=("uy", "mean"), n_players=("ux", "size"))
          .reset_index()
    )

    out["polarisation"] = np.sqrt(out["ux_mean"]**2 + out["uy_mean"]**2).clip(0, 1)
    keep = [c for c in [team_col, phase_col, t_col, "n_players", "polarisation"] if c in out.columns]
    return out[keep]


def compute_pmv(
    df_paths: pd.DataFrame,
    centroid_ts: pd.DataFrame | None = None,
    min_speed_mps: float = 0.0,
    team_col: str = "team",
    phase_col: str = "match_phase",
    player_col: str = "player_name",
    path_col: str = "path_uid",
    t_col: str = "_t",
    x_col: str = "x_m",
    y_col: str = "y_m",
    phases: tuple[str, ...] = ("1H", "2H"),
    status_col: str | None = None,       # optional safety filter
    active_value: str = "active",
) -> pd.DataFrame:
    """
    Returns: team, match_phase, _t, n_players, p_group, m_group, v_group_mps

    p_group = || <vhat> ||
    m_group = | < (rhat x vhat)_z > |
    where rhat points from centroid to player, vhat is unit velocity.
    """
    req = {player_col, t_col, path_col, x_col, y_col}
    missing = req - set(df_paths.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df_paths.copy()

    if status_col is not None and status_col in df.columns:
        df = df[df[status_col] == active_value].copy()

    df[t_col] = pd.to_datetime(df[t_col], errors="coerce")
    df = df.dropna(subset=[t_col, x_col, y_col, player_col, path_col])

    if phase_col in df.columns and phases is not None:
        df = df[df[phase_col].isin(phases)].copy()
        df[phase_col] = df[phase_col].astype(str)

    if team_col in df.columns:
        df[team_col] = df[team_col].astype(str)

    sort_cols = _dedup([c for c in [team_col, phase_col, player_col, path_col, t_col] if c in df.columns])
    df = df.sort_values(sort_cols)

    gcols = _dedup([c for c in [team_col, phase_col, player_col, path_col] if c in df.columns])
    g = df.groupby(gcols, sort=False, observed=True)

    dt = g[t_col].diff().dt.total_seconds().replace(0, np.nan)
    dx = g[x_col].diff()
    dy = g[y_col].diff()

    vx = dx / dt
    vy = dy / dt
    speed = np.sqrt(vx**2 + vy**2)

    df["ux"] = vx / speed
    df["uy"] = vy / speed
    df["speed_pitch_mps"] = speed

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ux", "uy", "speed_pitch_mps", x_col, y_col])

    if min_speed_mps > 0:
        df = df[df["speed_pitch_mps"] >= float(min_speed_mps)].copy()

    key_cols = _dedup([c for c in [team_col, phase_col, t_col] if c in df.columns])

    # centroid: reuse if provided (recommended)
    if centroid_ts is None:
        cent = (
            df.groupby(key_cols, observed=True)
              .agg(cx=(x_col, "mean"), cy=(y_col, "mean"))
              .reset_index()
        )
    else:
        cent = centroid_ts.copy()
        cent[t_col] = pd.to_datetime(cent[t_col], errors="coerce")
        if phase_col in cent.columns:
            cent[phase_col] = cent[phase_col].astype(str)
        if team_col in cent.columns:
            cent[team_col] = cent[team_col].astype(str)

        if "cx" not in cent.columns:
            cent = cent.rename(columns={x_col: "cx", y_col: "cy"})

        cent = cent[key_cols + ["cx", "cy"]].dropna()

    df = df.merge(cent, on=key_cols, how="left")

    rx = df[x_col] - df["cx"]
    ry = df[y_col] - df["cy"]
    rnorm = np.sqrt(rx**2 + ry**2)

    df["rx_hat"] = rx / rnorm
    df["ry_hat"] = ry / rnorm
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["rx_hat", "ry_hat"])

    df["cross_z"] = df["rx_hat"] * df["uy"] - df["ry_hat"] * df["ux"]

    out = (
        df.groupby(key_cols, observed=True)
          .agg(
              ux_mean=("ux", "mean"),
              uy_mean=("uy", "mean"),
              cross_mean=("cross_z", "mean"),
              v_group_mps=("speed_pitch_mps", "mean"),
              n_players=("ux", "size"),
          )
          .reset_index()
    )

    out["p_group"] = np.sqrt(out["ux_mean"]**2 + out["uy_mean"]**2).clip(0, 1)
    out["m_group"] = np.abs(out["cross_mean"]).clip(0, 1)

    return out[key_cols + ["n_players", "p_group", "m_group", "v_group_mps"]]

# ============================================================
# Figure-ready collective-order/run-state helpers
# ============================================================

def _nearest_value_at_time(
    sub: pd.DataFrame,
    t,
    value_col: str,
    *,
    t_col: str = "_t",
) -> float:
    """
    Nearest value in sub[value_col] at time t.
    sub must contain t_col and value_col.
    """
    if sub.empty or value_col not in sub.columns:
        return np.nan

    d = sub.dropna(subset=[t_col, value_col]).sort_values(t_col)
    if d.empty:
        return np.nan

    times = d[t_col].to_numpy(dtype="datetime64[ns]")
    vals = d[value_col].to_numpy(dtype=float)
    target = pd.Timestamp(t).to_datetime64()

    pos = np.searchsorted(times, target)

    if pos <= 0:
        idx = 0
    elif pos >= len(times):
        idx = len(times) - 1
    else:
        left = pos - 1
        right = pos
        dl = abs((times[left] - target).astype("timedelta64[ns]").astype(np.int64))
        dr = abs((times[right] - target).astype("timedelta64[ns]").astype(np.int64))
        idx = left if dl <= dr else right

    return float(vals[idx])


def assign_order_states_by_team(
    df: pd.DataFrame,
    *,
    state_col: str = "p_mean",
    team_col: str = "team",
    out_col: str = "p_state",
    q_low: float = 1 / 3,
    q_high: float = 2 / 3,
    min_n: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Assign low/mid/high order states separately within each team,
    using terciles of state_col.

    Adds:
      out_col
      p_state_source
      p_q1
      p_q2
    """
    d = df.copy()

    if state_col not in d.columns:
        raise KeyError(
            f"{state_col!r} not found. Available columns: {list(d.columns)}"
        )

    d[out_col] = pd.Series(index=d.index, dtype="object")
    d["p_state_source"] = state_col
    d["p_q1"] = np.nan
    d["p_q2"] = np.nan

    for team, g in d.groupby(team_col, sort=False, dropna=False):
        vals = pd.to_numeric(g[state_col], errors="coerce")
        vals = vals[np.isfinite(vals)].to_numpy(dtype=float)

        if verbose:
            print(f"Team={team} | finite {state_col}: {len(vals)} / {len(g)}")

        if len(vals) < int(min_n) or len(np.unique(vals)) < 3:
            if verbose:
                print(f"  WARNING: not enough finite/unique values to assign states for {team}")
            continue

        q1, q2 = np.quantile(vals, [q_low, q_high])

        idx_team = d[team_col].astype(str).eq(str(team))
        x = pd.to_numeric(d.loc[idx_team, state_col], errors="coerce")

        d.loc[idx_team & (x <= q1), out_col] = "low"
        d.loc[idx_team & (x > q1) & (x <= q2), out_col] = "mid"
        d.loc[idx_team & (x > q2), out_col] = "high"

        d.loc[idx_team, "p_q1"] = q1
        d.loc[idx_team, "p_q2"] = q2

        if verbose:
            print(f"  q1={q1:.3f}, q2={q2:.3f}")

    d[out_col] = pd.Categorical(
        d[out_col],
        categories=["low", "mid", "high"],
        ordered=True,
    )

    return d


def build_centroid_order_runs(
    transport: dict,
    df_pmv: pd.DataFrame,
    *,
    order_state_col: str = "p_mean",
    early_window_s: float = 3.0,
    team_col: str = "team",
    phase_col: str = "match_phase",
    t_col: str = "_t",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build one row per centroid run with collective-order summaries attached.

    Adds:
      p_start, p_early_3s, p_mean, p_median
      m_start, m_early_3s, m_mean
      v_group_start, v_group_early_3s, v_group_mean
      p_state assigned from order_state_col separately per team
    """
    if "runs_long" not in transport:
        raise KeyError("transport must contain 'runs_long'.")

    centroid_runs = transport["runs_long"].copy()
    centroid_runs = centroid_runs[centroid_runs["track_type"] == "centroid"].copy()

    required = {team_col, phase_col, "t_start", "t_end", "run_uid"}
    missing = required - set(centroid_runs.columns)
    if missing:
        raise KeyError(f"centroid runs missing required columns: {sorted(missing)}")

    centroid_runs[team_col] = centroid_runs[team_col].astype(str)
    centroid_runs[phase_col] = centroid_runs[phase_col].astype(str)
    centroid_runs["t_start"] = pd.to_datetime(centroid_runs["t_start"])
    centroid_runs["t_end"] = pd.to_datetime(centroid_runs["t_end"])

    pmv = df_pmv.copy()
    pmv[t_col] = pd.to_datetime(pmv[t_col])
    pmv[team_col] = pmv[team_col].astype(str)
    pmv[phase_col] = pmv[phase_col].astype(str)
    pmv = pmv.sort_values([team_col, phase_col, t_col]).reset_index(drop=True)

    rows = []

    for _, r in centroid_runs.iterrows():
        team = str(r[team_col])
        phase = str(r[phase_col])
        t0 = pd.Timestamp(r["t_start"])
        t1 = pd.Timestamp(r["t_end"])
        t_early_end = min(t1, t0 + pd.Timedelta(seconds=float(early_window_s)))

        sub = pmv[
            (pmv[team_col].astype(str) == team) &
            (pmv[phase_col].astype(str) == phase)
        ].copy()

        inside = sub[(sub[t_col] >= t0) & (sub[t_col] <= t1)]
        early = sub[(sub[t_col] >= t0) & (sub[t_col] <= t_early_end)]

        row = r.to_dict()

        # Start values
        row["p_start"] = _nearest_value_at_time(sub, t0, "p_group", t_col=t_col)
        row["m_start"] = _nearest_value_at_time(sub, t0, "m_group", t_col=t_col)
        row["v_group_start"] = _nearest_value_at_time(sub, t0, "v_group_mps", t_col=t_col)

        # Early-window values
        row["p_early_3s"] = early["p_group"].mean() if len(early) else np.nan
        row["m_early_3s"] = early["m_group"].mean() if len(early) else np.nan
        row["v_group_early_3s"] = early["v_group_mps"].mean() if len(early) else np.nan

        # Whole-run values
        if len(inside) > 0:
            row["p_mean"] = inside["p_group"].mean()
            row["p_median"] = inside["p_group"].median()
            row["m_mean"] = inside["m_group"].mean()
            row["v_group_mean"] = inside["v_group_mps"].mean()
            row["n_pmv_samples"] = int(len(inside))
        else:
            row["p_mean"] = np.nan
            row["p_median"] = np.nan
            row["m_mean"] = np.nan
            row["v_group_mean"] = np.nan
            row["n_pmv_samples"] = 0

        rows.append(row)

    out = pd.DataFrame(rows)

    out = assign_order_states_by_team(
        out,
        state_col=order_state_col,
        team_col=team_col,
        out_col="p_state",
        verbose=verbose,
    )

    return out


def build_collective_order_tables_from_transport(
    transport: dict,
    *,
    order_state_col: str = "p_mean",
    early_window_s: float = 3.0,
    min_speed_mps: float = 0.0,
    team_col: str = "team",
    phase_col: str = "match_phase",
    player_col: str = "player_name",
    path_col: str = "path_uid",
    t_col: str = "_t",
    x_col: str = "x_m",
    y_col: str = "y_m",
    phases: tuple[str, ...] = ("1H", "2H"),
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper for Figure 3-style collective-order analysis.

    Returns
    -------
    df_pmv, centroid_order_runs
    """
    if "df_paths" not in transport:
        raise KeyError("transport must contain 'df_paths'.")

    centroid_ts = transport.get("centroid_ts", None)

    df_pmv = compute_pmv(
        transport["df_paths"],
        centroid_ts=centroid_ts,
        min_speed_mps=min_speed_mps,
        team_col=team_col,
        phase_col=phase_col,
        player_col=player_col,
        path_col=path_col,
        t_col=t_col,
        x_col=x_col,
        y_col=y_col,
        phases=phases,
    )

    df_pmv[t_col] = pd.to_datetime(df_pmv[t_col])
    df_pmv[team_col] = df_pmv[team_col].astype(str)
    df_pmv[phase_col] = df_pmv[phase_col].astype(str)
    df_pmv = df_pmv.sort_values([team_col, phase_col, t_col]).reset_index(drop=True)

    centroid_order_runs = build_centroid_order_runs(
        transport,
        df_pmv,
        order_state_col=order_state_col,
        early_window_s=early_window_s,
        team_col=team_col,
        phase_col=phase_col,
        t_col=t_col,
        verbose=verbose,
    )

    return df_pmv, centroid_order_runs