# src/utils/player_status.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def _choose_time_col(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """
    Choose 'timestamp' if present else 'time', and parse to datetime.
    Returns (time_col, parsed_series).
    """
    if "timestamp" in df.columns:
        tcol = "timestamp"
    elif "time" in df.columns:
        tcol = "time"
    else:
        raise ValueError("df must contain either 'timestamp' or 'time' column.")

    t = pd.to_datetime(df[tcol], errors="coerce")
    if t.isna().all():
        raise ValueError(f"Could not parse any datetimes from column '{tcol}'.")
    return tcol, t


def _compute_depth_from_edges(
    df_xy: pd.DataFrame,
    pitch_xy,
    x_col: str = "x_m",
    y_col: str = "y_m",
) -> pd.Series:
    """
    For each sample, compute depth (>=0) inside the pitch rectangle:
      depth = min distance to any of the 4 pitch lines
      depth = 0 if outside the rectangle.
    pitch_xy is expected to be 4 corners or a rectangle-like set of points.
    """
    P = np.asarray(pitch_xy, dtype=float)
    xmin, ymin = P.min(axis=0)
    xmax, ymax = P.max(axis=0)

    x = df_xy[x_col].to_numpy(dtype=float)
    y = df_xy[y_col].to_numpy(dtype=float)

    dx_left = x - xmin
    dx_right = xmax - x
    dy_bottom = y - ymin
    dy_top = ymax - y

    depth = np.minimum.reduce([dx_left, dx_right, dy_bottom, dy_top])
    depth = np.where(depth < 0, 0.0, depth)  # 0 outside pitch
    return pd.Series(depth, index=df_xy.index, name="depth_from_edge")


def _compute_signed_depth_from_edges(
    df_xy: pd.DataFrame,
    pitch_xy,
    x_col: str = "x_m",
    y_col: str = "y_m",
) -> pd.Series:
    """
    Signed depth from pitch rectangle. Positive means inside the pitch,
    negative means outside, and magnitude is distance to nearest boundary.
    """
    P = np.asarray(pitch_xy, dtype=float)
    xmin, ymin = P.min(axis=0)
    xmax, ymax = P.max(axis=0)

    x = df_xy[x_col].to_numpy(dtype=float)
    y = df_xy[y_col].to_numpy(dtype=float)

    dx_left = x - xmin
    dx_right = xmax - x
    dy_bottom = y - ymin
    dy_top = ymax - y
    depth = np.minimum.reduce([dx_left, dx_right, dy_bottom, dy_top])
    return pd.Series(depth, index=df_xy.index, name="signed_depth_from_edge")


def _add_recent_motion_features(
    df: pd.DataFrame,
    *,
    player_col: str,
    time_col: str,
    x_col: str,
    y_col: str,
    window_s: float,
) -> pd.DataFrame:
    out = df.sort_values([player_col, time_col]).copy()
    dt = (
        out.groupby(player_col, sort=False)[time_col]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .clip(lower=0.0, upper=5.0)
    )
    dx = out.groupby(player_col, sort=False)[x_col].diff().fillna(0.0).to_numpy(float)
    dy = out.groupby(player_col, sort=False)[y_col].diff().fillna(0.0).to_numpy(float)
    dt_arr = dt.to_numpy(float)
    step_raw = np.sqrt(dx**2 + dy**2)
    step_raw = np.where(dt_arr > 0, step_raw, 0.0)

    # Clamp impossible jumps so GPS spikes do not create artificial playing evidence.
    max_step = 8.0 * np.maximum(dt_arr, 1.0)
    scale = np.ones_like(step_raw)
    moving = step_raw > 0
    scale[moving] = np.minimum(1.0, max_step[moving] / step_raw[moving])
    dx = dx * scale
    dy = dy * scale
    step = np.minimum(step_raw, max_step)

    out["_step_m"] = step
    out["vx_mps"] = np.divide(dx, dt_arr, out=np.zeros_like(dx), where=dt_arr > 0)
    out["vy_mps"] = np.divide(dy, dt_arr, out=np.zeros_like(dy), where=dt_arr > 0)
    out["speed_mps"] = np.divide(step, dt_arr, out=np.zeros_like(step), where=dt_arr > 0)

    window_n = max(2, int(round(float(window_s))))
    out["recent_path_m"] = (
        out.groupby(player_col, sort=False)["_step_m"]
        .rolling(window_n, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    out["recent_vx_mps"] = (
        out.groupby(player_col, sort=False)["vx_mps"]
        .rolling(window_n, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["recent_vy_mps"] = (
        out.groupby(player_col, sort=False)["vy_mps"]
        .rolling(window_n, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["recent_speed_mps"] = (
        out.groupby(player_col, sort=False)["speed_mps"]
        .rolling(window_n, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return out.drop(columns=["_step_m"])


def _velocity_coherence_score(
    candidates: pd.DataFrame,
    *,
    top_k: int = 7,
    min_speed_mps: float = 0.25,
) -> pd.Series:
    """Score whether each candidate belongs to the main recent-velocity component."""
    if candidates.empty:
        return pd.Series(dtype=float)

    vx = candidates.get("recent_vx_mps", pd.Series(0.0, index=candidates.index)).astype(float).fillna(0.0).to_numpy()
    vy = candidates.get("recent_vy_mps", pd.Series(0.0, index=candidates.index)).astype(float).fillna(0.0).to_numpy()
    speed = np.sqrt(vx**2 + vy**2)
    n = len(candidates)
    if n <= 1:
        return pd.Series(np.ones(n), index=candidates.index, dtype=float)

    unit = np.zeros((n, 2), dtype=float)
    valid = speed >= float(min_speed_mps)
    unit[valid, 0] = vx[valid] / speed[valid]
    unit[valid, 1] = vy[valid] / speed[valid]
    cos = unit @ unit.T
    cos[~valid, :] = np.nan
    cos[:, ~valid] = np.nan
    np.fill_diagonal(cos, np.nan)

    scores = np.zeros(n, dtype=float)
    k = max(1, min(int(top_k), n - 1))
    for i in range(n):
        vals = cos[i, :]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            scores[i] = 0.0
            continue
        vals = np.sort(vals)[-k:]
        scores[i] = float(np.clip((np.nanmean(vals) + 1.0) / 2.0, 0.0, 1.0))
    return pd.Series(scores, index=candidates.index, dtype=float)


def _seed_playing_xi(
    df: pd.DataFrame,
    *,
    player_col: str,
    time_col: str,
    x_col: str,
    y_col: str,
    seed_window_s: float,
    top_n: int,
    coordinate_sanity_margin_m: float,
) -> set[str]:
    t0 = pd.to_datetime(df[time_col]).min()
    seed = df.loc[df[time_col].between(t0, t0 + pd.Timedelta(seconds=float(seed_window_s)), inclusive="both")].copy()
    if seed.empty:
        return set()

    signed_depth = seed.get("signed_depth_from_edge", pd.Series(np.nan, index=seed.index)).astype(float).fillna(-np.inf)
    finite_xy = np.isfinite(seed[x_col].to_numpy(float)) & np.isfinite(seed[y_col].to_numpy(float))
    hard_position_valid = finite_xy & (signed_depth.to_numpy() >= -float(coordinate_sanity_margin_m))
    seed = seed.loc[hard_position_valid].copy()
    if seed.empty:
        return set()

    summary = (
        seed.groupby(player_col, observed=True)
        .agg(
            recent_path_m=("recent_path_m", "max"),
            in_zone_seconds=("in_active_zone", "sum"),
            on_pitch_seconds=("on_pitch", "sum"),
            median_depth=("depth_from_edge", "median"),
            geometry_active_seconds=("geometry_player_status", lambda s: int((s == "active").sum())),
        )
        .fillna(0.0)
    )
    summary["seed_score"] = (
        0.15 * summary["recent_path_m"]
        + 0.02 * summary["in_zone_seconds"]
        + 0.01 * summary["on_pitch_seconds"]
        + 0.03 * summary["median_depth"]
        + 0.02 * summary["geometry_active_seconds"]
    )
    return set(summary.sort_values("seed_score", ascending=False).head(int(top_n)).index.astype(str))


def _apply_sticky_hierarchical_active_xi(
    df_labeled: pd.DataFrame,
    *,
    player_col: str,
    time_col: str,
    x_col: str,
    y_col: str,
    label_col: str,
    max_active_players: int,
    likelihood_window_s: float,
    seed_window_s: float,
    seed_top_n: int,
    position_margin_m: float,
    coordinate_sanity_margin_m: float,
    replacement_confirm_s: float,
    weak_confirm_s: float,
    switch_margin: float,
    weak_score_threshold: float,
    seed_bonus: float,
    continuity_bonus: float,
    movement_weight: float,
    depth_weight: float,
    geometry_active_bonus: float,
    in_zone_bonus: float,
    velocity_coherence_weight: float,
    velocity_top_k: int,
    min_velocity_speed_mps: float,
) -> pd.DataFrame:
    df = df_labeled.copy()
    df = _add_recent_motion_features(
        df,
        player_col=player_col,
        time_col=time_col,
        x_col=x_col,
        y_col=y_col,
        window_s=likelihood_window_s,
    )
    seed_xi = _seed_playing_xi(
        df,
        player_col=player_col,
        time_col=time_col,
        x_col=x_col,
        y_col=y_col,
        seed_window_s=seed_window_s,
        top_n=seed_top_n,
        coordinate_sanity_margin_m=coordinate_sanity_margin_m,
    )

    df["position_gate"] = False
    signed_depth_all = df["signed_depth_from_edge"].astype(float).fillna(-np.inf).to_numpy()
    finite_xy_all = np.isfinite(df[x_col].to_numpy(float)) & np.isfinite(df[y_col].to_numpy(float))
    df["hard_position_valid"] = finite_xy_all & (signed_depth_all >= -float(coordinate_sanity_margin_m))
    df["position_gate_score"] = np.nan
    df["velocity_coherence_score"] = np.nan
    df["playing_likelihood_score"] = np.nan
    df["hierarchical_active_score"] = np.nan
    df["playing_seed_xi"] = df[player_col].astype(str).isin(seed_xi)
    df["playing_selected"] = False
    df["playing_selection_event"] = ""
    df[label_col] = "bench"

    selected_prev: set[str] = set(seed_xi)
    weak_seconds: dict[str, float] = {name: 0.0 for name in selected_prev}
    challenger_seconds: dict[str, float] = {}
    last_t: pd.Timestamp | None = None

    for t, idx in df.groupby(time_col, sort=True).groups.items():
        t = pd.Timestamp(t)
        step_s = 1.0 if last_t is None else max(0.0, min(5.0, (t - last_t).total_seconds()))
        if step_s <= 0:
            step_s = 1.0
        last_t = t

        g = df.loc[idx].copy()
        names = g[player_col].astype(str)
        signed_depth = g["signed_depth_from_edge"].astype(float).fillna(-np.inf)
        hard_position_valid = g["hard_position_valid"].to_numpy(dtype=bool)
        previous_active = names.isin(selected_prev)
        position_gate = pd.Series(
            hard_position_valid & ((signed_depth >= -float(position_margin_m)).to_numpy() | previous_active.to_numpy()),
            index=g.index,
        )
        candidates = g.loc[position_gate].copy()
        selected_now = set(selected_prev)

        if not candidates.empty:
            cand_names = candidates[player_col].astype(str)
            coherence = _velocity_coherence_score(
                candidates,
                top_k=int(velocity_top_k),
                min_speed_mps=float(min_velocity_speed_mps),
            )
            position_score = np.clip(
                (candidates["signed_depth_from_edge"].astype(float).fillna(-float(position_margin_m)) + float(position_margin_m))
                / max(1e-6, float(position_margin_m) + 12.0),
                0.0,
                1.0,
            )
            base_score = (
                movement_weight * candidates["recent_path_m"].astype(float).fillna(0.0)
                + depth_weight * candidates["depth_from_edge"].astype(float).fillna(0.0)
                + geometry_active_bonus * candidates["geometry_player_status"].astype(str).eq("active").astype(float)
                + in_zone_bonus * candidates["in_active_zone"].astype(bool).astype(float)
                + seed_bonus * cand_names.isin(seed_xi).astype(float)
                + velocity_coherence_weight * coherence.reindex(candidates.index).fillna(0.0)
                + 0.35 * position_score.reindex(candidates.index).fillna(0.0)
            )
            sticky_score = base_score + continuity_bonus * cand_names.isin(selected_prev).astype(float)
            candidates["position_gate_score"] = position_score
            candidates["velocity_coherence_score"] = coherence
            candidates["playing_likelihood_score"] = base_score
            candidates["hierarchical_active_score"] = sticky_score

            df.loc[candidates.index, "position_gate"] = True
            df.loc[candidates.index, "position_gate_score"] = candidates["position_gate_score"]
            df.loc[candidates.index, "velocity_coherence_score"] = candidates["velocity_coherence_score"]
            df.loc[candidates.index, "playing_likelihood_score"] = candidates["playing_likelihood_score"]
            df.loc[candidates.index, "hierarchical_active_score"] = candidates["hierarchical_active_score"]

            score_by_name = candidates.set_index(cand_names)["playing_likelihood_score"].astype(float).to_dict()
            present_names = set(cand_names)
            if not selected_now:
                selected_now = set(
                    candidates.sort_values("hierarchical_active_score", ascending=False)
                    .head(int(max_active_players))[player_col]
                    .astype(str)
                )
            selected_now = {name for name in selected_now if name in present_names}

            while len(selected_now) < int(max_active_players):
                pool = candidates.loc[~cand_names.isin(selected_now)].copy()
                if pool.empty:
                    break
                add_name = str(pool.sort_values("hierarchical_active_score", ascending=False).iloc[0][player_col])
                selected_now.add(add_name)
                challenger_seconds[add_name] = 0.0

            for name in list(selected_now):
                current_score = float(score_by_name.get(name, 0.0))
                weak_seconds[name] = weak_seconds.get(name, 0.0) + step_s if current_score < weak_score_threshold else 0.0

            selected_score_min = min((score_by_name.get(name, -np.inf) for name in selected_now), default=-np.inf)
            weak_selected = [
                name
                for name in selected_now
                if weak_seconds.get(name, 0.0) >= float(weak_confirm_s) or name not in present_names
            ]
            challengers = candidates.loc[~cand_names.isin(selected_now)].copy()
            if not challengers.empty:
                challengers = challengers.sort_values("hierarchical_active_score", ascending=False)
                for _, row in challengers.iterrows():
                    cname = str(row[player_col])
                    cscore = float(row["playing_likelihood_score"])
                    challenger_seconds[cname] = (
                        challenger_seconds.get(cname, 0.0) + step_s
                        if cscore >= selected_score_min + float(switch_margin)
                        else 0.0
                    )

                while weak_selected:
                    ready = [
                        (name, challenger_seconds.get(name, 0.0), score_by_name.get(name, -np.inf))
                        for name in challengers[player_col].astype(str)
                        if challenger_seconds.get(name, 0.0) >= float(replacement_confirm_s)
                    ]
                    if not ready:
                        break
                    ready.sort(key=lambda item: (item[1], item[2]), reverse=True)
                    in_name = ready[0][0]
                    out_name = sorted(
                        weak_selected,
                        key=lambda name: (score_by_name.get(name, -np.inf), -weak_seconds.get(name, 0.0)),
                    )[0]
                    selected_now.discard(out_name)
                    selected_now.add(in_name)
                    weak_seconds[in_name] = 0.0
                    challenger_seconds[in_name] = 0.0
                    weak_selected = [
                        name
                        for name in selected_now
                        if weak_seconds.get(name, 0.0) >= float(weak_confirm_s)
                    ]

            selected_now = set(
                candidates.loc[cand_names.isin(selected_now)]
                .sort_values("hierarchical_active_score", ascending=False)
                .head(int(max_active_players))[player_col]
                .astype(str)
            )

        selected_mask = names.isin(selected_now)
        selected_idx = g.loc[selected_mask].index
        rejected_idx = g.loc[position_gate & ~selected_mask].index
        df.loc[selected_idx, label_col] = "active"
        df.loc[selected_idx, "playing_selected"] = True
        df.loc[rejected_idx, label_col] = "rejected_hoverer"

        entered = selected_now - selected_prev
        exited = selected_prev - selected_now
        if entered:
            df.loc[g.index[names.isin(entered)], "playing_selection_event"] = "hierarchical_selected"
        if exited:
            df.loc[g.index[names.isin(exited)], "playing_selection_event"] = "hierarchical_removed"

        selected_prev = selected_now
        weak_seconds = {name: weak_seconds.get(name, 0.0) for name in selected_prev}
        challenger_seconds = {
            name: val
            for name, val in challenger_seconds.items()
            if not candidates.empty and name in set(candidates[player_col].astype(str)) and name not in selected_prev
        }

    df.attrs["playing_likelihood_params"] = {
        "mode": "sticky_hierarchical_active_xi",
        "max_active_players": int(max_active_players),
        "likelihood_window_s": float(likelihood_window_s),
        "seed_window_s": float(seed_window_s),
        "seed_top_n": int(seed_top_n),
        "position_margin_m": float(position_margin_m),
        "coordinate_sanity_margin_m": float(coordinate_sanity_margin_m),
        "replacement_confirm_s": float(replacement_confirm_s),
        "weak_confirm_s": float(weak_confirm_s),
        "switch_margin": float(switch_margin),
        "weak_score_threshold": float(weak_score_threshold),
        "velocity_coherence_weight": float(velocity_coherence_weight),
        "seed_xi": sorted(seed_xi),
    }
    return df


def label_active_players(
    df_xy: pd.DataFrame,
    pitch_xy,
    player_col: str = "player_name",
    x_col: str = "x_m",
    y_col: str = "y_m",
    active_depth_m: float = 3.0,
    activate_s: float = 60.0,
    bench_off_s: float = 120.0,
    on_pitch_eps_m: float = 0.1,
    label_col: str = "player_status",
    keep_debug_cols: bool = False,
    active_method: str = "sticky_hierarchical_active_xi",
    max_active_players: int = 11,
    likelihood_window_s: float = 45.0,
    seed_window_s: float = 120.0,
    seed_top_n: int = 11,
    position_margin_m: float = 5.0,
    coordinate_sanity_margin_m: float = 30.0,
    replacement_confirm_s: float = 35.0,
    weak_confirm_s: float = 35.0,
    switch_margin: float = 0.75,
    weak_score_threshold: float = 1.6,
    velocity_coherence_weight: float = 1.5,
) -> pd.DataFrame:
    """
    Add per-row active-player labels.

    Default method is ``sticky_hierarchical_active_xi``:
      position gate -> continuity memory -> movement/velocity evidence ->
      velocity-coherence bonus -> delayed replacement. This rejects tracked
      hoverers while protecting goalkeepers and temporarily decorrelated players.
      ``coordinate_sanity_margin_m`` is a hard pitch-coordinate sanity gate:
      continuity can override the soft position margin, but not impossible
      coordinates far outside the calibrated pitch.

    Set ``active_method="geometry_state_machine"`` to recover the old behavior:
      - Bench -> Active:
          must be continuously in_active_zone for >= activate_s seconds.
      - Active -> Bench:
          must be continuously OFF pitch (not on_pitch) for >= bench_off_s seconds.

    Notes:
      - Uses per-player actual dt from timestamps (robust to irregular sampling).
      - Does NOT filter rows: it only labels. Filter afterwards if desired.
    """
    if player_col not in df_xy.columns:
        raise ValueError(f"df_xy must contain '{player_col}' column.")
    for c in (x_col, y_col):
        if c not in df_xy.columns:
            raise ValueError(f"df_xy must contain '{c}' column.")

    df = df_xy.copy()

    # --- time handling ---
    time_col, t = _choose_time_col(df)
    df[time_col] = t
    df = df.dropna(subset=[time_col])
    if df.empty:
        raise ValueError("No valid time values after parsing 'time'/'timestamp'.")

    # --- depth & flags ---
    df["signed_depth_from_edge"] = _compute_signed_depth_from_edges(df, pitch_xy, x_col=x_col, y_col=y_col)
    df["depth_from_edge"] = _compute_depth_from_edges(df, pitch_xy, x_col=x_col, y_col=y_col)
    df["on_pitch"] = df["signed_depth_from_edge"] >= float(on_pitch_eps_m)
    df["in_active_zone"] = df["signed_depth_from_edge"] >= float(active_depth_m)

    # --- sort & per-player dt ---
    df = df.sort_values([player_col, time_col])

    dt_s = (
        df.groupby(player_col, sort=False)[time_col]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .clip(lower=0.0)
    )
    df["_dt_s"] = dt_s

    # --- hysteresis per player ---
    df[label_col] = "bench"
    df["geometry_player_status"] = "bench"
    df["active_filter_event"] = ""

    # Iterate per player group efficiently
    for pid, idx in df.groupby(player_col, sort=False).groups.items():
        g = df.loc[idx]

        in_active = g["in_active_zone"].to_numpy(dtype=bool)
        on_pitch = g["on_pitch"].to_numpy(dtype=bool)
        dts = g["_dt_s"].to_numpy(dtype=float)

        status_arr = np.empty(len(g), dtype=object)
        event_arr = np.array([""] * len(g), dtype=object)
        current = "bench"
        time_in_active = 0.0
        time_off_pitch = 0.0

        for k in range(len(g)):
            dt = dts[k]

            if current == "bench":
                if in_active[k]:
                    time_in_active += dt
                else:
                    time_in_active = 0.0

                if time_in_active >= activate_s:
                    current = "active"
                    time_off_pitch = 0.0
                    event_arr[k] = "bench_to_active"

            else:  # active
                if not on_pitch[k]:
                    time_off_pitch += dt
                else:
                    time_off_pitch = 0.0

                if time_off_pitch >= bench_off_s:
                    current = "bench"
                    time_in_active = 0.0
                    event_arr[k] = "active_to_bench"

            status_arr[k] = current

        df.loc[idx, label_col] = status_arr
        df.loc[idx, "geometry_player_status"] = status_arr
        df.loc[idx, "active_filter_event"] = event_arr

    df = df.drop(columns=["_dt_s"])

    if active_method == "geometry_state_machine":
        pass
    elif active_method == "sticky_hierarchical_active_xi":
        df = _apply_sticky_hierarchical_active_xi(
            df,
            player_col=player_col,
            time_col=time_col,
            x_col=x_col,
            y_col=y_col,
            label_col=label_col,
            max_active_players=max_active_players,
            likelihood_window_s=likelihood_window_s,
            seed_window_s=seed_window_s,
            seed_top_n=seed_top_n,
            position_margin_m=position_margin_m,
            coordinate_sanity_margin_m=coordinate_sanity_margin_m,
            replacement_confirm_s=replacement_confirm_s,
            weak_confirm_s=weak_confirm_s,
            switch_margin=switch_margin,
            weak_score_threshold=weak_score_threshold,
            seed_bonus=0.35,
            continuity_bonus=2.5,
            movement_weight=0.10,
            depth_weight=0.025,
            geometry_active_bonus=0.25,
            in_zone_bonus=0.20,
            velocity_coherence_weight=velocity_coherence_weight,
            velocity_top_k=7,
            min_velocity_speed_mps=0.25,
        )
    else:
        raise ValueError("active_method must be 'sticky_hierarchical_active_xi' or 'geometry_state_machine'")

    # Store params so viz can auto-match overlay to classifier
    df.attrs["active_method"] = active_method
    df.attrs["active_depth_m"] = float(active_depth_m)
    df.attrs["activate_s"] = float(activate_s)
    df.attrs["bench_off_s"] = float(bench_off_s)
    df.attrs["on_pitch_eps_m"] = float(on_pitch_eps_m)
    df.attrs["label_col"] = label_col
    df.attrs["max_active_players"] = int(max_active_players)
    df.attrs["coordinate_sanity_margin_m"] = float(coordinate_sanity_margin_m)

    if not keep_debug_cols:
        debug_cols = [
            "signed_depth_from_edge",
            "depth_from_edge",
            "on_pitch",
            "in_active_zone",
            "geometry_player_status",
            "active_filter_event",
            "position_gate",
            "hard_position_valid",
            "position_gate_score",
            "velocity_coherence_score",
            "playing_likelihood_score",
            "hierarchical_active_score",
            "playing_seed_xi",
            "playing_selected",
            "playing_selection_event",
            "vx_mps",
            "vy_mps",
            "speed_mps",
            "recent_path_m",
            "recent_vx_mps",
            "recent_vy_mps",
            "recent_speed_mps",
        ]
        df = df.drop(columns=[c for c in debug_cols if c in df.columns])

    return df


def detect_substitutions(
    df_labeled: pd.DataFrame,
    player_col: str = "player_name",
    time_col: str = "timestamp",
    status_col: str = "player_status",
    active_value: str = "active",
) -> pd.DataFrame:
    """
    Simple utility: detect (bench->active) and (active->bench) transition times.
    Returns a table with player_name, event_time, event_type.
    """
    if time_col not in df_labeled.columns:
        # fallback
        time_col, _ = _choose_time_col(df_labeled)

    df = df_labeled.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values([player_col, time_col])

    out = []
    for pid, g in df.groupby(player_col, sort=False):
        s = g[status_col].astype(str).to_numpy()
        t = g[time_col].to_numpy()
        prev = None
        for i in range(len(s)):
            cur = s[i]
            if prev is None:
                prev = cur
                continue
            if prev != cur:
                if prev != active_value and cur == active_value:
                    out.append((pid, t[i], "on"))
                elif prev == active_value and cur != active_value:
                    out.append((pid, t[i], "off"))
            prev = cur

    return pd.DataFrame(out, columns=[player_col, "event_time", "event_type"])
