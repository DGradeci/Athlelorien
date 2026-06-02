# src/utils/trajectory_stats.py
from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------
# Core path building (active-only)
# -----------------------------
def build_paths_from_active(
    df_active: pd.DataFrame,
    player_col: str = "player_name",
    time_col: str = "timestamp",   # preferred
    half_col: str = "half",        # optional; if present we force splits across halves
    phase_col: str = "match_phase",# optional; used if half_col not present (or alongside if you prefer)
    max_gap_s: float = 2.0,
    min_len: int = 30,             # minimum samples per path (e.g. 30s at 1 Hz)
) -> pd.DataFrame:
    """
    Given df_active (already filtered to active rows + match time),
    split into continuous paths per player and avoid mixing across discontinuities.

    Discontinuities that trigger a new path:
      - time gap > max_gap_s
      - change in half (if half_col exists)
      - else change in match_phase (if phase_col exists)

    Adds:
      - _t : parsed datetime
      - path_uid : unique path id string per player/path
    """
    df = df_active.copy()

    if time_col not in df.columns:
        raise ValueError(f"df_active must contain '{time_col}'")

    df["_t"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["_t"])

    # choose phase splitter
    split_col = None
    if half_col in df.columns:
        split_col = half_col
    elif phase_col in df.columns:
        split_col = phase_col

    sort_cols = [player_col, "_t"] if split_col is None else [player_col, split_col, "_t"]
    df = df.sort_values(sort_cols)

    dt = df.groupby(player_col)["_t"].diff().dt.total_seconds().fillna(0.0)
    new_path = dt.gt(float(max_gap_s))

    if split_col is not None:
        new_path = new_path | df[split_col].ne(df.groupby(player_col)[split_col].shift())

    path_n = new_path.groupby(df[player_col]).cumsum().astype(int)
    df["path_uid"] = df[player_col].astype(str) + "_p" + path_n.astype(str)

    counts = df["path_uid"].value_counts()
    keep = counts[counts >= int(min_len)].index
    return df[df["path_uid"].isin(keep)].copy()


# -----------------------------
# Per-sample steps (micro-steps)
# -----------------------------
def add_steps(
    df_paths: pd.DataFrame,
    x: str = "x_m",
    y: str = "y_m",
    t: str = "_t",
    path_col: str = "path_uid",
) -> pd.DataFrame:
    """
    Adds per-sample steps within each path:
      dx, dy, dt_s, step_m
    """
    df = df_paths.sort_values([path_col, t]).copy()
    df["dx"] = df.groupby(path_col)[x].diff()
    df["dy"] = df.groupby(path_col)[y].diff()
    df["dt_s"] = df.groupby(path_col)[t].diff().dt.total_seconds()
    df["step_m"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    return df.dropna(subset=["step_m", "dt_s"])


# -----------------------------
# MSD (within-path pairs only)
# -----------------------------
def msd_by_lag(
    df_paths: pd.DataFrame,
    max_k: int = 120,
    x: str = "x_m",
    y: str = "y_m",
    t: str = "_t",
    player_col: str = "player_name",
    path_col: str = "path_uid",
) -> pd.DataFrame:
    """
    MSD per entity using all within-path pairs (i, i+k), never mixing across paths.

    Returns columns:
      [player_col, k, tau_s, msd]
    """
    rows = []
    df = df_paths.sort_values([player_col, path_col, t])

    for (ent, pid), g in df.groupby([player_col, path_col], sort=False):
        X = g[[x, y]].to_numpy()
        tt = g[t].to_numpy()
        n = len(g)
        Kmax = min(int(max_k), n - 1)
        if Kmax < 1:
            continue

        for k in range(1, Kmax + 1):
            dX = X[k:] - X[:-k]
            msd = float(np.mean(np.sum(dX * dX, axis=1)))
            tau = float(np.mean((tt[k:] - tt[:-k]) / np.timedelta64(1, "s")))
            rows.append((ent, k, tau, msd))

    res = pd.DataFrame(rows, columns=[player_col, "k", "tau_s", "msd"])
    return res.groupby([player_col, "k"], as_index=False).agg(
        tau_s=("tau_s", "mean"),
        msd=("msd", "mean"),
    )


# -----------------------------
# Turning-based run segmentation (your "steps of runs")
# -----------------------------
def compute_turn_runs(
    df_paths: pd.DataFrame,
    player_col: str = "player_name",
    path_col: str = "path_uid",
    x_col: str = "x_m",
    y_col: str = "y_m",
    t_col: str = "_t",
    theta_deg: float = 30.0,
    min_run_samples: int = 2,
    min_disp_m: float = 1e-6,
) -> pd.DataFrame:
    """
    Segment each (player, path) into 'runs' based on a turning-angle threshold.

    Rule:
      - Track successive displacement vectors v_i = x_i - x_{i-1}.
      - When the angle between v_{i-1} and v_i exceeds theta_deg, close the current run.

    Returns a run-level table with:
      player_name, path_uid, run_id,
      t_start, t_end, duration_s,
      run_length_m (ARC LENGTH), v_mean_mps
    """
    if t_col not in df_paths.columns:
        raise ValueError(f"df_paths must contain '{t_col}' (expected parsed datetime).")

    theta_rad = float(np.deg2rad(theta_deg))
    out = []

    df = df_paths.sort_values([player_col, path_col, t_col])

    for (pid, pth), g in df.groupby([player_col, path_col], sort=False):
        xy = g[[x_col, y_col]].to_numpy(dtype=float)
        tt = g[t_col].to_numpy()
        n = len(g)
        if n < 2:
            continue

        # per-sample displacements and norms
        dxy = xy[1:] - xy[:-1]
        dnorm = np.linalg.norm(dxy, axis=1)

        run_start = 0  # index in xy
        prev_vec = None
        run_id = 0

        for i in range(1, n):  # i indexes xy
            v = xy[i] - xy[i - 1]
            nv = float(np.linalg.norm(v))
            if nv <= min_disp_m:
                continue

            if prev_vec is None:
                prev_vec = v
                continue

            pv = prev_vec
            npv = float(np.linalg.norm(pv))
            if npv <= min_disp_m:
                prev_vec = v
                continue

            cosang = float(np.dot(pv, v)) / (npv * nv)
            cosang = float(np.clip(cosang, -1.0, 1.0))
            ang = float(np.arccos(cosang))

            if ang > theta_rad:
                # close run: indices [run_start .. i-1]
                s = run_start
                e = i - 1
                if (e - s + 1) >= int(min_run_samples):
                    # arc length = sum of step distances along the run
                    L = float(np.nansum(dnorm[s:e]))  # dnorm index aligns with segments between points
                    t0 = tt[s]
                    t1 = tt[e]
                    dur = float((t1 - t0) / np.timedelta64(1, "s"))
                    if L > 0 and dur >= 0:
                        vmean = float(L / dur) if dur > 0 else np.nan
                        out.append((pid, pth, run_id, t0, t1, dur, L, vmean))

                run_id += 1
                run_start = i - 1
                prev_vec = v
            else:
                prev_vec = v

        # close final run: [run_start .. n-1]
        s = run_start
        e = n - 1
        if (e - s + 1) >= int(min_run_samples):
            L = float(np.nansum(dnorm[s:e]))
            t0 = tt[s]
            t1 = tt[e]
            dur = float((t1 - t0) / np.timedelta64(1, "s"))
            if L > 0 and dur >= 0:
                vmean = float(L / dur) if dur > 0 else np.nan
                out.append((pid, pth, run_id, t0, t1, dur, L, vmean))

    return pd.DataFrame(
        out,
        columns=[
            player_col,
            path_col,
            "run_id",
            "t_start",
            "t_end",
            "duration_s",
            "run_length_m",
            "v_mean_mps",
        ],
    )


# Optional convenience: return only run lengths (legacy name)
def compute_step_lengths(
    df_paths: pd.DataFrame,
    player_col: str = "player_name",
    path_col: str = "path_uid",
    x_col: str = "x_m",
    y_col: str = "y_m",
    t_col: str = "_t",
    theta_deg: float = 30.0,
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper: returns columns [player_col, path_col, step_length]
    where step_length is run_length_m from turning-based segmentation.
    """
    runs = compute_turn_runs(
        df_paths,
        player_col=player_col,
        path_col=path_col,
        x_col=x_col,
        y_col=y_col,
        t_col=t_col,
        theta_deg=theta_deg,
    )
    return runs.rename(columns={"run_length_m": "step_length"})[
        [player_col, path_col, "step_length"]
    ]


# -----------------------------
# Distribution helpers
# -----------------------------
def ccdf(x) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def log_pdf(x, nbins: int = 25, xmin=None, xmax=None):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) == 0:
        return np.array([]), np.array([])

    if xmin is None:
        xmin = float(np.min(x))
    if xmax is None:
        xmax = float(np.max(x))
    if xmax <= xmin:
        return np.array([]), np.array([])

    edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    counts, _ = np.histogram(x, bins=edges)
    widths = edges[1:] - edges[:-1]
    centers = np.sqrt(edges[1:] * edges[:-1])

    pdf = counts / (np.sum(counts) * widths)
    m = counts > 0
    return centers[m], pdf[m]

def build_player_maps(
    msd_player: pd.DataFrame,
    runs: pd.DataFrame,
    player_col: str = "player_name",
    runlen_col: str = "run_length_m",
    nbins: int = 25,
):
    """
    Precompute and store per-player:
      - msd_map:  tau_s vs msd
      - pdf_map:  log-binned PDF of run lengths
      - ccdf_map: CCDF of run lengths
    Also returns long tables + per-player metadata.
    """

    # ---- MSD map (tau_s vs msd) ----
    msd_map = {}
    if msd_player is not None and len(msd_player) > 0:
        for p, g in msd_player.groupby(player_col, sort=False):
            msd_map[p] = g.sort_values("tau_s")[["tau_s", "msd"]].reset_index(drop=True).copy()

    # ---- Run-length distributions (PDF + CCDF) ----
    pdf_parts = []
    ccdf_parts = []
    meta_rows = []

    if runs is None or len(runs) == 0:
        pdf_df = pd.DataFrame(columns=[player_col, "x", "pdf"])
        ccdf_df = pd.DataFrame(columns=[player_col, "x", "ccdf"])
        meta_df = pd.DataFrame(columns=[player_col, "n_runs", "min_L", "max_L"])
        return msd_map, {}, {}, pdf_df, ccdf_df, meta_df

    for p, g in runs.groupby(player_col, sort=False):
        L = g[runlen_col].to_numpy(dtype=float)
        L = L[np.isfinite(L)]
        L = L[L > 0]

        meta_rows.append(
            {
                player_col: p,
                "n_runs": int(len(L)),
                "min_L": float(L.min()) if len(L) else np.nan,
                "max_L": float(L.max()) if len(L) else np.nan,
            }
        )

        if len(L) == 0:
            continue

        bx, bp = log_pdf(L, nbins=nbins)
        if len(bx):
            pdf_parts.append(pd.DataFrame({player_col: p, "x": bx, "pdf": bp}))

        xs, ys = ccdf(L)
        if len(xs):
            ccdf_parts.append(pd.DataFrame({player_col: p, "x": xs, "ccdf": ys}))

    pdf_df = (
        pd.concat(pdf_parts, ignore_index=True)
        if pdf_parts
        else pd.DataFrame(columns=[player_col, "x", "pdf"])
    )
    ccdf_df = (
        pd.concat(ccdf_parts, ignore_index=True)
        if ccdf_parts
        else pd.DataFrame(columns=[player_col, "x", "ccdf"])
    )
    meta_df = pd.DataFrame(meta_rows)

    pdf_map = {
        p: d[["x", "pdf"]].reset_index(drop=True).copy()
        for p, d in pdf_df.groupby(player_col, sort=False)
    }
    ccdf_map = {
        p: d[["x", "ccdf"]].reset_index(drop=True).copy()
        for p, d in ccdf_df.groupby(player_col, sort=False)
    }

    return msd_map, pdf_map, ccdf_map, pdf_df, ccdf_df, meta_df



# -----------------------------
# Extra distribution maps (duration, speed, etc.)
# -----------------------------
def build_value_maps(
    runs: pd.DataFrame,
    player_col: str,
    value_col: str,
    nbins: int = 25,
    positive_only: bool = True,
):
    """
    Per-player log-binned PDF + CCDF for an arbitrary run-level value column
    (e.g. duration_s, v_mean_mps).

    Returns:
      value_pdf_map, value_ccdf_map, value_pdf_df, value_ccdf_df, value_meta_df
    """
    pdf_parts = []
    ccdf_parts = []
    meta_rows = []

    if runs is None or len(runs) == 0 or value_col not in runs.columns:
        empty_pdf = pd.DataFrame(columns=[player_col, "x", "pdf"])
        empty_ccdf = pd.DataFrame(columns=[player_col, "x", "ccdf"])
        empty_meta = pd.DataFrame(columns=[player_col, "n", "min", "max"])
        return {}, {}, empty_pdf, empty_ccdf, empty_meta

    for p, g in runs.groupby(player_col, sort=False):
        x = g[value_col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if positive_only:
            x = x[x > 0]

        meta_rows.append(
            {
                player_col: p,
                "n": int(len(x)),
                "min": float(x.min()) if len(x) else np.nan,
                "max": float(x.max()) if len(x) else np.nan,
            }
        )

        if len(x) == 0:
            continue

        bx, bp = log_pdf(x, nbins=nbins)
        if len(bx):
            pdf_parts.append(pd.DataFrame({player_col: p, "x": bx, "pdf": bp}))

        xs, ys = ccdf(x)
        if len(xs):
            ccdf_parts.append(pd.DataFrame({player_col: p, "x": xs, "ccdf": ys}))

    pdf_df = (
        pd.concat(pdf_parts, ignore_index=True)
        if pdf_parts
        else pd.DataFrame(columns=[player_col, "x", "pdf"])
    )
    ccdf_df = (
        pd.concat(ccdf_parts, ignore_index=True)
        if ccdf_parts
        else pd.DataFrame(columns=[player_col, "x", "ccdf"])
    )
    meta_df = pd.DataFrame(meta_rows)

    pdf_map = {
        p: d[["x", "pdf"]].reset_index(drop=True).copy()
        for p, d in pdf_df.groupby(player_col, sort=False)
    }
    ccdf_map = {
        p: d[["x", "ccdf"]].reset_index(drop=True).copy()
        for p, d in ccdf_df.groupby(player_col, sort=False)
    }

    return pdf_map, ccdf_map, pdf_df, ccdf_df, meta_df


def _infer_group_keys(df: pd.DataFrame, team_col: str | None, phase_col: str | None) -> list[str]:
    keys: list[str] = []
    if team_col and team_col in df.columns:
        keys.append(team_col)
    if phase_col and phase_col in df.columns:
        keys.append(phase_col)
    return keys


def attach_centroid_to_player_paths(
    df_paths: pd.DataFrame,
    centroid_ts: pd.DataFrame,
    t_col: str = "_t",
    x_col: str = "x_m",
    y_col: str = "y_m",
    team_col: str | None = None,
    phase_col: str | None = "match_phase",
    out_xc: str = "x_centroid",
    out_yc: str = "y_centroid",
) -> pd.DataFrame:
    """
    Merge centroid coordinates onto each player sample (same time + same team/phase if available).
    """
    df = df_paths.copy()
    c = centroid_ts.copy()

    df[t_col] = pd.to_datetime(df[t_col], errors="coerce")
    c[t_col] = pd.to_datetime(c[t_col], errors="coerce")

    keys = _infer_group_keys(df, team_col, phase_col)
    join_cols = keys + [t_col]

    # rename centroid coordinates to avoid clobbering
    c = c.rename(columns={x_col: out_xc, y_col: out_yc})

    # keep only join cols + centroid coords
    keep_cols = join_cols + [out_xc, out_yc]
    c = c[keep_cols].drop_duplicates(subset=join_cols)

    merged = df.merge(c, on=join_cols, how="left", validate="m:1")
    return merged


def analyse_paths_df(
    df_paths: pd.DataFrame,
    player_col: str,
    path_col: str,
    x_col: str,
    y_col: str,
    t_col: str,
    theta_deg: float,
    max_k: int,
    nbins: int,
) -> dict:
    """
    Shared analysis core used for players / centroid / relative frame.
    """
    msd_tbl = msd_by_lag(
        df_paths,
        max_k=max_k,
        x=x_col,
        y=y_col,
        t=t_col,
        player_col=player_col,
        path_col=path_col,
    )

    runs_tbl = compute_turn_runs(
        df_paths,
        player_col=player_col,
        path_col=path_col,
        x_col=x_col,
        y_col=y_col,
        t_col=t_col,
        theta_deg=theta_deg,
    )

    msd_map, pdf_map, ccdf_map, pdf_df, ccdf_df, meta_df = build_player_maps(
        msd_player=msd_tbl,
        runs=runs_tbl,
        player_col=player_col,
        runlen_col="run_length_m",
        nbins=nbins,
    )

    # Extra (physicsy) distributions: duration + mean speed
    dur_pdf_map, dur_ccdf_map, dur_pdf_df, dur_ccdf_df, dur_meta_df = build_value_maps(
        runs_tbl, player_col=player_col, value_col="duration_s", nbins=nbins
    )
    spd_pdf_map, spd_ccdf_map, spd_pdf_df, spd_ccdf_df, spd_meta_df = build_value_maps(
        runs_tbl, player_col=player_col, value_col="v_mean_mps", nbins=nbins
    )

    return {
        "msd": msd_tbl,
        "runs": runs_tbl,
        "msd_map": msd_map,
        "pdf_map": pdf_map,
        "ccdf_map": ccdf_map,
        "pdf_df": pdf_df,
        "ccdf_df": ccdf_df,
        "meta_df": meta_df,
        # duration distributions
        "dur_pdf_map": dur_pdf_map,
        "dur_ccdf_map": dur_ccdf_map,
        "dur_pdf_df": dur_pdf_df,
        "dur_ccdf_df": dur_ccdf_df,
        "dur_meta_df": dur_meta_df,
        # speed distributions
        "spd_pdf_map": spd_pdf_map,
        "spd_ccdf_map": spd_ccdf_map,
        "spd_pdf_df": spd_pdf_df,
        "spd_ccdf_df": spd_ccdf_df,
        "spd_meta_df": spd_meta_df,
    }


def analyse_active_match(
    df_active: pd.DataFrame,
    theta_deg: float = 35.0,
    max_k: int = 120,
    max_gap_s: float = 2.0,
    min_len: int = 30,
    nbins: int = 25,
    player_col: str = "player_name",
    time_col: str = "timestamp",
    half_col: str = "half",
    phase_col: str = "match_phase",
    team_col: str | None = None,
    x_col: str = "x_m",
    y_col: str = "y_m",
    min_players_centroid: int = 7,
) -> dict:
    """
    One-stop pipeline on df_active that returns:
      - player df_paths + (msd/pdf/ccdf)
      - centroid_ts + df_centroid + (msd/pdf/ccdf)
      - player-relative-to-centroid df_paths_rel + (msd/pdf/ccdf)

    Notes:
      - Path splitting uses half_col if available, else match_phase if available.
      - For head-to-head, set team_col="team" to keep per-team centroids separate.
    """

    # ---------- 1) player paths ----------
    df_paths = build_paths_from_active(
        df_active,
        player_col=player_col,
        time_col=time_col,
        half_col=half_col,
        phase_col=phase_col,
        max_gap_s=max_gap_s,
        min_len=min_len,
    )

    # auto-detect team_col if not supplied
    if team_col is None and "team" in df_paths.columns:
        team_col = "team"

    players_out = analyse_paths_df(
        df_paths=df_paths,
        player_col=player_col,
        path_col="path_uid",
        x_col=x_col,
        y_col=y_col,
        t_col="_t",
        theta_deg=theta_deg,
        max_k=max_k,
        nbins=nbins,
    )

    # ---------- 2) centroid time series + centroid paths ----------
    centroid_ts = build_team_centroid_timeseries(
        df_paths,
        t_col="_t",
        x_col=x_col,
        y_col=y_col,
        player_col=player_col,
        phase_col=phase_col,
        team_col=team_col,
        min_players=min_players_centroid,
    )

    df_centroid = build_centroid_paths(
        centroid_ts,
        t_col="_t",
        x_col=x_col,
        y_col=y_col,
        phase_col=phase_col,
        team_col=team_col,
        max_gap_s=max_gap_s,
        min_len=min_len,
        entity_col="entity",
        entity_name="centroid",
        path_col="path_uid",
    )

    # if multiple teams exist, make unique entity labels per team
    if team_col is not None and team_col in df_centroid.columns:
        df_centroid = df_centroid.copy()
        df_centroid["entity"] = "centroid:" + df_centroid[team_col].astype(str)

    centroid_out = analyse_paths_df(
        df_paths=df_centroid,
        player_col="entity",
        path_col="path_uid",
        x_col=x_col,
        y_col=y_col,
        t_col="_t",
        theta_deg=theta_deg,
        max_k=max_k,
        nbins=nbins,
    )

    # ---------- 3) players in centroid frame ----------
    df_paths_c = attach_centroid_to_player_paths(
        df_paths,
        centroid_ts,
        t_col="_t",
        x_col=x_col,
        y_col=y_col,
        team_col=team_col,
        phase_col=phase_col,
        out_xc="x_centroid",
        out_yc="y_centroid",
    )

    # compute relative coords; drop rows where centroid missing
    df_paths_rel = df_paths_c.dropna(subset=["x_centroid", "y_centroid"]).copy()
    df_paths_rel["x_rel"] = df_paths_rel[x_col] - df_paths_rel["x_centroid"]
    df_paths_rel["y_rel"] = df_paths_rel[y_col] - df_paths_rel["y_centroid"]

    rel_out = analyse_paths_df(
        df_paths=df_paths_rel,
        player_col=player_col,
        path_col="path_uid",
        x_col="x_rel",
        y_col="y_rel",
        t_col="_t",
        theta_deg=theta_deg,
        max_k=max_k,
        nbins=nbins,
    )

    # ---------- return: keep old top-level keys for backward compatibility ----------
    out = {
        # players (backward compatible)
        "df_paths": df_paths,
        "msd_player": players_out["msd"],
        "runs": players_out["runs"],
        "msd_map": players_out["msd_map"],
        "pdf_map": players_out["pdf_map"],
        "ccdf_map": players_out["ccdf_map"],
        "pdf_df": players_out["pdf_df"],
        "ccdf_df": players_out["ccdf_df"],
        "meta_df": players_out["meta_df"],
        "dur_pdf_map": players_out.get("dur_pdf_map", {}),
        "dur_ccdf_map": players_out.get("dur_ccdf_map", {}),
        "dur_pdf_df": players_out.get("dur_pdf_df", pd.DataFrame()),
        "dur_ccdf_df": players_out.get("dur_ccdf_df", pd.DataFrame()),
        "dur_meta_df": players_out.get("dur_meta_df", pd.DataFrame()),
        "spd_pdf_map": players_out.get("spd_pdf_map", {}),
        "spd_ccdf_map": players_out.get("spd_ccdf_map", {}),
        "spd_pdf_df": players_out.get("spd_pdf_df", pd.DataFrame()),
        "spd_ccdf_df": players_out.get("spd_ccdf_df", pd.DataFrame()),
        "spd_meta_df": players_out.get("spd_meta_df", pd.DataFrame()),


        # centroid
        "centroid_ts": centroid_ts,
        "df_centroid": df_centroid,
        "centroid_msd": centroid_out["msd"],
        "centroid_runs": centroid_out["runs"],
        "centroid_msd_map": centroid_out["msd_map"],
        "centroid_pdf_map": centroid_out["pdf_map"],
        "centroid_ccdf_map": centroid_out["ccdf_map"],
        "centroid_pdf_df": centroid_out["pdf_df"],
        "centroid_ccdf_df": centroid_out["ccdf_df"],
        "centroid_meta_df": centroid_out["meta_df"],
        "centroid_dur_pdf_map": centroid_out.get("dur_pdf_map", {}),
        "centroid_dur_ccdf_map": centroid_out.get("dur_ccdf_map", {}),
        "centroid_dur_pdf_df": centroid_out.get("dur_pdf_df", pd.DataFrame()),
        "centroid_dur_ccdf_df": centroid_out.get("dur_ccdf_df", pd.DataFrame()),
        "centroid_dur_meta_df": centroid_out.get("dur_meta_df", pd.DataFrame()),
        "centroid_spd_pdf_map": centroid_out.get("spd_pdf_map", {}),
        "centroid_spd_ccdf_map": centroid_out.get("spd_ccdf_map", {}),
        "centroid_spd_pdf_df": centroid_out.get("spd_pdf_df", pd.DataFrame()),
        "centroid_spd_ccdf_df": centroid_out.get("spd_ccdf_df", pd.DataFrame()),
        "centroid_spd_meta_df": centroid_out.get("spd_meta_df", pd.DataFrame()),


        # players relative to centroid
        "df_paths_rel": df_paths_rel,
        "rel_msd": rel_out["msd"],
        "rel_runs": rel_out["runs"],
        "rel_msd_map": rel_out["msd_map"],
        "rel_pdf_map": rel_out["pdf_map"],
        "rel_ccdf_map": rel_out["ccdf_map"],
        "rel_pdf_df": rel_out["pdf_df"],
        "rel_ccdf_df": rel_out["ccdf_df"],
        "rel_meta_df": rel_out["meta_df"],
        "rel_dur_pdf_map": rel_out.get("dur_pdf_map", {}),
        "rel_dur_ccdf_map": rel_out.get("dur_ccdf_map", {}),
        "rel_dur_pdf_df": rel_out.get("dur_pdf_df", pd.DataFrame()),
        "rel_dur_ccdf_df": rel_out.get("dur_ccdf_df", pd.DataFrame()),
        "rel_dur_meta_df": rel_out.get("dur_meta_df", pd.DataFrame()),
        "rel_spd_pdf_map": rel_out.get("spd_pdf_map", {}),
        "rel_spd_ccdf_map": rel_out.get("spd_ccdf_map", {}),
        "rel_spd_pdf_df": rel_out.get("spd_pdf_df", pd.DataFrame()),
        "rel_spd_ccdf_df": rel_out.get("spd_ccdf_df", pd.DataFrame()),
        "rel_spd_meta_df": rel_out.get("spd_meta_df", pd.DataFrame()),
    }
    return out

def build_run_heading_and_microturn_df(
    df_paths: pd.DataFrame,
    player_col: str = "player_name",
    path_col: str = "path_uid",
    x_col: str = "x_m",
    y_col: str = "y_m",
    t_col: str = "_t",
    theta_deg: float = 30.0,
    min_run_samples: int = 2,
    min_disp_m: float = 1e-6,
):
    """
    Returns:
      step_df: run-level segments defined by turning threshold theta_deg
        columns:
          player_col, path_col, run_id,
          step_start_idx, step_end_idx,
          step_length_m (ARC LENGTH),
          heading_rad, heading_deg,
          heading_weight (= step_length_m)

      micro_turn_df: frame-to-frame turning angles (micro Δθ), signed and wrapped to (-pi, pi]
        columns:
          player_col, path_col,
          turning_angle_rad, turning_angle_deg
    """
    if t_col not in df_paths.columns:
        raise ValueError(f"df_paths must contain '{t_col}' (parsed datetime).")

    theta_rad = float(np.deg2rad(theta_deg))
    run_rows = []
    turn_rows = []

    df = df_paths.sort_values([player_col, path_col, t_col])

    for (pid, pth), g in df.groupby([player_col, path_col], sort=False):
        xy = g[[x_col, y_col]].to_numpy(dtype=float)
        idxs = g.index.to_numpy()
        n = len(g)
        if n < 2:
            continue

        # micro-step vectors and norms
        dxy = xy[1:] - xy[:-1]
        dnorm = np.linalg.norm(dxy, axis=1)

        # ---------- 1) RUNS (turn threshold) ----------
        run_start = 0
        prev_vec = None
        run_id = 0

        def _close_run(s, e, run_id):
            # close run over indices [s..e] in xy
            if (e - s + 1) < int(min_run_samples):
                return
            # arc length: sum of per-step norms within [s..e]
            L = float(np.nansum(dnorm[s:e]))  # dnorm index aligns with segments
            if L <= 0:
                return
            # heading: chord direction from start to end
            chord = xy[e] - xy[s]
            if float(np.linalg.norm(chord)) <= min_disp_m:
                return
            heading_rad = float(np.arctan2(chord[1], chord[0]))
            run_rows.append({
                player_col: pid,
                path_col: pth,
                "run_id": int(run_id),
                "step_start_idx": int(idxs[s]),
                "step_end_idx": int(idxs[e]),
                "step_length_m": L,
                "heading_rad": heading_rad,
                "heading_deg": float(np.degrees(heading_rad)),
                "heading_weight": L,
            })

        for i in range(1, n):
            v = xy[i] - xy[i-1]
            nv = float(np.linalg.norm(v))
            if nv <= min_disp_m:
                continue

            if prev_vec is None:
                prev_vec = v
                continue

            pv = prev_vec
            npv = float(np.linalg.norm(pv))
            if npv <= min_disp_m:
                prev_vec = v
                continue

            cosang = float(np.dot(pv, v)) / (npv * nv)
            cosang = float(np.clip(cosang, -1.0, 1.0))
            ang = float(np.arccos(cosang))

            if ang > theta_rad:
                _close_run(run_start, i-1, run_id)
                run_id += 1
                run_start = i - 1
                prev_vec = v
            else:
                prev_vec = v

        # close final run
        _close_run(run_start, n-1, run_id)

        # ---------- 2) MICRO TURNING ANGLES (frame-to-frame) ----------
        if n < 3:
            continue

        v = dxy
        nv = dnorm

        for i in range(len(v) - 1):
            v1, v2 = v[i], v[i+1]
            if nv[i] <= min_disp_m or nv[i+1] <= min_disp_m:
                continue

            dot = float(np.dot(v1, v2))
            denom = float(nv[i] * nv[i+1])
            cosang = float(np.clip(dot / denom, -1.0, 1.0))
            dtheta = float(np.arccos(cosang))

            # sign via 2D cross product
            cross = float(v1[0]*v2[1] - v1[1]*v2[0])
            if cross < 0:
                dtheta = -dtheta

            # wrap to (-pi, pi]
            dtheta = float((dtheta + np.pi) % (2*np.pi) - np.pi)

            turn_rows.append({
                player_col: pid,
                path_col: pth,
                "turning_angle_rad": dtheta,
                "turning_angle_deg": float(np.degrees(dtheta)),
            })

    step_df = pd.DataFrame(run_rows)
    micro_turn_df = pd.DataFrame(turn_rows)
    return step_df, micro_turn_df
import pandas as pd

def wrap_pi(theta):
    theta = np.asarray(theta, dtype=float)
    return (theta + np.pi) % (2*np.pi) - np.pi


def build_run_heading_and_microturn_df_from_paths(
    df_paths,
    player_col="player_name",
    path_col="path_uid",
    x_col="x_m",
    y_col="y_m",
    t_col="_t",
    theta_deg=35.0,
    phase_col="match_phase",
    min_disp_m=1e-6,
    min_run_samples=2,
):
    """
    From df_paths (already time-ordered, continuous paths), build:
      - step_seg_df: run-level segments (turn-threshold), with heading + arc length + match_phase
      - micro_turn_df: frame-to-frame signed turning angles
    """
    theta_rad = float(np.deg2rad(theta_deg))
    run_rows, turn_rows = [], []

    df = df_paths.sort_values([player_col, path_col, t_col])

    for (pid, pth), g in df.groupby([player_col, path_col], sort=False):
        xy = g[[x_col, y_col]].to_numpy(float)
        idxs = g.index.to_numpy()
        phase = g[phase_col].astype(str).to_numpy() if phase_col in g.columns else None

        n = len(g)
        if n < 2:
            continue

        dxy = xy[1:] - xy[:-1]
        dnorm = np.linalg.norm(dxy, axis=1)

        # ----- runs via turning threshold -----
        run_start = 0
        prev_vec = None
        run_id = 0

        def close_run(s, e, run_id):
            if (e - s + 1) < int(min_run_samples):
                return
            L = float(np.nansum(dnorm[s:e]))  # arc length
            if L <= 0:
                return
            chord = xy[e] - xy[s]
            if float(np.linalg.norm(chord)) <= min_disp_m:
                return
            heading = float(np.arctan2(chord[1], chord[0]))
            row = {
                player_col: pid,
                path_col: pth,
                "run_id": int(run_id),
                "step_start_idx": int(idxs[s]),
                "step_end_idx": int(idxs[e]),
                "step_length_m": L,
                "heading_rad": heading,
                "heading_deg": float(np.degrees(heading)),
                "heading_weight": L,
            }
            if phase is not None:
                row[phase_col] = str(phase[s])  # phase at run start
            run_rows.append(row)

        for i in range(1, n):
            v = xy[i] - xy[i-1]
            nv = float(np.linalg.norm(v))
            if nv <= min_disp_m:
                continue
            if prev_vec is None:
                prev_vec = v
                continue

            pv = prev_vec
            npv = float(np.linalg.norm(pv))
            if npv <= min_disp_m:
                prev_vec = v
                continue

            cosang = float(np.dot(pv, v) / (npv * nv))
            cosang = float(np.clip(cosang, -1.0, 1.0))
            ang = float(np.arccos(cosang))

            if ang > theta_rad:
                close_run(run_start, i-1, run_id)
                run_id += 1
                run_start = i - 1
                prev_vec = v
            else:
                prev_vec = v

        close_run(run_start, n-1, run_id)

        # ----- micro turning angles -----
        if n < 3:
            continue

        v = dxy
        nv = dnorm
        for i in range(len(v) - 1):
            v1, v2 = v[i], v[i+1]
            if nv[i] <= min_disp_m or nv[i+1] <= min_disp_m:
                continue

            cosang = float(np.dot(v1, v2) / (nv[i] * nv[i+1]))
            cosang = float(np.clip(cosang, -1.0, 1.0))
            dtheta = float(np.arccos(cosang))

            cross = float(v1[0]*v2[1] - v1[1]*v2[0])
            if cross < 0:
                dtheta = -dtheta

            dtheta = float((dtheta + np.pi) % (2*np.pi) - np.pi)

            row = {
                player_col: pid,
                path_col: pth,
                "turning_angle_rad": dtheta,
                "turning_angle_deg": float(np.degrees(dtheta)),
            }
            if phase is not None:
                row[phase_col] = str(phase[i])  # phase at that moment
            turn_rows.append(row)

    return pd.DataFrame(run_rows), pd.DataFrame(turn_rows)

def combine_halves_headings(
    step_df: pd.DataFrame,
    heading_col: str = "heading_rad",
    phase_col: str = "match_phase",
    phase_2h: str = "2H",
    out_col: str = "heading_combined_rad",
) -> pd.DataFrame:
    """
    Combine 1H+2H by mapping 2H headings into the 1H frame (x -> -x):
        theta' = wrap_pi(pi - theta) for 2H, else theta.

    Useful for rose plots and signed stats. For A=<cos(2θ)>, optional.
    """
    if phase_col not in step_df.columns:
        raise ValueError(f"step_df must contain '{phase_col}' to combine halves.")
    df = step_df.copy()

    th = df[heading_col].to_numpy(dtype=float)
    ph = df[phase_col].astype(str).to_numpy()

    th2 = th.copy()
    m2 = (ph == str(phase_2h))
    th2[m2] = wrap_pi(np.pi - th2[m2])

    df[out_col] = th2
    return df


def anisotropy_A_cos2(theta_rad: np.ndarray, weights: np.ndarray | None = None) -> float:
    """
    Axial anisotropy along x-axis:
        A = <cos(2θ)>
    +x and -x are equivalent (axis metric).
    """
    theta = np.asarray(theta_rad, dtype=float)
    x = np.cos(2.0 * theta)

    if weights is None:
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    w = np.asarray(weights, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def bootstrap_ci_A_cos2(
    theta_rad: np.ndarray,
    weights: np.ndarray | None = None,
    B: int = 500,
    ci: tuple[float, float] = (0.025, 0.975),
    seed: int = 0,
) -> tuple[float, float]:
    """
    Bootstrap CI for A=<cos(2θ)> by resampling angles with replacement.
    If weights provided, resample them alongside angles.
    """
    rng = np.random.default_rng(seed)

    theta = np.asarray(theta_rad, dtype=float)
    if weights is None:
        m = np.isfinite(theta)
        theta = theta[m]
        w = None
    else:
        w = np.asarray(weights, dtype=float)
        m = np.isfinite(theta) & np.isfinite(w) & (w > 0)
        theta, w = theta[m], w[m]

    n = len(theta)
    if n < 3:
        return (np.nan, np.nan)

    vals = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        vals[b] = anisotropy_A_cos2(theta[idx], None if w is None else w[idx])

    lo, hi = np.quantile(vals, ci)
    return float(lo), float(hi)


def anisotropy_by_player(
    step_df: pd.DataFrame,
    player_col: str = "player_name",
    heading_col: str = "heading_rad",      # or "heading_combined_rad"
    weight_col: str = "step_length_m",
    B: int = 500,
    ci: tuple[float, float] = (0.025, 0.975),
    seed: int = 1,
) -> pd.DataFrame:
    """
    Per-player anisotropy with CIs:
      A_unw = <cos(2θ)>, A_w = weighted by step_length_m
    Returns a tidy DataFrame.
    """
    rows = []
    for p, g in step_df.groupby(player_col, sort=False):
        th = g[heading_col].to_numpy(dtype=float)

        A = anisotropy_A_cos2(th, None)
        Alo, Ahi = bootstrap_ci_A_cos2(th, None, B=B, ci=ci, seed=seed)

        w = g[weight_col].to_numpy(dtype=float) if weight_col in g.columns else None
        Aw = anisotropy_A_cos2(th, w)
        Awlo, Awhi = bootstrap_ci_A_cos2(th, w, B=B, ci=ci, seed=seed + 13)

        rows.append({
            player_col: p,
            "n_runs": int(len(g)),
            "A_unw": A, "A_unw_lo": Alo, "A_unw_hi": Ahi,
            "A_w": Aw, "A_w_lo": Awlo, "A_w_hi": Awhi,
        })

    return pd.DataFrame(rows)

import numpy as np
import pandas as pd


def build_team_centroid_timeseries(
    df_paths: pd.DataFrame,
    t_col: str = "_t",
    x_col: str = "x_m",
    y_col: str = "y_m",
    player_col: str = "player_name",
    phase_col: str = "match_phase",      # keep halves separate automatically
    team_col: str | None = None,         # set to "team" if you have both teams
    min_players: int = 6,                # drop frames with too few players contributing
) -> pd.DataFrame:
    """
    Aggregate player positions -> centroid time series.
    Returns a df with one row per time (and per team/phase if provided):
      columns: [_t, match_phase?, team?, x_m, y_m, n_players]
    """
    if t_col not in df_paths.columns:
        raise ValueError(f"df_paths must contain '{t_col}'")
    for c in (x_col, y_col, player_col):
        if c not in df_paths.columns:
            raise ValueError(f"df_paths must contain '{c}'")

    group_cols = []
    if team_col is not None and team_col in df_paths.columns:
        group_cols.append(team_col)
    if phase_col is not None and phase_col in df_paths.columns:
        group_cols.append(phase_col)
    group_cols.append(t_col)

    df = df_paths.dropna(subset=[t_col, x_col, y_col]).copy()

    # robust: ensure time is datetime-like
    df[t_col] = pd.to_datetime(df[t_col], errors="coerce")
    df = df.dropna(subset=[t_col])

    # centroid per frame (and per team/phase if present)
    g = df.groupby(group_cols, sort=True, observed=False)
    out = g.agg(
        **{
            x_col: (x_col, "mean"),
            y_col: (y_col, "mean"),
            "n_players": (player_col, "nunique"),
        }
    ).reset_index()

    # keep only frames with enough players contributing
    out = out[out["n_players"] >= int(min_players)].copy()

    # sort for later path splitting
    sort_cols = [c for c in group_cols if c != t_col] + [t_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def build_centroid_paths(
    centroid_ts: pd.DataFrame,
    t_col: str = "_t",
    x_col: str = "x_m",
    y_col: str = "y_m",
    phase_col: str = "match_phase",
    team_col: str | None = None,
    max_gap_s: float = 2.0,
    min_len: int = 30,
    entity_col: str = "entity",
    entity_name: str = "centroid",
    path_col: str = "path_uid",
) -> pd.DataFrame:
    """
    Take centroid time series -> continuous paths (split on time gaps and phase/team changes).
    Output resembles df_paths: has entity, path_uid, _t, x_m, y_m, (team/match_phase), n_players.
    """
    if t_col not in centroid_ts.columns:
        raise ValueError(f"centroid_ts must contain '{t_col}'")

    df = centroid_ts.copy()
    df[t_col] = pd.to_datetime(df[t_col], errors="coerce")
    df = df.dropna(subset=[t_col, x_col, y_col]).sort_values(t_col)

    # grouping dimensions to avoid mixing (team and half)
    grp_cols = []
    if team_col is not None and team_col in df.columns:
        grp_cols.append(team_col)
    if phase_col is not None and phase_col in df.columns:
        grp_cols.append(phase_col)

    # assign path ids
    path_ids = np.full(len(df), -1, dtype=int)
    global_pid = 0

    if grp_cols:
        groups = df.groupby(grp_cols, sort=False, observed=False)
        for _, idx in groups.groups.items():
            idx = np.asarray(list(idx))
            sub = df.loc[idx].sort_values(t_col)
            t = sub[t_col].to_numpy()

            # gap in seconds between consecutive centroid frames
            dt = (pd.Series(t).diff().dt.total_seconds().fillna(0.0)).to_numpy()

            local = np.empty(len(sub), dtype=int)
            for i in range(len(sub)):
                if i == 0 or dt[i] > float(max_gap_s):
                    global_pid += 1
                local[i] = global_pid
            path_ids[sub.index.values] = local
    else:
        t = df[t_col].to_numpy()
        dt = (pd.Series(t).diff().dt.total_seconds().fillna(0.0)).to_numpy()
        for i in range(len(df)):
            if i == 0 or dt[i] > float(max_gap_s):
                global_pid += 1
            path_ids[i] = global_pid

    df[path_col] = path_ids
    df[entity_col] = entity_name

    # filter short paths
    counts = df.groupby(path_col, sort=False).size()
    keep = counts[counts >= int(min_len)].index
    df = df[df[path_col].isin(keep)].copy()

    # final tidy sort
    sort_cols = ([entity_col] + grp_cols + [path_col, t_col])
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def centroid_step_lengths(
    centroid_paths: pd.DataFrame,
    t_col: str = "_t",
    x_col: str = "x_m",
    y_col: str = "y_m",
    path_col: str = "path_uid",
) -> pd.DataFrame:
    """
    1-step (frame-to-frame) displacement lengths of the centroid, within each centroid path.
    Returns a df with step_length_m and dt_s.
    """
    df = centroid_paths.sort_values([path_col, t_col]).copy()
    df[t_col] = pd.to_datetime(df[t_col], errors="coerce")
    df = df.dropna(subset=[t_col, x_col, y_col])

    dx = df.groupby(path_col)[x_col].diff()
    dy = df.groupby(path_col)[y_col].diff()
    dt = df.groupby(path_col)[t_col].diff().dt.total_seconds()

    out = df[[path_col]].copy()
    out["dt_s"] = dt
    out["step_length_m"] = np.sqrt(dx * dx + dy * dy)
    out = out.dropna(subset=["dt_s", "step_length_m"])
    out = out[out["dt_s"] > 0].copy()
    return out
# ============================================================
# Quantification helpers (MSD exponent + tail fits)
# ============================================================

def fit_msd_exponent(
    msd_tbl: pd.DataFrame,
    player_col: str = "player_name",
    tau_min: float = 2.0,
    tau_max: float = 20.0,
) -> pd.DataFrame:
    """
    Fit log(MSD) = a + beta log(tau) over a chosen lag window [tau_min, tau_max].

    Returns per player:
      beta, n_points, tau_min, tau_max
    """
    if msd_tbl is None or len(msd_tbl) == 0:
        return pd.DataFrame(columns=[player_col, "beta", "n_points", "tau_min", "tau_max"])

    rows = []
    for p, g in msd_tbl.groupby(player_col, sort=False):
        gg = g[(g["tau_s"] >= float(tau_min)) & (g["tau_s"] <= float(tau_max))].copy()
        gg = gg[np.isfinite(gg["tau_s"]) & (gg["tau_s"] > 0) & np.isfinite(gg["msd"]) & (gg["msd"] > 0)]
        if len(gg) < 3:
            rows.append({player_col: p, "beta": np.nan, "n_points": int(len(gg)), "tau_min": tau_min, "tau_max": tau_max})
            continue

        x = np.log(gg["tau_s"].to_numpy(dtype=float))
        y = np.log(gg["msd"].to_numpy(dtype=float))
        beta, a = np.polyfit(x, y, 1)
        rows.append({player_col: p, "beta": float(beta), "n_points": int(len(gg)), "tau_min": tau_min, "tau_max": tau_max})

    return pd.DataFrame(rows)


def powerlaw_alpha_mle(x: np.ndarray, xmin: float) -> float:
    """
    Continuous power-law MLE for x >= xmin:
      p(x) = (alpha-1) xmin^{alpha-1} x^{-alpha}
      alpha_hat = 1 + n / sum(log(x/xmin))
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= float(xmin)]
    if len(x) < 2:
        return np.nan
    denom = np.sum(np.log(x / float(xmin)))
    if denom <= 0:
        return np.nan
    return float(1.0 + len(x) / denom)


def powerlaw_ks_distance(x: np.ndarray, xmin: float, alpha: float) -> float:
    """
    KS distance between empirical CDF of tail x>=xmin and power-law model CDF.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= float(xmin)]
    n = len(x)
    if n == 0 or not np.isfinite(alpha):
        return np.nan

    xs = np.sort(x)
    # empirical CDF
    F_emp = np.arange(1, n + 1) / n

    # model CDF: F(x) = 1 - (x/xmin)^{-(alpha-1)}
    F_mod = 1.0 - (xs / float(xmin)) ** (-(alpha - 1.0))
    return float(np.max(np.abs(F_emp - F_mod)))


def fit_powerlaw_tail(
    x: np.ndarray,
    xmin: float | None = None,
    min_tail: int = 50,
    n_xmin_grid: int = 30,
) -> dict:
    """
    Simple, self-contained tail fit:
      - if xmin is None, pick xmin that minimises KS distance over a grid of candidate xmins
      - fit alpha by MLE on x>=xmin
      - report KS, n_tail

    This is intentionally lightweight (no external powerlaw package).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) < max(10, int(min_tail)):
        return {"xmin": np.nan, "alpha": np.nan, "ks": np.nan, "n_tail": int(len(x))}

    xs = np.sort(x)

    if xmin is None:
        # candidate xmins from quantiles of the positive data
        qs = np.linspace(0.05, 0.70, int(n_xmin_grid))
        cand = np.unique(np.quantile(xs, qs))
        best = None
        for xm in cand:
            tail = xs[xs >= xm]
            if len(tail) < int(min_tail):
                continue
            a = powerlaw_alpha_mle(tail, xm)
            ks = powerlaw_ks_distance(tail, xm, a)
            if best is None or (np.isfinite(ks) and ks < best["ks"]):
                best = {"xmin": float(xm), "alpha": float(a), "ks": float(ks), "n_tail": int(len(tail))}
        if best is None:
            # fallback: use a high quantile
            xm = float(np.quantile(xs, 0.50))
            tail = xs[xs >= xm]
            a = powerlaw_alpha_mle(tail, xm)
            ks = powerlaw_ks_distance(tail, xm, a)
            return {"xmin": float(xm), "alpha": float(a), "ks": float(ks), "n_tail": int(len(tail))}
        return best

    # fixed xmin
    xm = float(xmin)
    tail = xs[xs >= xm]
    if len(tail) < int(min_tail):
        return {"xmin": float(xm), "alpha": np.nan, "ks": np.nan, "n_tail": int(len(tail))}
    a = powerlaw_alpha_mle(tail, xm)
    ks = powerlaw_ks_distance(tail, xm, a)
    return {"xmin": float(xm), "alpha": float(a), "ks": float(ks), "n_tail": int(len(tail))}

# ============================================================
# Paper-facing transport tables, standalone v2
# ============================================================
# Main entry point:
#
#   transport = build_transport_tables_from_active_v2(df_active, ...)
#
# Outputs:
#   trajectory_long
#   msd_long
#   runs_long
#   ccdf_long
#   msd_decomp_long
#   figure_index
#
# This function DOES NOT call analyse_active_match().
# It is designed for clean one-match paper figures and later
# multi-match aggregation.
# ============================================================


def _btv2_pick_col(df, candidates, required=True, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find {label}. Tried {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _btv2_ccdf_nonzero(x):
    """
    Empirical CCDF with final non-zero point = 1/n.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) == 0:
        return np.array([]), np.array([])

    xs = np.sort(x)
    n = len(xs)
    surv = np.arange(n, 0, -1, dtype=float) / float(n)
    return xs, surv


def _btv2_seconds_between(t2, t1):
    return (t2 - t1) / np.timedelta64(1, "s")


def _btv2_standardise_active(
    df_active,
    *,
    player_col="player_name",
    time_col="timestamp",
    phase_col="match_phase",
    half_col="half",
    team_col="team",
    source_col=None,
    match_col="match_id",
    default_match_id="single_match",
    x_col="x_m",
    y_col="y_m",
    phases=("1H", "2H"),
):
    """
    Standardise df_active into one clean active tracking table.

    Required output columns:
      match_id, match_phase, team, source_key, player_name, _t, x_m, y_m
    """
    df = df_active.copy()

    # Time column fallback.
    if time_col not in df.columns:
        if "_t" in df.columns:
            time_col = "_t"
        elif "time" in df.columns:
            time_col = "time"
        else:
            raise KeyError(
                f"Could not find time_col={time_col}, '_t', or 'time'. "
                f"Available columns: {list(df.columns)}"
            )

    # Match id.
    if match_col in df.columns:
        df["match_id"] = df[match_col].astype(str)
    elif "match_id" in df.columns:
        df["match_id"] = df["match_id"].astype(str)
    else:
        df["match_id"] = str(default_match_id)

    # Source metadata.
    if source_col is None:
        if "source_key" in df.columns:
            source_col = "source_key"
        elif "source" in df.columns:
            source_col = "source"
        else:
            source_col = None

    if source_col is not None and source_col in df.columns:
        df["source_key"] = df[source_col].astype(str)
    else:
        df["source_key"] = "unknown_source"

    # Phase.
    if phase_col in df.columns:
        df["match_phase"] = df[phase_col].astype(str)
    elif half_col in df.columns:
        df["match_phase"] = df[half_col].astype(str)
    else:
        df["match_phase"] = "unknown_phase"

    # Team.
    if team_col in df.columns:
        df["team"] = df[team_col].astype(str)
    else:
        df["team"] = "unknown_team"

    if player_col not in df.columns:
        raise KeyError(f"df_active must contain player_col={player_col}")

    df["player_name"] = df[player_col].astype(str)
    df["_t"] = pd.to_datetime(df[time_col], errors="coerce")
    df["x_m"] = pd.to_numeric(df[x_col], errors="coerce")
    df["y_m"] = pd.to_numeric(df[y_col], errors="coerce")

    df = df.dropna(subset=["_t", "x_m", "y_m", "player_name", "team"]).copy()

    if phases is not None and "match_phase" in df.columns:
        df = df[df["match_phase"].isin([str(p) for p in phases])].copy()

    df = df.sort_values(
        ["match_id", "match_phase", "team", "source_key", "player_name", "_t"]
    ).reset_index(drop=True)

    return df


def _btv2_build_player_paths(
    df_active_std,
    *,
    max_gap_s=2.0,
    min_len=30,
):
    """
    Split active player data into continuous paths.
    Team/source/phase safe.
    """
    df = df_active_std.copy()

    group_cols = ["match_id", "match_phase", "team", "source_key", "player_name"]
    df = df.sort_values(group_cols + ["_t"]).reset_index(drop=True)

    pieces = []

    for keys, g in df.groupby(group_cols, sort=False, dropna=False):
        g = g.copy().sort_values("_t").reset_index(drop=True)

        dt = g["_t"].diff().dt.total_seconds()
        new_path = dt.isna() | dt.gt(float(max_gap_s))
        g["path_index"] = new_path.cumsum().astype(int) - 1

        match_id, phase, team, source, player = keys
        base = (
            "player_path"
            + "__match="
            + str(match_id)
            + "__phase="
            + str(phase)
            + "__team="
            + str(team)
            + "__source="
            + str(source)
            + "__player="
            + str(player)
        )

        g["path_uid"] = base + "__p=" + g["path_index"].astype(str)
        pieces.append(g)

    if len(pieces) == 0:
        return pd.DataFrame(columns=list(df.columns) + ["path_uid", "path_index"])

    df_paths = pd.concat(pieces, ignore_index=True)

    counts = df_paths["path_uid"].value_counts()
    keep = counts[counts >= int(min_len)].index
    df_paths = df_paths[df_paths["path_uid"].isin(keep)].copy()

    df_paths = df_paths.sort_values(
        ["match_id", "match_phase", "team", "source_key", "player_name", "path_uid", "_t"]
    ).reset_index(drop=True)

    return df_paths


def _btv2_build_centroid_timeseries(
    df_paths,
    *,
    min_players_centroid=7,
):
    """
    Team centroid per match/phase/team/source/time.
    """
    group_cols = ["match_id", "match_phase", "team", "source_key", "_t"]

    centroid_ts = (
        df_paths
        .groupby(group_cols, as_index=False, sort=True)
        .agg(
            x_m=("x_m", "mean"),
            y_m=("y_m", "mean"),
            n_players=("player_name", "nunique"),
        )
    )

    centroid_ts = centroid_ts[
        centroid_ts["n_players"] >= int(min_players_centroid)
    ].copy()

    centroid_ts = centroid_ts.sort_values(group_cols).reset_index(drop=True)

    return centroid_ts


def _btv2_build_centroid_paths(
    centroid_ts,
    *,
    max_gap_s=2.0,
    min_len=30,
):
    """
    Split centroid time series into continuous paths.
    """
    df = centroid_ts.copy()
    group_cols = ["match_id", "match_phase", "team", "source_key"]

    df = df.sort_values(group_cols + ["_t"]).reset_index(drop=True)

    pieces = []

    for keys, g in df.groupby(group_cols, sort=False, dropna=False):
        g = g.copy().sort_values("_t").reset_index(drop=True)

        dt = g["_t"].diff().dt.total_seconds()
        new_path = dt.isna() | dt.gt(float(max_gap_s))
        g["path_index"] = new_path.cumsum().astype(int) - 1

        match_id, phase, team, source = keys
        base = (
            "centroid_path"
            + "__match="
            + str(match_id)
            + "__phase="
            + str(phase)
            + "__team="
            + str(team)
            + "__source="
            + str(source)
        )

        g["path_uid"] = base + "__p=" + g["path_index"].astype(str)
        g["entity"] = "centroid:" + str(team)
        pieces.append(g)

    if len(pieces) == 0:
        return pd.DataFrame(columns=list(df.columns) + ["path_uid", "entity"])

    df_centroid = pd.concat(pieces, ignore_index=True)

    counts = df_centroid["path_uid"].value_counts()
    keep = counts[counts >= int(min_len)].index
    df_centroid = df_centroid[df_centroid["path_uid"].isin(keep)].copy()

    df_centroid = df_centroid.sort_values(
        ["match_id", "match_phase", "team", "source_key", "path_uid", "_t"]
    ).reset_index(drop=True)

    return df_centroid


def _btv2_attach_centroid_to_players(df_paths, centroid_ts):
    """
    Attach same-team centroid position to each player frame.
    """
    join_cols = ["match_id", "match_phase", "team", "source_key", "_t"]

    cent = centroid_ts[
        join_cols + ["x_m", "y_m", "n_players"]
    ].copy()

    cent = cent.rename(
        columns={
            "x_m": "x_centroid",
            "y_m": "y_centroid",
        }
    )

    out = df_paths.merge(
        cent,
        on=join_cols,
        how="left",
        validate="many_to_one",
    )

    out = out.dropna(subset=["x_centroid", "y_centroid"]).copy()
    out["x_rel"] = out["x_m"] - out["x_centroid"]
    out["y_rel"] = out["y_m"] - out["y_centroid"]

    return out


def _btv2_make_track_uids(df):
    df = df.copy()

    df["track_entity_uid"] = (
        "match="
        + df["match_id"].astype(str)
        + "__phase="
        + df["match_phase"].astype(str)
        + "__team="
        + df["team"].astype(str)
        + "__source="
        + df["source_key"].astype(str)
        + "__type="
        + df["track_type"].astype(str)
        + "__player="
        + df["player_name"].astype(str)
    )

    df["track_uid"] = (
        df["track_entity_uid"].astype(str)
        + "__path="
        + df["path_uid"].astype(str)
    )

    return df


def _btv2_build_trajectory_long(df_paths, df_centroid, df_paths_rel):
    """
    Long trajectory table with:
      player_abs, player_rel, centroid
    """
    parts = []

    # Player absolute.
    pa = df_paths[
        [
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "player_name",
            "path_uid",
            "_t",
            "x_m",
            "y_m",
        ]
    ].copy()
    pa["track_type"] = "player_abs"
    parts.append(pa)

    # Player relative.
    pr = df_paths_rel[
        [
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "player_name",
            "path_uid",
            "_t",
            "x_rel",
            "y_rel",
        ]
    ].copy()
    pr = pr.rename(columns={"x_rel": "x_m", "y_rel": "y_m"})
    pr["track_type"] = "player_rel"
    parts.append(pr)

    # Centroid.
    ct = df_centroid[
        [
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "path_uid",
            "_t",
            "x_m",
            "y_m",
        ]
    ].copy()
    ct["player_name"] = "__centroid__"
    ct["track_type"] = "centroid"
    parts.append(ct)

    trajectory_long = pd.concat(parts, ignore_index=True)
    trajectory_long = trajectory_long.rename(columns={"_t": "t"})

    trajectory_long = _btv2_make_track_uids(trajectory_long)

    trajectory_long = trajectory_long[
        [
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "track_type",
            "track_entity_uid",
            "track_uid",
            "player_name",
            "path_uid",
            "t",
            "x_m",
            "y_m",
        ]
    ].sort_values(
        [
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "track_type",
            "player_name",
            "path_uid",
            "t",
        ]
    ).reset_index(drop=True)

    return trajectory_long


def _btv2_compute_msd_long(trajectory_long, *, max_k=120):
    """
    Compute MSD within each path, then aggregate across paths for each entity.
    """
    rows = []

    meta_cols = [
        "match_id",
        "match_phase",
        "team",
        "source_key",
        "track_type",
        "track_entity_uid",
        "player_name",
    ]

    for (entity_uid, track_uid), g in trajectory_long.groupby(
        ["track_entity_uid", "track_uid"], sort=False
    ):
        g = g.sort_values("t")
        n = len(g)

        if n < 2:
            continue

        X = g[["x_m", "y_m"]].to_numpy(dtype=float)
        tt = g["t"].to_numpy(dtype="datetime64[ns]")

        meta = g.iloc[0][meta_cols].to_dict()

        kmax = min(int(max_k), n - 1)

        for k in range(1, kmax + 1):
            dX = X[k:] - X[:-k]
            sq = np.sum(dX * dX, axis=1)
            dt = _btv2_seconds_between(tt[k:], tt[:-k])

            good = np.isfinite(sq) & np.isfinite(dt) & (dt > 0)

            if good.sum() == 0:
                continue

            row = dict(meta)
            row.update(
                {
                    "k": int(k),
                    "tau_s": float(np.nanmean(dt[good])),
                    "msd_m2": float(np.nanmean(sq[good])),
                    "n_pairs": int(good.sum()),
                }
            )
            rows.append(row)

    raw = pd.DataFrame(rows)

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "match_phase",
                "team",
                "source_key",
                "track_type",
                "track_entity_uid",
                "player_name",
                "k",
                "tau_s",
                "msd_m2",
                "n_pairs",
            ]
        )

    group_cols = [
        "match_id",
        "match_phase",
        "team",
        "source_key",
        "track_type",
        "track_entity_uid",
        "player_name",
        "k",
    ]

    raw["tau_w"] = raw["tau_s"] * raw["n_pairs"]
    raw["msd_w"] = raw["msd_m2"] * raw["n_pairs"]

    out = (
        raw.groupby(group_cols, as_index=False)
        .agg(
            tau_w=("tau_w", "sum"),
            msd_w=("msd_w", "sum"),
            n_pairs=("n_pairs", "sum"),
        )
    )

    out["tau_s"] = out["tau_w"] / out["n_pairs"]
    out["msd_m2"] = out["msd_w"] / out["n_pairs"]

    out = out[group_cols + ["tau_s", "msd_m2", "n_pairs"]].sort_values(
        [
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "track_type",
            "player_name",
            "k",
        ]
    ).reset_index(drop=True)

    return out


def _btv2_compute_runs_long(
    trajectory_long,
    *,
    theta_deg=35.0,
    min_run_samples=2,
    min_disp_m=1e-6,
):
    """
    Turning-angle run segmentation for all track types.
    Uses existing compute_turn_runs().
    """
    if trajectory_long.empty:
        return pd.DataFrame()

    tmp = trajectory_long.copy()
    tmp = tmp.rename(columns={"t": "_t"})

    runs = compute_turn_runs(
        tmp,
        player_col="track_entity_uid",
        path_col="track_uid",
        x_col="x_m",
        y_col="y_m",
        t_col="_t",
        theta_deg=theta_deg,
        min_run_samples=min_run_samples,
        min_disp_m=min_disp_m,
    )

    if runs.empty:
        return pd.DataFrame()

    meta_cols = [
        "match_id",
        "match_phase",
        "team",
        "source_key",
        "track_type",
        "track_entity_uid",
        "track_uid",
        "player_name",
        "path_uid",
    ]

    meta = trajectory_long[meta_cols].drop_duplicates(
        subset=["track_entity_uid", "track_uid"]
    )

    runs_long = runs.merge(
        meta,
        on=["track_entity_uid", "track_uid"],
        how="left",
        validate="many_to_one",
    )

    runs_long["run_uid"] = (
        runs_long["track_uid"].astype(str)
        + "__run="
        + runs_long["run_id"].astype(str)
    )

    keep_cols = [
        "match_id",
        "match_phase",
        "team",
        "source_key",
        "track_type",
        "track_entity_uid",
        "track_uid",
        "run_uid",
        "player_name",
        "path_uid",
        "run_id",
        "t_start",
        "t_end",
        "duration_s",
        "run_length_m",
        "v_mean_mps",
    ]

    runs_long = runs_long[keep_cols].copy()

    runs_long["t_start"] = pd.to_datetime(runs_long["t_start"], errors="coerce")
    runs_long["t_end"] = pd.to_datetime(runs_long["t_end"], errors="coerce")
    runs_long["duration_s"] = pd.to_numeric(runs_long["duration_s"], errors="coerce")
    runs_long["run_length_m"] = pd.to_numeric(runs_long["run_length_m"], errors="coerce")
    runs_long["v_mean_mps"] = pd.to_numeric(runs_long["v_mean_mps"], errors="coerce")

    runs_long = runs_long.dropna(subset=["duration_s", "run_length_m"]).copy()

    runs_long = runs_long.sort_values(
        [
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "track_type",
            "player_name",
            "path_uid",
            "run_id",
        ]
    ).reset_index(drop=True)

    return runs_long


def _btv2_build_ccdf_long(runs_long):
    """
    CCDFs for run_length_m, duration_s, and v_mean_mps per track entity.
    """
    if runs_long is None or runs_long.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "match_phase",
                "team",
                "source_key",
                "track_type",
                "track_entity_uid",
                "player_name",
                "variable",
                "threshold",
                "ccdf",
                "n",
            ]
        )

    rows = []

    group_cols = [
        "match_id",
        "match_phase",
        "team",
        "source_key",
        "track_type",
        "track_entity_uid",
        "player_name",
    ]

    value_cols = ["run_length_m", "duration_s", "v_mean_mps"]

    for keys, g in runs_long.groupby(group_cols, sort=False, dropna=False):
        meta = dict(zip(group_cols, keys))

        for col in value_cols:
            xs, surv = _btv2_ccdf_nonzero(g[col].to_numpy(dtype=float))

            if len(xs) == 0:
                continue

            for x, y in zip(xs, surv):
                row = dict(meta)
                row.update(
                    {
                        "variable": col,
                        "threshold": float(x),
                        "ccdf": float(y),
                        "n": int(len(xs)),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


def _btv2_compute_msd_decomp_long(df_paths_rel, *, max_k=120):
    """
    MSD decomposition:
      Δplayer = Δcentroid + Δrelative

    Components:
      total_player, centroid, relative, cross
    where:
      total_player = centroid + relative + cross
      cross = 2 <Δcentroid · Δrelative>
    """
    d = df_paths_rel.copy()
    d = d.dropna(
        subset=[
            "x_m",
            "y_m",
            "x_centroid",
            "y_centroid",
            "x_rel",
            "y_rel",
            "_t",
        ]
    ).copy()

    rows = []

    group_cols = ["match_id", "match_phase", "team", "source_key", "player_name", "path_uid"]

    for keys, g in d.groupby(group_cols, sort=False, dropna=False):
        match_id, phase, team, source, player, path_uid = keys

        g = g.sort_values("_t")
        n = len(g)

        if n < 2:
            continue

        xp = g[["x_m", "y_m"]].to_numpy(dtype=float)
        xc = g[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        xr = g[["x_rel", "y_rel"]].to_numpy(dtype=float)
        tt = g["_t"].to_numpy(dtype="datetime64[ns]")

        kmax = min(int(max_k), n - 1)

        track_entity_uid = (
            "match="
            + str(match_id)
            + "__phase="
            + str(phase)
            + "__team="
            + str(team)
            + "__source="
            + str(source)
            + "__type=player_decomp"
            + "__player="
            + str(player)
        )

        for k in range(1, kmax + 1):
            dp = xp[k:] - xp[:-k]
            dc = xc[k:] - xc[:-k]
            dr = xr[k:] - xr[:-k]

            total = np.sum(dp * dp, axis=1)
            centroid = np.sum(dc * dc, axis=1)
            relative = np.sum(dr * dr, axis=1)
            cross = 2.0 * np.sum(dc * dr, axis=1)

            dt = _btv2_seconds_between(tt[k:], tt[:-k])
            good = np.isfinite(total) & np.isfinite(dt) & (dt > 0)

            if good.sum() == 0:
                continue

            tau_s = float(np.nanmean(dt[good]))

            vals = {
                "total_player": float(np.nanmean(total[good])),
                "centroid": float(np.nanmean(centroid[good])),
                "relative": float(np.nanmean(relative[good])),
                "cross": float(np.nanmean(cross[good])),
            }

            for component, value in vals.items():
                rows.append(
                    {
                        "match_id": str(match_id),
                        "match_phase": str(phase),
                        "team": str(team),
                        "source_key": str(source),
                        "track_entity_uid": track_entity_uid,
                        "player_name": str(player),
                        "path_uid": str(path_uid),
                        "k": int(k),
                        "tau_s": tau_s,
                        "component": component,
                        "value_m2": float(value),
                        "n_pairs": int(good.sum()),
                    }
                )

    raw = pd.DataFrame(rows)

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "match_phase",
                "team",
                "source_key",
                "track_entity_uid",
                "player_name",
                "k",
                "tau_s",
                "component",
                "value_m2",
                "fraction_of_total",
                "n_pairs",
            ]
        )

    group_cols2 = [
        "match_id",
        "match_phase",
        "team",
        "source_key",
        "track_entity_uid",
        "player_name",
        "k",
        "component",
    ]

    raw["value_w"] = raw["value_m2"] * raw["n_pairs"]
    raw["tau_w"] = raw["tau_s"] * raw["n_pairs"]

    out = (
        raw.groupby(group_cols2, as_index=False)
        .agg(
            value_w=("value_w", "sum"),
            tau_w=("tau_w", "sum"),
            n_pairs=("n_pairs", "sum"),
        )
    )

    out["value_m2"] = out["value_w"] / out["n_pairs"]
    out["tau_s"] = out["tau_w"] / out["n_pairs"]

    totals = (
        out[out["component"] == "total_player"][
            [
                "match_id",
                "match_phase",
                "team",
                "source_key",
                "track_entity_uid",
                "player_name",
                "k",
                "value_m2",
            ]
        ]
        .rename(columns={"value_m2": "total_player_m2"})
    )

    out = out.merge(
        totals,
        on=[
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "track_entity_uid",
            "player_name",
            "k",
        ],
        how="left",
    )

    out["fraction_of_total"] = out["value_m2"] / out["total_player_m2"]

    out = out[
        [
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "track_entity_uid",
            "player_name",
            "k",
            "tau_s",
            "component",
            "value_m2",
            "fraction_of_total",
            "n_pairs",
        ]
    ].sort_values(
        [
            "match_id",
            "match_phase",
            "team",
            "source_key",
            "player_name",
            "k",
            "component",
        ]
    ).reset_index(drop=True)

    return out


def _btv2_build_figure_index(trajectory_long, runs_long):
    """
    Candidate player/team/phase table for choosing example panels.
    """
    traj_idx = (
        trajectory_long[trajectory_long["track_type"] == "player_abs"]
        .groupby(
            ["match_id", "match_phase", "team", "source_key", "player_name"],
            as_index=False,
        )
        .agg(
            n_frames=("t", "size"),
            n_paths=("path_uid", "nunique"),
            t_start=("t", "min"),
            t_end=("t", "max"),
        )
    )

    traj_idx["duration_available_s"] = (
        traj_idx["t_end"] - traj_idx["t_start"]
    ).dt.total_seconds()

    if runs_long is None or runs_long.empty:
        traj_idx["n_runs_abs"] = np.nan
        traj_idx["median_run_length_m"] = np.nan
        traj_idx["max_run_length_m"] = np.nan
        traj_idx["median_duration_s"] = np.nan
        traj_idx["max_duration_s"] = np.nan
        return traj_idx

    run_idx = (
        runs_long[runs_long["track_type"] == "player_abs"]
        .groupby(
            ["match_id", "match_phase", "team", "source_key", "player_name"],
            as_index=False,
        )
        .agg(
            n_runs_abs=("run_id", "size"),
            median_run_length_m=("run_length_m", "median"),
            max_run_length_m=("run_length_m", "max"),
            median_duration_s=("duration_s", "median"),
            max_duration_s=("duration_s", "max"),
        )
    )

    out = traj_idx.merge(
        run_idx,
        on=["match_id", "match_phase", "team", "source_key", "player_name"],
        how="left",
    )

    out = out.sort_values(
        ["n_frames", "n_runs_abs", "max_run_length_m"],
        ascending=False,
    ).reset_index(drop=True)

    return out


def build_transport_tables_from_active_v2(
    df_active,
    *,
    theta_deg=35.0,
    max_k=120,
    max_gap_s=2.0,
    min_len=30,
    min_players_centroid=7,
    player_col="player_name",
    time_col="timestamp",
    phase_col="match_phase",
    half_col="half",
    team_col="team",
    source_col=None,
    match_col="match_id",
    default_match_id="single_match",
    x_col="x_m",
    y_col="y_m",
    phases=("1H", "2H"),
    min_run_samples=2,
    min_disp_m=1e-6,
):
    """
    Standalone paper-facing transport pipeline.

    Parameters
    ----------
    df_active : pd.DataFrame
        Active/on-pitch tracking table. Can contain one or two teams.

    Returns
    -------
    dict
        trajectory_long
        msd_long
        runs_long
        ccdf_long
        msd_decomp_long
        figure_index

        plus debug/base tables:
        df_active_std
        df_paths
        centroid_ts
        df_centroid
        df_paths_rel
    """

    df_active_std = _btv2_standardise_active(
        df_active,
        player_col=player_col,
        time_col=time_col,
        phase_col=phase_col,
        half_col=half_col,
        team_col=team_col,
        source_col=source_col,
        match_col=match_col,
        default_match_id=default_match_id,
        x_col=x_col,
        y_col=y_col,
        phases=phases,
    )

    df_paths = _btv2_build_player_paths(
        df_active_std,
        max_gap_s=max_gap_s,
        min_len=min_len,
    )

    centroid_ts = _btv2_build_centroid_timeseries(
        df_paths,
        min_players_centroid=min_players_centroid,
    )

    df_centroid = _btv2_build_centroid_paths(
        centroid_ts,
        max_gap_s=max_gap_s,
        min_len=min_len,
    )

    df_paths_rel = _btv2_attach_centroid_to_players(
        df_paths,
        centroid_ts,
    )

    trajectory_long = _btv2_build_trajectory_long(
        df_paths=df_paths,
        df_centroid=df_centroid,
        df_paths_rel=df_paths_rel,
    )

    msd_long = _btv2_compute_msd_long(
        trajectory_long,
        max_k=max_k,
    )

    runs_long = _btv2_compute_runs_long(
        trajectory_long,
        theta_deg=theta_deg,
        min_run_samples=min_run_samples,
        min_disp_m=min_disp_m,
    )

    ccdf_long = _btv2_build_ccdf_long(
        runs_long,
    )

    msd_decomp_long = _btv2_compute_msd_decomp_long(
        df_paths_rel,
        max_k=max_k,
    )

    figure_index = _btv2_build_figure_index(
        trajectory_long,
        runs_long,
    )

    return {
        "trajectory_long": trajectory_long,
        "msd_long": msd_long,
        "runs_long": runs_long,
        "ccdf_long": ccdf_long,
        "msd_decomp_long": msd_decomp_long,
        "figure_index": figure_index,

        # debug/base tables
        "df_active_std": df_active_std,
        "df_paths": df_paths,
        "centroid_ts": centroid_ts,
        "df_centroid": df_centroid,
        "df_paths_rel": df_paths_rel,
    }

    # ============================================================
# Heading / axial-anisotropy helpers for transport tables
# ============================================================

import numpy as np
import pandas as pd


def _wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def _nearest_time_index(times, target):
    """
    Return nearest index in sorted numpy datetime64[ns] array.
    """
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


def _add_run_endpoints_from_trajectory(runs, trajectory):
    """
    Add x_start_m, y_start_m, x_end_m, y_end_m to runs_long
    by matching t_start/t_end and track_uid against trajectory_long.
    """
    r = runs.copy().reset_index(drop=True)
    t = trajectory.copy()

    if r.empty or t.empty:
        return r

    required_runs = {"track_uid", "t_start", "t_end"}
    required_traj = {"track_uid", "t", "x_m", "y_m"}

    missing_runs = required_runs - set(r.columns)
    missing_traj = required_traj - set(t.columns)

    if missing_runs:
        raise KeyError(f"runs table missing required columns: {missing_runs}")
    if missing_traj:
        raise KeyError(f"trajectory table missing required columns: {missing_traj}")

    r["t_start"] = pd.to_datetime(r["t_start"])
    r["t_end"] = pd.to_datetime(r["t_end"])
    t["t"] = pd.to_datetime(t["t"])

    endpoint_cols = ["x_start_m", "y_start_m", "x_end_m", "y_end_m"]
    for c in endpoint_cols:
        r[c] = np.nan

    traj_by_uid = {
        uid: g.sort_values("t").reset_index(drop=True)
        for uid, g in t.groupby("track_uid", sort=False)
    }

    for uid, rg in r.groupby("track_uid", sort=False):
        if uid not in traj_by_uid:
            continue

        g = traj_by_uid[uid]
        times = g["t"].to_numpy(dtype="datetime64[ns]")
        xs = g["x_m"].to_numpy(dtype=float)
        ys = g["y_m"].to_numpy(dtype=float)

        for idx, rr in rg.iterrows():
            t0 = pd.Timestamp(rr["t_start"]).to_datetime64()
            t1 = pd.Timestamp(rr["t_end"]).to_datetime64()

            i0 = _nearest_time_index(times, t0)
            i1 = _nearest_time_index(times, t1)

            r.loc[idx, "x_start_m"] = xs[i0]
            r.loc[idx, "y_start_m"] = ys[i0]
            r.loc[idx, "x_end_m"] = xs[i1]
            r.loc[idx, "y_end_m"] = ys[i1]

    return r


def prepare_heading_runs_from_transport(
    transport,
    *,
    match_id=None,
    team=None,
    source_key=None,
    track_type="player_abs",
    combine_halves=True,
    min_run_length_m=0.0,
    phase_col="match_phase",
):
    """
    Build a clean run-level heading table from transport outputs.

    Parameters
    ----------
    transport : dict
        Output from build_transport_tables_from_active(...).

    Returns
    -------
    heading_runs : pd.DataFrame
        runs_long subset with:
          heading_rad
          heading_combined_rad
          cos2theta
          x_start_m, y_start_m, x_end_m, y_end_m
    """

    runs_long = transport["runs_long"].copy()
    trajectory_long = transport["trajectory_long"].copy()

    r = runs_long[
        (runs_long["track_type"] == track_type) &
        (runs_long["player_name"] != "__centroid__")
    ].copy()

    t = trajectory_long[
        (trajectory_long["track_type"] == track_type) &
        (trajectory_long["player_name"] != "__centroid__")
    ].copy()

    if match_id is not None:
        r = r[r["match_id"] == match_id].copy()
        t = t[t["match_id"] == match_id].copy()

    if team is not None:
        r = r[r["team"] == team].copy()
        t = t[t["team"] == team].copy()

    if source_key is not None and "source_key" in r.columns:
        r = r[r["source_key"].astype(str) == str(source_key)].copy()
        t = t[t["source_key"].astype(str) == str(source_key)].copy()

    if min_run_length_m is not None:
        r = r[pd.to_numeric(r["run_length_m"], errors="coerce") >= min_run_length_m].copy()

    # Reconstruct start/end coordinates if not already present.
    needed = {"x_start_m", "y_start_m", "x_end_m", "y_end_m"}
    if not needed.issubset(r.columns):
        r = _add_run_endpoints_from_trajectory(r, t)

    r = r.dropna(subset=["x_start_m", "y_start_m", "x_end_m", "y_end_m"]).copy()

    r["dx_m"] = r["x_end_m"] - r["x_start_m"]
    r["dy_m"] = r["y_end_m"] - r["y_start_m"]

    r["heading_rad"] = _wrap_to_pi(np.arctan2(r["dy_m"], r["dx_m"]))

    # Directional half-combined heading for visual histogram.
    # Axial anisotropy cos(2θ) is invariant to θ -> θ + π.
    r["heading_combined_rad"] = r["heading_rad"]

    if combine_halves and phase_col in r.columns:
        is_second_half = r[phase_col].astype(str).str.upper().isin(
            ["2H", "H2", "SECOND", "SECOND_HALF"]
        )
        r.loc[is_second_half, "heading_combined_rad"] = _wrap_to_pi(
            r.loc[is_second_half, "heading_combined_rad"] + np.pi
        )

    r["cos2theta"] = np.cos(2.0 * r["heading_rad"])

    return r.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def _bootstrap_mean_ci(values, weights=None, B=500, seed=1):
    rng = np.random.default_rng(seed)

    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        mask &= np.isfinite(weights) & (weights >= 0)

    values = values[mask]

    if weights is not None:
        weights = weights[mask]

    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    def stat(v, w=None):
        if w is None:
            return float(np.mean(v))
        if np.sum(w) <= 0:
            return np.nan
        return float(np.average(v, weights=w))

    est = stat(values, weights)

    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        if weights is None:
            boots[b] = stat(values[idx], None)
        else:
            boots[b] = stat(values[idx], weights[idx])

    lo, hi = np.nanpercentile(boots, [2.5, 97.5])

    return est, lo, hi


def anisotropy_by_player_transport(
    heading_runs,
    *,
    player_cols=None,
    weight_col="run_length_m",
    B=500,
    seed=1,
):
    """
    Compute axial run-heading anisotropy per player:

        A = <cos(2 theta)>

    Returns both unweighted and run-length-weighted estimates.
    """

    d = heading_runs.copy()

    if player_cols is None:
        player_cols = ["player_name"]
        if "team" in d.columns:
            player_cols = ["team"] + player_cols
        if "source_key" in d.columns:
            player_cols = ["source_key"] + player_cols

    rows = []

    for keys, g in d.groupby(player_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple): # th
            keys = (keys,)

        row = dict(zip(player_cols, keys))

        values = g["cos2theta"].to_numpy(dtype=float)
        weights = g[weight_col].to_numpy(dtype=float) if weight_col in g.columns else None

        A_unw, A_unw_lo, A_unw_hi = _bootstrap_mean_ci(
            values,
            weights=None,
            B=B,
            seed=seed,
        )

        A_w, A_w_lo, A_w_hi = _bootstrap_mean_ci(
            values,
            weights=weights,
            B=B,
            seed=seed + 19,
        )

        row.update({
            "n_runs": len(g),
            "A_unw": A_unw,
            "A_unw_lo": A_unw_lo,
            "A_unw_hi": A_unw_hi,
            "A_w": A_w,
            "A_w_lo": A_w_lo,
            "A_w_hi": A_w_hi,
        })

        rows.append(row)

    return pd.DataFrame(rows)