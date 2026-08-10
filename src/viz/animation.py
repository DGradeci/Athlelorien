"""
Animation utilities for player movement on the pitch.

Features:
- Works with df that has either 'timestamp' or 'time'
- Optional time windowing: start_time / end_time
- Optional frame cap: max_frames (prevents embed_limit blowups)
- Optional trails: show_trails with trail_s seconds of history
- Trails can be restricted to active players only (trail_only_active=True)
- Match title support: full date/time + home/away + score via matches + game_number
"""

from __future__ import annotations

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd

from .pitch_draw import draw_pitch


# -----------------------------
# Helpers
# -----------------------------
def _choose_time(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce")
    if "time" in df.columns:
        return pd.to_datetime(df["time"], errors="coerce")
    raise ValueError("df must have 'timestamp' or 'time'.")


def _apply_time_window(df: pd.DataFrame, start_time=None, end_time=None) -> pd.DataFrame:
    """Filter df by optional start/end times using df['_ts']."""
    if start_time is not None:
        t0 = pd.to_datetime(start_time, errors="coerce")
        if pd.isna(t0):
            raise ValueError(f"Could not parse start_time={start_time}")
        df = df[df["_ts"] >= t0]

    if end_time is not None:
        t1 = pd.to_datetime(end_time, errors="coerce")
        if pd.isna(t1):
            raise ValueError(f"Could not parse end_time={end_time}")
        df = df[df["_ts"] <= t1]

    return df


def _make_frames(df: pd.DataFrame, step_s: int, max_frames: int | None) -> pd.DatetimeIndex:
    if df.empty:
        raise ValueError("No valid rows remain after time parsing/windowing.")

    t0 = df["_ts"].min().floor("s")
    t1 = df["_ts"].max().ceil("s")
    frames = pd.date_range(t0, t1, freq=f"{int(step_s)}s")

    if max_frames is not None and len(frames) > int(max_frames):
        frames = frames[: int(max_frames)]

    return frames


def _format_match_title(matches: pd.DataFrame | None, game_number: int | None) -> str | None:
    """
    Build a nice title: 'YYYY-MM-DD HH:MM — HOME vs AWAY (SCORE)'
    Assumes matches has columns: date, time, home, away, score.
    """
    if matches is None or game_number is None:
        return None

    try:
        row = matches.iloc[int(game_number) - 1]
    except Exception:
        return None

    date_val = row.get("date", "")
    time_val = row.get("time", "")
    home_val = row.get("home", "")
    away_val = row.get("away", "")
    score_val = row.get("score", "")

    try:
        dt = pd.to_datetime(f"{date_val} {time_val}", errors="coerce")
        if pd.isna(dt):
            dt_str = f"{date_val} {time_val}".strip()
        else:
            dt_str = dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        dt_str = f"{date_val} {time_val}".strip()

    title = f"{dt_str} - {home_val} vs {away_val}"
    if str(score_val).strip():
        title += f" ({score_val})"
    return title


def _prep_positions(
    df_xy: pd.DataFrame,
    player_col: str,
    x_col: str,
    y_col: str,
    step_s: int,
    pad_limit_s: float,
    start_time=None,
    end_time=None,
    max_frames: int | None = None,
):
    df = df_xy[[player_col, x_col, y_col]].copy()

    df["_ts"] = _choose_time(df_xy)
    df = df.dropna(subset=["_ts"])
    df = _apply_time_window(df, start_time=start_time, end_time=end_time)
    df = df.sort_values("_ts")

    frames = _make_frames(df, step_s=step_s, max_frames=max_frames)

    players = df[player_col].dropna().unique().tolist()
    nF, nP = len(frames), len(players)
    pos = np.full((nF, nP, 2), np.nan, dtype=float)

    for j, pid in enumerate(players):
        g = df[df[player_col] == pid].dropna(subset=["_ts"])
        if g.empty:
            continue

        s = g.set_index("_ts")[[x_col, y_col]].sort_index()
        r = s.reindex(frames, method="pad")

        last_real = s.index.to_series().reindex(frames, method="pad")
        age = (
            (frames.to_series().reset_index(drop=True) - last_real.reset_index(drop=True))
            .dt.total_seconds()
            .to_numpy()
        )

        r.loc[age > float(pad_limit_s), [x_col, y_col]] = np.nan
        pos[:, j, :] = r[[x_col, y_col]].to_numpy()

    return frames.to_numpy(), players, pos


def _prep_positions_with_status(
    df_xy: pd.DataFrame,
    player_col: str,
    x_col: str,
    y_col: str,
    status_col: str,
    step_s: int,
    pad_limit_s: float,
    start_time=None,
    end_time=None,
    max_frames: int | None = None,
):
    df = df_xy[[player_col, x_col, y_col, status_col]].copy()

    df["_ts"] = _choose_time(df_xy)
    df = df.dropna(subset=["_ts"])
    df = _apply_time_window(df, start_time=start_time, end_time=end_time)
    df = df.sort_values("_ts")

    frames = _make_frames(df, step_s=step_s, max_frames=max_frames)

    players = df[player_col].dropna().unique().tolist()
    nF, nP = len(frames), len(players)

    pos = np.full((nF, nP, 2), np.nan, dtype=float)
    status = np.full((nF, nP), "", dtype=object)

    for j, pid in enumerate(players):
        g = df[df[player_col] == pid].dropna(subset=["_ts"])
        if g.empty:
            continue

        s = g.set_index("_ts")[[x_col, y_col, status_col]].sort_index()
        r = s.reindex(frames, method="pad")

        last_real = s.index.to_series().reindex(frames, method="pad")
        age = (
            (frames.to_series().reset_index(drop=True) - last_real.reset_index(drop=True))
            .dt.total_seconds()
            .to_numpy()
        )

        r.loc[age > float(pad_limit_s), [x_col, y_col]] = np.nan
        pos[:, j, :] = r[[x_col, y_col]].to_numpy()
        status[:, j] = r[status_col].to_numpy()

    return frames.to_numpy(), players, pos, status


def _fade_color(color, fade: float = 0.58):
    """
    Blend a color toward white.
    fade=0 -> original color
    fade=1 -> white
    """
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    white = np.ones(3, dtype=float)
    out = (1.0 - float(fade)) * rgb + float(fade) * white
    return tuple(np.clip(out, 0.0, 1.0))


def _infer_player_teams(
    df_xy: pd.DataFrame,
    players,
    player_col: str = "player_name",
    team_col: str = "team",
):
    """
    Infer one team label per player. Returns:
      player_teams: list aligned with `players`
      unique_teams: unique non-empty teams in player order of appearance
    """
    if team_col not in df_xy.columns:
        return [None for _ in players], []

    tmp = df_xy[[player_col, team_col]].dropna()
    if tmp.empty:
        return [None for _ in players], []

    def _mode_or_first(s):
        s = s.astype(str)
        m = s.mode()
        return str(m.iloc[0]) if not m.empty else str(s.iloc[0])

    team_map = tmp.groupby(player_col)[team_col].agg(_mode_or_first).to_dict()
    player_teams = [team_map.get(pid, None) for pid in players]

    seen = []
    for t in player_teams:
        if t is not None and t not in seen:
            seen.append(t)

    return player_teams, seen


def _build_player_style_arrays(
    df_xy: pd.DataFrame,
    players,
    player_col: str = "player_name",
    team_col: str = "team",
    active_color: str = "dodgerblue",
    bench_color: str = "crimson",
):
    """
    Prepare color arrays aligned with `players`.

    If >=2 teams are present:
      - colors are team-based for both active and bench players
      - bench colors are faded versions of team colors

    If <2 teams are present:
      - preserve the original look:
          * per-player colors for generic/player coloring
          * active_color / bench_color for status coloring
    """
    cmap = plt.get_cmap("tab20").colors
    per_player = np.array([mcolors.to_rgba(cmap[i % len(cmap)]) for i in range(len(players))], dtype=float)

    c_active = np.array(mcolors.to_rgba(active_color), dtype=float)
    c_bench = np.array(mcolors.to_rgba(bench_color), dtype=float)

    player_teams, teams = _infer_player_teams(df_xy, players, player_col=player_col, team_col=team_col)
    multi_team = len(teams) >= 2

    if not multi_team:
        active_faces = np.tile(c_active, (len(players), 1))
        bench_faces = np.tile(c_bench, (len(players), 1))
        edge_active = np.tile(np.array(mcolors.to_rgba("black"), dtype=float), (len(players), 1))
        edge_bench = np.tile(np.array(mcolors.to_rgba("black"), dtype=float), (len(players), 1))
        return {
            "multi_team": False,
            "player_teams": player_teams,
            "teams": teams,
            "per_player": per_player,
            "active_faces": active_faces,
            "bench_faces": bench_faces,
            "edge_active": edge_active,
            "edge_bench": edge_bench,
        }

    preferred = [active_color, bench_color, "mediumseagreen", "orchid", "goldenrod", "slateblue"]
    team_base = {}
    for i, team in enumerate(teams):
        col = preferred[i] if i < len(preferred) else cmap[i % len(cmap)]
        team_base[team] = np.array(mcolors.to_rgba(col), dtype=float)

    active_faces = np.array([team_base.get(t, c_active) for t in player_teams], dtype=float)
    bench_faces = np.array([mcolors.to_rgba(_fade_color(team_base.get(t, c_active), fade=0.60)) for t in player_teams], dtype=float)
    edge_active = np.tile(np.array(mcolors.to_rgba("black"), dtype=float), (len(players), 1))
    edge_bench = np.array([team_base.get(t, c_active) for t in player_teams], dtype=float)

    return {
        "multi_team": True,
        "player_teams": player_teams,
        "teams": teams,
        "per_player": active_faces.copy(),  # use team colors whenever two teams are present
        "active_faces": active_faces,
        "bench_faces": bench_faces,
        "edge_active": edge_active,
        "edge_bench": edge_bench,
    }


# -----------------------------
# Public API
# -----------------------------
def animate_players_simple(
    df_xy: pd.DataFrame,
    pitch_xy,
    player_col: str = "player_name",
    x_col: str = "x_m",
    y_col: str = "y_m",
    step_s: int = 1,
    pad_limit_s: float = 2.0,
    margin_m: float = 6.0,
    fps: int = 25,
    # performance knobs
    start_time=None,
    end_time=None,
    max_frames: int | None = 1500,
    # optional match title
    matches: pd.DataFrame | None = None,
    game_number: int | None = None,
    # trails
    show_trails: bool = False,
    trail_s: float = 8.0,
    trail_lw: float = 1.6,
    trail_alpha: float = 0.25,
    trail_step: int = 1,
    team_col: str = "team",
):
    """
    Simple animation: dots moving on the pitch with a clock.

    start_time/end_time: limit animation to a time window.
    max_frames: hard cap on number of frames.
    show_trails: draw last `trail_s` seconds of each player's path.
    """
    frames, players, pos = _prep_positions(
        df_xy,
        player_col,
        x_col,
        y_col,
        step_s,
        pad_limit_s,
        start_time=start_time,
        end_time=end_time,
        max_frames=max_frames,
    )
    nF, nP = pos.shape[0], pos.shape[1]
    if nP == 0 or nF == 0:
        raise ValueError("No frames or players to animate (check your df_xy/time window).")

    fig, ax = plt.subplots(figsize=(10, 6))

    title = _format_match_title(matches, game_number)
    if title:
        fig.suptitle(title, fontsize=14, y=0.99)

    draw_pitch(ax, pitch_xy, margin_m=margin_m)

    styles = _build_player_style_arrays(
        df_xy,
        players,
        player_col=player_col,
        team_col=team_col,
    )
    player_colors = styles["per_player"]

    offsets_init = np.full((nP, 2), np.nan)
    scat = ax.scatter(
        offsets_init[:, 0],
        offsets_init[:, 1],
        s=40,
        edgecolors="black",
        linewidths=0.6,
        zorder=9,
    )
    scat.set_facecolors(player_colors)

    # Trails (one line per player)
    trail_lines = []
    if show_trails:
        for j in range(nP):
            (ln,) = ax.plot([], [], lw=trail_lw, zorder=8)
            c = player_colors[j].copy()
            c[3] = float(trail_alpha)  # controlled by caller
            ln.set_color(c)
            trail_lines.append(ln)

    clock_txt = ax.text(
        0.5,
        1.02,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
        zorder=10,
        clip_on=False,
    )

    # Ensure at least 2 frames worth of history so a line can exist
    if show_trails:
        K = max(2, int(np.ceil(float(trail_s) / float(step_s))))
    else:
        K = 1
    trail_step = max(1, int(trail_step))

    def init():
        scat.set_offsets(offsets_init)
        clock_txt.set_text("")
        if show_trails:
            for ln in trail_lines:
                ln.set_data([], [])
        return [scat, clock_txt, *trail_lines]

    def update(i):
        XY = pos[i]
        scat.set_offsets(XY)
        clock_txt.set_text(pd.to_datetime(frames[i]).strftime("%H:%M:%S"))

        if show_trails:
            i0 = max(0, i - K + 1)
            window = pos[i0 : i + 1 : trail_step]  # (m, nP, 2)

            for j, ln in enumerate(trail_lines):
                tr = window[:, j, :]
                m = np.isfinite(tr[:, 0]) & np.isfinite(tr[:, 1])
                if m.sum() >= 2:
                    ln.set_data(tr[m, 0], tr[m, 1])
                else:
                    ln.set_data([], [])
        return [scat, clock_txt, *trail_lines]

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=nF,
        interval=1000 / fps,
        blit=False,
    )
    plt.close(fig)
    return ani


def animate_players_with_status(
    df_xy: pd.DataFrame,
    pitch_xy,
    player_col: str = "player_name",
    x_col: str = "x_m",
    y_col: str = "y_m",
    status_col: str = "player_status",
    step_s: int = 1,
    pad_limit_s: float = 2.0,
    margin_m: float = 6.0,
    fps: int = 25,
    active_depth_m: float | None = None,
    # match title
    matches: pd.DataFrame | None = None,
    game_number: int | None = None,
    # colouring options
    colour_by: str = "player",          # "player" or "status"
    active_value: str = "active",
    active_color: str = "dodgerblue",
    bench_color: str = "crimson",
    # performance knobs
    start_time=None,
    end_time=None,
    max_frames: int | None = 1500,
    # trails
    show_trails: bool = False,
    trail_s: float = 8.0,
    trail_lw: float = 1.6,
    trail_alpha: float = 0.25,          # controlled by caller
    trail_step: int = 1,
    trail_only_active: bool = True,     # NEW: only draw trails for active players
    team_col: str = "team",
):
    """
    Animation with status labels and optional trails.

    - If active_depth_m is None, we try df_xy.attrs['active_depth_m'] (keeps pitch shading consistent with labeling).
    - Trails can be restricted to active players only.
    - Title can include full date/time + matchup + score using matches + game_number.
    """
    frames, players, pos, status = _prep_positions_with_status(
        df_xy,
        player_col,
        x_col,
        y_col,
        status_col,
        step_s,
        pad_limit_s,
        start_time=start_time,
        end_time=end_time,
        max_frames=max_frames,
    )
    nF, nP = pos.shape[0], pos.shape[1]
    if nP == 0 or nF == 0:
        raise ValueError("No frames or players to animate (check your df_xy/time window).")

    # Auto-pull active_depth_m from df attrs if not provided
    if active_depth_m is None:
        active_depth_m = df_xy.attrs.get("active_depth_m", None)

    fig, ax = plt.subplots(figsize=(10, 6))

    title = _format_match_title(matches, game_number)
    if title:
        fig.suptitle(title, fontsize=14, y=0.99)

    draw_pitch(
        ax,
        pitch_xy,
        margin_m=margin_m,
        show_outer_box=True,
        active_depth_m=active_depth_m,
        show_active_zone=True,
        show_scale_bar=True,
        scale_bar_length_m=10.0,
    )

    offsets_init = np.full((nP, 2), np.nan)

    styles = _build_player_style_arrays(
        df_xy,
        players,
        player_col=player_col,
        team_col=team_col,
        active_color=active_color,
        bench_color=bench_color,
    )
    player_colors = styles["per_player"]

    scat = ax.scatter(
        offsets_init[:, 0],
        offsets_init[:, 1],
        s=40,
        edgecolors="black",
        linewidths=0.6,
        zorder=9,
    )

    # Trails (one line per player)
    trail_lines = []
    if show_trails:
        for _ in range(nP):
            (ln,) = ax.plot([], [], lw=trail_lw, zorder=8)
            trail_lines.append(ln)

    clock_txt = ax.text(
        0.5,
        1.02,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
        zorder=10,
        clip_on=False,
    )

    status_texts = []
    for _ in range(nP):
        txt = ax.text(
            0,
            0,
            "",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            zorder=10,
            clip_on=True,
        )
        status_texts.append(txt)

    # Ensure at least 2 frames worth of history so a line can exist
    if show_trails:
        K = max(2, int(np.ceil(float(trail_s) / float(step_s))))
    else:
        K = 1
    trail_step = max(1, int(trail_step))

    def init():
        scat.set_offsets(offsets_init)
        clock_txt.set_text("")
        for txt in status_texts:
            txt.set_text("")
            txt.set_visible(False)
        if show_trails:
            for ln in trail_lines:
                ln.set_data([], [])
        return [scat, clock_txt, *status_texts, *trail_lines]

    def update(i):
        XY = pos[i]
        scat.set_offsets(XY)
        clock_txt.set_text(pd.to_datetime(frames[i]).strftime("%H:%M:%S"))

        # Always compute is_active (used for status colouring and for trail_only_active)
        labs = status[i]
        is_active = np.array([str(s) == active_value for s in labs], dtype=bool)

        # Facecolors / edgecolors
        if colour_by == "player":
            face = player_colors.copy()
            edge = styles["edge_active"].copy() if styles["multi_team"] else np.tile(np.array(mcolors.to_rgba("black"), dtype=float), (nP, 1))
        elif colour_by == "status":
            face = np.where(is_active[:, None], styles["active_faces"], styles["bench_faces"])
            edge = np.where(is_active[:, None], styles["edge_active"], styles["edge_bench"])
        else:
            raise ValueError("colour_by must be 'player' or 'status'")

        scat.set_facecolors(face)
        scat.set_edgecolors(edge)

        # Trails: only active if requested, and alpha controlled by caller
        if show_trails:
            i0 = max(0, i - K + 1)
            window = pos[i0 : i + 1 : trail_step]  # (m, nP, 2)

            for j, ln in enumerate(trail_lines):
                if trail_only_active and (not is_active[j]):
                    ln.set_data([], [])
                    continue

                tr = window[:, j, :]
                m = np.isfinite(tr[:, 0]) & np.isfinite(tr[:, 1])
                if m.sum() >= 2:
                    ln.set_data(tr[m, 0], tr[m, 1])
                    c = np.array(face[j], dtype=float).copy()
                    c[3] = float(trail_alpha)  # caller-controlled
                    ln.set_color(c)
                else:
                    ln.set_data([], [])

        # Status labels above players
        for j in range(nP):
            x, y = XY[j]
            if np.isnan(x) or np.isnan(y):
                status_texts[j].set_visible(False)
                continue

            status_texts[j].set_visible(True)
            status_texts[j].set_position((x, y + 0.8))
            lab = status[i, j]
            status_texts[j].set_text("" if (lab is None or lab != lab) else str(lab))

        return [scat, clock_txt, *status_texts, *trail_lines]

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=nF,
        interval=1000 / fps,
        blit=False,
    )
    plt.close(fig)
    return ani

# -----------------------------
# Media / showcase animation (arrows + polarisation overlay)
# -----------------------------
def _vel_from_pos(pos: np.ndarray, step_s: float, k: int = 1):
    """
    pos: (nF, nP, 2)
    Returns:
      vel:   (nF, nP, 2) in m/s (NaN where unavailable)
      speed: (nF, nP)    in m/s
      unit:  (nF, nP, 2) unit direction (0 where speed==0)
    """
    pos = np.asarray(pos, dtype=float)
    nF = pos.shape[0]

    # Finite-difference velocity using a lag of k frames.
    vel = np.full_like(pos, np.nan, dtype=float)
    if nF > k:
        vel[k:, :, :] = (pos[k:, :, :] - pos[:-k, :, :]) / (float(k) * float(step_s))

    speed = np.hypot(vel[:, :, 0], vel[:, :, 1])

    # Unit vectors without divide-by-zero warnings
    unit = np.zeros_like(vel, dtype=float)
    m = np.isfinite(speed) & (speed > 0)

    unit[:, :, 0] = np.divide(
        vel[:, :, 0],
        speed,
        out=np.zeros_like(vel[:, :, 0], dtype=float),
        where=m,
    )
    unit[:, :, 1] = np.divide(
        vel[:, :, 1],
        speed,
        out=np.zeros_like(vel[:, :, 1], dtype=float),
        where=m,
    )
    return vel, speed, unit


def _pmv_series_to_frames(
    df_pmv: pd.DataFrame,
    frames: pd.DatetimeIndex,
    value_col: str = "p_group",
    time_col: str = "_t",
    team=None,
    phase=None,
    team_col: str = "team",
    phase_col: str = "match_phase",
    pad_limit_s: float = 3.0,
):
    """
    Align a (time,value) series (e.g. df_pmv) to the animation frames via forward-fill,
    and blank values when they are older than pad_limit_s.
    """
    d = df_pmv.copy()

    if team is not None and team_col in d.columns:
        d = d[d[team_col] == team]
    if phase is not None and phase_col in d.columns:
        d = d[d[phase_col] == phase]

    if time_col not in d.columns or value_col not in d.columns or d.empty:
        return np.full(len(frames), np.nan, dtype=float)

    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col, value_col]).sort_values(time_col)

    s = d.set_index(time_col)[value_col].astype(float)
    r = s.reindex(frames, method="pad")

    last_real = s.index.to_series().reindex(frames, method="pad")
    age = (frames.to_series().reset_index(drop=True) - last_real.reset_index(drop=True)).dt.total_seconds().to_numpy()
    r.loc[age > float(pad_limit_s)] = np.nan

    return r.to_numpy(dtype=float)


def animate_players_media(
    df_xy: pd.DataFrame,
    pitch_xy,
    player_col: str = "player_name",
    x_col: str = "x_m",
    y_col: str = "y_m",
    status_col: str = "player_status",
    step_s: int = 1,
    pad_limit_s: float = 2.0,
    margin_m: float = 6.0,
    fps: int = 25,
    active_depth_m: float | None = None,
    # match title
    matches: pd.DataFrame | None = None,
    game_number: int | None = None,
    # colouring options
    colour_by: str = "status",          # "player" or "status"
    active_value: str = "active",
    active_color: str = "dodgerblue",
    bench_color: str = "crimson",
    # performance knobs
    start_time=None,
    end_time=None,
    max_frames: int | None = 800,
    # trails
    show_trails: bool = True,
    trail_s: float = 10.0,
    trail_lw: float = 1.8,
    trail_alpha: float = 0.30,
    trail_step: int = 1,
    trail_only_active: bool = True,
    # arrows (players)
    show_arrows: bool = True,
    arrow_min_speed_mps: float = 0.0,
    arrow_len_m: float = 3.4,
    arrow_window_s: float | None = None,   # if None, uses one frame step
    arrow_scale_by_speed: bool = True,
    arrow_speed_ref_mps: float = 5.0,
    arrow_width: float = 0.0046,
    arrow_headwidth: float = 4.8,
    arrow_headlength: float = 6.8,
    arrow_headaxislength: float = 5.6,
    # team mean-direction arrow (optional, no centroid marker is drawn)
    show_team_arrow: bool = False,
    team_arrow_len_m: float = 10.0,
    team_min_speed_mps: float = 0.7,
    # external polarisation series (optional)
    df_pmv: pd.DataFrame | None = None,
    pmv_time_col: str = "_t",
    pmv_value_col: str = "p_group",
    pmv_team=None,
    pmv_phase=None,
    show_pmv_bar: bool = True,
    show_pmv_trace: bool = False,
    pmv_pad_limit_s: float = 4.0,
    # team speed meter
    show_team_speed_bar: bool = True,
    team_speed_max_mps: float | None = None,
    team_speed_value_col: str | None = "v_group_mps",
    team_speed_min_mps: float = 0.2,
    # meter styling (axes coordinates)
    meter_y0: float = 0.18,
    meter_h: float = 0.70,
    meter_w: float = 0.018,
    meter_gap: float = 0.012,
    meter_alpha_bg: float = 0.18,
    phi_meter_color: str = "mediumpurple",
    speed_meter_color: str = "seagreen",
    # names (active players)
    show_names: bool = True,
    name_only_active: bool = True,
    name_fontsize: int = 8,
    name_dy_m: float = 1.2,
    team_col: str = "team",
):
    """
    Media-friendly animation:
      - Status colouring (active vs bench)
      - Direction-of-travel arrows for ALL players once they become active
      - Polarisation gauge (Φ) shown as a vertical meter on the right
      - Team speed meter shown as a vertical meter next to Φ (defaults to df_pmv['v_group_mps'] if provided, else mean active player speed)
      - Optional team mean-direction arrow (no centroid marker)

    Notes
    -----
    - Arrows are shown for players with status == active_value (i.e. "once they become active").
    - Player names are shown once active (by default), placed slightly above the marker.
    - If df_pmv is provided (e.g. output of utils.collective_stats.compute_pmv),
      pmv_value_col is used as Φ; otherwise Φ is computed from player displacement
      directions.
    - The old Φ-vs-time inset is intentionally disabled for this media animation.
    """
    frames_np, players, pos, status = _prep_positions_with_status(
        df_xy,
        player_col,
        x_col,
        y_col,
        status_col,
        step_s,
        pad_limit_s,
        start_time=start_time,
        end_time=end_time,
        max_frames=max_frames,
    )
    nF, nP = pos.shape[0], pos.shape[1]
    if nP == 0 or nF == 0:
        raise ValueError("No frames or players to animate (check your df_xy/time window).")

    # active_depth_m for pitch shading
    if active_depth_m is None:
        active_depth_m = df_xy.attrs.get("active_depth_m", None)

    frames = pd.to_datetime(frames_np)

    # Choose arrow lag for direction estimate
    if arrow_window_s is None:
        k = 1
    else:
        k = max(1, int(round(float(arrow_window_s) / float(step_s))))

    vel, speed, unit = _vel_from_pos(pos, step_s=float(step_s), k=k)

    is_active = (status == active_value)
    valid_xy = np.isfinite(pos[:, :, 0]) & np.isfinite(pos[:, :, 1])
    valid_dir = np.isfinite(unit[:, :, 0]) & np.isfinite(unit[:, :, 1])
    valid_speed = np.isfinite(speed)

    # Team mean direction / polarisation from player directions (active players only)
    team_u = np.zeros((nF, 2), dtype=float)
    phi_from_players = np.full(nF, np.nan, dtype=float)
    for i in range(nF):
        m = is_active[i] & valid_xy[i] & valid_dir[i] & valid_speed[i] & (speed[i] >= float(team_min_speed_mps))
        if m.any():
            mu = np.nanmean(unit[i, m, :], axis=0)
            team_u[i] = mu
            phi_from_players[i] = float(np.hypot(mu[0], mu[1]))

    # Optional external polarisation series
    if df_pmv is not None and show_pmv_bar:
        phi_series = _pmv_series_to_frames(
            df_pmv=df_pmv,
            frames=frames,
            value_col=pmv_value_col,
            time_col=pmv_time_col,
            team=pmv_team,
            phase=pmv_phase,
            pad_limit_s=pmv_pad_limit_s,
        )
    else:
        phi_series = phi_from_players.copy()

    # Clean phi to [0,1]
    phi_series = np.where(np.isfinite(phi_series), np.clip(phi_series, 0.0, 1.0), np.nan)

    # Team speed series
    # Prefer using df_pmv (already computed collective speed) if available, otherwise fall back to mean player speed.
    team_speed = np.full(nF, np.nan, dtype=float)
    used_external_speed = False

    if df_pmv is not None and show_team_speed_bar:
        speed_col = team_speed_value_col
        if speed_col is None:
            for cand in ("v_group_mps", "v_group", "v_mean_mps", "speed_group_mps", "v_group_mean_mps"):
                if cand in df_pmv.columns:
                    speed_col = cand
                    break
        if speed_col is not None and speed_col in df_pmv.columns:
            team_speed = _pmv_series_to_frames(
                df_pmv=df_pmv,
                frames=frames,
                value_col=speed_col,
                time_col=pmv_time_col,
                team=pmv_team,
                phase=pmv_phase,
                pad_limit_s=pmv_pad_limit_s,
            )
            used_external_speed = True

    if not used_external_speed:
        # Fallback: compute from mean speed of active players (from displacement)
        for i in range(nF):
            m = is_active[i] & valid_speed[i] & np.isfinite(speed[i])
            if float(team_speed_min_mps) > 0:
                m = m & (speed[i] >= float(team_speed_min_mps))
            if m.any():
                team_speed[i] = float(np.nanmean(speed[i, m]))

    # Apply min-speed threshold to external series too (optional)
    if float(team_speed_min_mps) > 0:
        team_speed = np.where(
            np.isfinite(team_speed) & (team_speed >= float(team_speed_min_mps)),
            team_speed,
            np.nan,
        )

    if team_speed_max_mps is None:
        vv = team_speed[np.isfinite(team_speed)]
        if vv.size:
            vmax = float(np.nanpercentile(vv, 95))
            vmax = float(np.clip(vmax, 1.0, 12.0))
        else:
            vmax = 8.0
    else:
        vmax = float(team_speed_max_mps)

    # (Optional) origin for team arrow: centroid of active players (not displayed)
    centroid = None
    if show_team_arrow:
        centroid = np.full((nF, 2), np.nan, dtype=float)
        for i in range(nF):
            m = is_active[i] & valid_xy[i]
            if m.any():
                centroid[i, 0] = np.nanmean(pos[i, m, 0])
                centroid[i, 1] = np.nanmean(pos[i, m, 1])

    # Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.84)

    title = _format_match_title(matches, game_number)
    if title:
        fig.suptitle(title, fontsize=14, y=0.99)

    draw_pitch(
        ax,
        pitch_xy,
        margin_m=margin_m,
        show_outer_box=True,
        active_depth_m=active_depth_m,
        show_active_zone=True,
        show_scale_bar=True,
        scale_bar_length_m=10.0,
    )

    # Colours
    styles = _build_player_style_arrays(
        df_xy,
        players,
        player_col=player_col,
        team_col=team_col,
        active_color=active_color,
        bench_color=bench_color,
    )
    player_colors = styles["per_player"]

    offsets_init = np.full((nP, 2), np.nan)
    scat = ax.scatter(
        offsets_init[:, 0],
        offsets_init[:, 1],
        s=42,
        edgecolors="black",
        linewidths=0.6,
        zorder=9,
    )

    # Trails
    trail_lines = []
    if show_trails:
        for _ in range(nP):
            (ln,) = ax.plot([], [], lw=trail_lw, zorder=8)
            trail_lines.append(ln)

    # Names
    name_texts = []
    if show_names:
        for _ in range(nP):
            txt = ax.text(
                0.0, 0.0, "",
                ha="center", va="bottom",
                fontsize=int(name_fontsize),
                color="black",
                zorder=12,
                clip_on=True,
            )
            txt.set_visible(False)
            name_texts.append(txt)

    # Clock
    clock_txt = ax.text(
        0.5,
        1.02,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        zorder=10,
        clip_on=False,
    )

    # Player arrows (quiver)
    if show_arrows:
        q_players = ax.quiver(
            np.zeros(nP), np.zeros(nP),
            np.zeros(nP), np.zeros(nP),
            angles="xy", scale_units="xy", scale=1,
            width=float(arrow_width),
            headwidth=float(arrow_headwidth),
            headlength=float(arrow_headlength),
            headaxislength=float(arrow_headaxislength),
            alpha=0.9,
            zorder=11,
        )
    else:
        q_players = None

    # Team arrow (no centroid marker)
    if show_team_arrow:
        q_team = ax.quiver(
            [0.0], [0.0], [0.0], [0.0],
            angles="xy", scale_units="xy", scale=1,
            width=0.006, alpha=0.95,
            zorder=13,
        )
    else:
        q_team = None

    # Vertical meters (right side): polarisation Φ and team speed
    phi_txt = None
    phi_bg = None
    phi_fg = None
    phi_lab = None

    spd_txt = None
    spd_bg = None
    spd_fg = None
    spd_lab = None

    # Layout: two bars on the right, moved outside the pitch/axes area
    x_phi = 1.035
    x_spd = x_phi - float(meter_gap) - float(meter_w)
    y0 = float(meter_y0)
    H = float(meter_h)
    W = float(meter_w)

    if show_pmv_bar:
        phi_bg = patches.Rectangle((x_phi, y0), W, H, transform=ax.transAxes,
                           facecolor="black", alpha=float(meter_alpha_bg),
                           edgecolor="black", lw=1.0, zorder=20, clip_on=False)
        phi_fg = patches.Rectangle((x_phi, y0), W, 0.001, transform=ax.transAxes,
                           facecolor=phi_meter_color, alpha=0.85,
                           edgecolor="none", zorder=21, clip_on=False)
        ax.add_patch(phi_bg)
        ax.add_patch(phi_fg)

        phi_lab = ax.text(x_phi + W/2, y0 - 0.03, r"$\Phi$",
                          transform=ax.transAxes, ha="center", va="top",
                          fontsize=11, zorder=23, clip_on=False)
        phi_txt = ax.text(x_phi + W/2, y0 + H + 0.02, "",
                          transform=ax.transAxes, ha="center", va="bottom",
                          fontsize=11,
                          bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.80),
                          zorder=23, clip_on=False)

    if show_team_speed_bar:
        spd_bg = patches.Rectangle((x_spd, y0), W, H, transform=ax.transAxes,
                           facecolor="black", alpha=float(meter_alpha_bg),
                           edgecolor="black", lw=1.0, zorder=20, clip_on=False)
        spd_fg = patches.Rectangle((x_spd, y0), W, 0.001, transform=ax.transAxes,
                           facecolor=speed_meter_color, alpha=0.85,
                           edgecolor="none", zorder=21, clip_on=False)
        ax.add_patch(spd_bg)
        ax.add_patch(spd_fg)

        spd_lab = ax.text(x_spd + W/2, y0 - 0.03, r"$\bar{v}$",
                          transform=ax.transAxes, ha="center", va="top",
                          fontsize=11, zorder=23, clip_on=False)
        spd_txt = ax.text(x_spd + W/2, y0 + H + 0.02, "",
                          transform=ax.transAxes, ha="center", va="bottom",
                          fontsize=11,
                          bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.80),
                          zorder=23, clip_on=False)

    # Trail history length
    if show_trails:
        K = max(2, int(np.ceil(float(trail_s) / float(step_s))))
    else:
        K = 1
    trail_step = max(1, int(trail_step))

    def init():
        scat.set_offsets(offsets_init)
        clock_txt.set_text("")
        if show_trails:
            for ln in trail_lines:
                ln.set_data([], [])
        if q_players is not None:
            q_players.set_offsets(np.zeros((nP, 2)))
            q_players.set_UVC(np.zeros(nP), np.zeros(nP))
        if q_team is not None:
            q_team.set_offsets([[0.0, 0.0]])
            q_team.set_UVC([0.0], [0.0])
        if show_names:
            for txt in name_texts:
                txt.set_text("")
                txt.set_visible(False)
        if show_pmv_bar and phi_txt is not None:
            phi_txt.set_text("")
        if show_team_speed_bar and spd_txt is not None:
            spd_txt.set_text("")
        return [
            scat, clock_txt,
            *(trail_lines if show_trails else []),
            *( [q_players] if q_players is not None else []),
            *( [q_team] if q_team is not None else []),
            *(name_texts if show_names else []),
            *( [phi_bg, phi_fg, phi_txt, phi_lab] if show_pmv_bar else []),
            *( [spd_bg, spd_fg, spd_txt, spd_lab] if show_team_speed_bar else []),
        ]

    def update(i):
        XY = pos[i]
        act = is_active[i]

        # facecolors / edgecolors
        if colour_by == "player":
            face = player_colors.copy()
            edge = styles["edge_active"].copy() if styles["multi_team"] else np.tile(np.array(mcolors.to_rgba("black"), dtype=float), (nP, 1))
        elif colour_by == "status":
            face = np.where(act[:, None], styles["active_faces"], styles["bench_faces"])
            edge = np.where(act[:, None], styles["edge_active"], styles["edge_bench"])
        else:
            raise ValueError("colour_by must be 'player' or 'status'")
        scat.set_facecolors(face)
        scat.set_edgecolors(edge)
        scat.set_offsets(XY)

        clock_txt.set_text(pd.to_datetime(frames_np[i]).strftime("%H:%M:%S"))

        # Trails
        if show_trails:
            i0 = max(0, i - K + 1)
            window = pos[i0 : i + 1 : trail_step]
            for j, ln in enumerate(trail_lines):
                if trail_only_active and (not act[j]):
                    ln.set_data([], [])
                    continue
                tr = window[:, j, :]
                m = np.isfinite(tr[:, 0]) & np.isfinite(tr[:, 1])
                if m.sum() >= 2:
                    ln.set_data(tr[m, 0], tr[m, 1])
                    c = np.array(face[j], dtype=float).copy()
                    c[3] = float(trail_alpha)
                    ln.set_color(c)
                else:
                    ln.set_data([], [])

        # Names (show once active by default)
        if show_names:
            for j, txt in enumerate(name_texts):
                x, y = XY[j, 0], XY[j, 1]
                ok = np.isfinite(x) and np.isfinite(y)
                if name_only_active:
                    ok = ok and bool(act[j])
                if ok:
                    txt.set_text(str(players[j]))
                    txt.set_position((float(x), float(y) + float(name_dy_m)))
                    txt.set_visible(True)
                else:
                    txt.set_visible(False)

        # Player arrows (shown for active players once they become active)
        if q_players is not None:
            U = np.zeros(nP, dtype=float)
            V = np.zeros(nP, dtype=float)

            m_ok = valid_xy[i] & valid_dir[i] & valid_speed[i] & act
            if float(arrow_min_speed_mps) > 0:
                m_ok = m_ok & (speed[i] >= float(arrow_min_speed_mps))

            if arrow_scale_by_speed:
                L = float(arrow_len_m) * np.clip(speed[i] / float(arrow_speed_ref_mps), 0.0, 1.0)
                L = np.where(np.isfinite(L), L, 0.0)
            else:
                L = np.full(nP, float(arrow_len_m), dtype=float)

            U[m_ok] = L[m_ok] * unit[i, m_ok, 0]
            V[m_ok] = L[m_ok] * unit[i, m_ok, 1]

            XY0 = np.where(np.isfinite(XY), XY, 0.0)
            q_players.set_offsets(XY0)
            q_players.set_UVC(U, V)

        # Optional team mean-direction arrow (no centroid marker)
        if q_team is not None and centroid is not None:
            cx, cy = centroid[i]
            if np.isfinite(cx) and np.isfinite(cy) and np.isfinite(phi_from_players[i]):
                phi_here = phi_from_players[i]
                ux, uy = team_u[i]
                # scale by phi (visual)
                Uc = float(team_arrow_len_m) * float(phi_here) * float(ux)
                Vc = float(team_arrow_len_m) * float(phi_here) * float(uy)
                q_team.set_offsets([[cx, cy]])
                q_team.set_UVC([Uc], [Vc])
            else:
                q_team.set_offsets([[0.0, 0.0]])
                q_team.set_UVC([0.0], [0.0])

        # Polarisation vertical meter
        if show_pmv_bar and phi_fg is not None and phi_txt is not None:
            phi_val = phi_series[i]
            if np.isfinite(phi_val):
                phi_fg.set_height(H * float(phi_val))
                phi_txt.set_text(f"{float(phi_val):0.2f}")
            else:
                phi_fg.set_height(0.001)
                phi_txt.set_text("--")

        # Team speed vertical meter
        if show_team_speed_bar and spd_fg is not None and spd_txt is not None:
            v = team_speed[i]
            if np.isfinite(v) and vmax > 0:
                vn = float(np.clip(v / vmax, 0.0, 1.0))
                spd_fg.set_height(H * vn)
                spd_txt.set_text(f"{float(v):0.1f} m/s")
            else:
                spd_fg.set_height(0.001)
                spd_txt.set_text("--")

        artists = [scat, clock_txt]
        if show_trails:
            artists.extend(trail_lines)
        if q_players is not None:
            artists.append(q_players)
        if q_team is not None:
            artists.append(q_team)
        if show_names:
            artists.extend(name_texts)
        if show_pmv_bar:
            artists.extend([phi_bg, phi_fg, phi_txt, phi_lab])
        if show_team_speed_bar:
            artists.extend([spd_bg, spd_fg, spd_txt, spd_lab])
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=nF,
        interval=1000 / fps,
        blit=False,
    )
    plt.close(fig)
    return ani
