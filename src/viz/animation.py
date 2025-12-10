"""
Animation utilities for player movement on the pitch.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.pitch_draw import draw_pitch


def _choose_time(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce")
    if "time" in df.columns:
        return pd.to_datetime(df["time"], errors="coerce")
    raise ValueError("df must have 'timestamp' or 'time'.")


def _prep_positions(
    df_xy: pd.DataFrame,
    player_col: str,
    x_col: str,
    y_col: str,
    step_s: int,
    pad_limit_s: float,
):
    # Use local df
    df = df_xy[[player_col, x_col, y_col]].copy()

    # Parse timestamps correctly
    df["_ts"] = _choose_time(df_xy)
    df = df.dropna(subset=["_ts"]).sort_values("_ts")

    if df.empty:
        raise ValueError("No valid timestamps after parsing 'time'/'timestamp'.")

    t0 = df["_ts"].min().floor("s")
    t1 = df["_ts"].max().ceil("s")
    frames = pd.date_range(t0, t1, freq=f"{int(step_s)}s")

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
            (
                frames.to_series().reset_index(drop=True)
                - last_real.reset_index(drop=True)
            )
            .dt.total_seconds()
            .to_numpy()
        )

        r.loc[age > float(pad_limit_s), [x_col, y_col]] = np.nan

        pos[:, j, :] = r[[x_col, y_col]].to_numpy()

    return frames.to_numpy(), players, pos


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
):
    """
    Simple animation: dots moving on the pitch with a clock.
    """
    frames, players, pos = _prep_positions(
        df_xy, player_col, x_col, y_col, step_s, pad_limit_s
    )
    nF, nP = pos.shape[0], pos.shape[1]
    if nP == 0 or nF == 0:
        raise ValueError("No frames or players to animate (check your df_xy).")

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax, pitch_xy, margin_m=margin_m)

    cmap = plt.get_cmap("tab20").colors
    colors = [cmap[i % len(cmap)] for i in range(nP)]

    offsets_init = np.full((nP, 2), np.nan)
    scat = ax.scatter(
        offsets_init[:, 0],
        offsets_init[:, 1],
        s=40,
        c=colors,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

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

    def init():
        scat.set_offsets(offsets_init)
        clock_txt.set_text("")
        return scat, clock_txt

    def update(i):
        XY = pos[i]
        scat.set_offsets(XY)
        clock_txt.set_text(pd.to_datetime(frames[i]).strftime("%H:%M:%S"))
        return scat, clock_txt

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


def _prep_positions_with_status(
    df_xy: pd.DataFrame,
    player_col: str,
    x_col: str,
    y_col: str,
    status_col: str,
    step_s: int,
    pad_limit_s: float,
):
    df = df_xy[[player_col, x_col, y_col, status_col]].copy()
    df["_ts"] = _choose_time(df_xy)
    df = df.dropna(subset=["_ts"]).sort_values("_ts")

    if df.empty:
        raise ValueError("No valid timestamps after parsing 'time'/'timestamp'.")

    t0 = df["_ts"].min().floor("s")
    t1 = df["_ts"].max().ceil("s")
    frames = pd.date_range(t0, t1, freq=f"{int(step_s)}s")

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
            (
                frames.to_series().reset_index(drop=True)
                - last_real.reset_index(drop=True)
            )
            .dt.total_seconds()
            .to_numpy()
        )

        r.loc[age > float(pad_limit_s), [x_col, y_col]] = np.nan

        pos[:, j, :] = r[[x_col, y_col]].to_numpy()
        status[:, j] = r[status_col].to_numpy()

    return frames.to_numpy(), players, pos, status


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
    matches: pd.DataFrame | None = None,
    game_number: int | None = None,
):
    """
    Animation with player_status labels and coloured pitch (active/bench zones).
    """
    frames, players, pos, status = _prep_positions_with_status(
        df_xy, player_col, x_col, y_col, status_col, step_s, pad_limit_s
    )
    nF, nP = pos.shape[0], pos.shape[1]
    if nP == 0 or nF == 0:
        raise ValueError("No frames or players to animate (check your df_xy).")

    fig, ax = plt.subplots(figsize=(10, 6))

    # optional title from matches table
    if matches is not None and game_number is not None:
        try:
            row = matches.iloc[game_number - 1]
            date_val = row.get("date", "")
            time_val = row.get("time", "")
            home_val = row.get("home", "")
            away_val = row.get("away", "")
            score_val = row.get("score", "")

            try:
                dt_str = pd.to_datetime(f"{date_val} {time_val}").strftime(
                    "%Y-%m-%d %H:%M"
                )
            except Exception:
                dt_str = f"{date_val} {time_val}"

            title_str = f"{dt_str} — {home_val} vs {away_val} ({score_val})"
            fig.suptitle(title_str, fontsize=14, y=0.99)
        except Exception:
            pass

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

    cmap = plt.get_cmap("tab20").colors
    colors = [cmap[i % len(cmap)] for i in range(nP)]

    offsets_init = np.full((nP, 2), np.nan)
    scat = ax.scatter(
        offsets_init[:, 0],
        offsets_init[:, 1],
        s=40,
        c=colors,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

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
    for j in range(nP):
        txt = ax.text(
            0,
            0,
            "",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            zorder=4,
            clip_on=True,
        )
        status_texts.append(txt)

    def init():
        scat.set_offsets(offsets_init)
        clock_txt.set_text("")
        for txt in status_texts:
            txt.set_text("")
            txt.set_visible(False)
        return [scat, clock_txt, *status_texts]

    def update(i):
        XY = pos[i]
        scat.set_offsets(XY)
        clock_txt.set_text(pd.to_datetime(frames[i]).strftime("%H:%M:%S"))

        for j in range(nP):
            x, y = XY[j]
            if np.isnan(x) or np.isnan(y):
                status_texts[j].set_visible(False)
                continue

            status_texts[j].set_visible(True)
            status_texts[j].set_position((x, y + 0.8))

            lab = status[i, j]
            status_texts[j].set_text("" if (lab is None or lab != lab) else str(lab))

        return [scat, clock_txt, *status_texts]

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
