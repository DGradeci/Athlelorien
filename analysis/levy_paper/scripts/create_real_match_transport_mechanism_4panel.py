from __future__ import annotations

import json
import importlib.util
import shutil
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, ConnectionPatch, Ellipse, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import ScaledTranslation

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[3]
LEVY_DIR = ROOT / "analysis" / "levy_paper"
METADATA_DIR = LEVY_DIR / "metadata"
DATA_ROOT = ROOT / "analysis" / "levy_paper" / "data"
DATA_DIR = DATA_ROOT / "processed" / "all_team_2020_2021_sticky_active"
OUT_DIR = ROOT / "analysis" / "levy_paper" / "outputs" / "final" / "figure1_transport_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARE_DIR = ROOT / "analysis" / "levy_paper" / "chatgpt_share"
SHARE_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_repo_module(module_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / rel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {rel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pitch_calibration = load_repo_module("fig1_pitch_calibration", "src/utils/pitch_calibration.py")
player_status = load_repo_module("fig1_player_status", "src/utils/player_status.py")
pitch_draw = load_repo_module("fig1_pitch_draw", "src/viz/pitch_draw.py")
attach_xy_from_pitch = pitch_calibration.attach_xy_from_pitch
calibrate_pitch_from_df = pitch_calibration.calibrate_pitch_from_df
polygon_latlon_centroid = pitch_calibration.polygon_latlon_centroid
polygon_latlon_to_xy = pitch_calibration.polygon_latlon_to_xy
label_active_players = player_status.label_active_players
draw_pitch = pitch_draw.draw_pitch

try:  # noqa: E402
    from analysis.levy_paper.util.paper_utils import (
        ACTIVE_PLAYER_METHOD as PAPER_ACTIVE_PLAYER_METHOD,
        ACTIVATE_S as PAPER_ACTIVATE_S,
        BENCH_OFF_S as PAPER_BENCH_OFF_S,
        MAX_ACTIVE_PLAYERS as PAPER_MAX_ACTIVE_PLAYERS,
        configure_paper_plotting,
    )
except Exception:  # pragma: no cover
    configure_paper_plotting = None
    PAPER_ACTIVE_PLAYER_METHOD = "sticky_hierarchical_active_xi"
    PAPER_ACTIVATE_S = 70
    PAPER_BENCH_OFF_S = 110
    PAPER_MAX_ACTIVE_PLAYERS = 11


WINDOW_DURATION_S = 20
ACTIVE_DEPTH_M = 6.0
ACTIVE_PLAYER_METHOD = PAPER_ACTIVE_PLAYER_METHOD
ACTIVATE_S = PAPER_ACTIVATE_S
BENCH_OFF_S = PAPER_BENCH_OFF_S
MAX_ACTIVE_PLAYERS = PAPER_MAX_ACTIVE_PLAYERS
TURN_THRESHOLD_DEG = 30.0
SEGMENT_MIN_DURATION_S = 4.0
SEGMENT_SEQUENCE_RUNS = (4,)
LOCAL_TANGENT_SAMPLES = 4
SNAPSHOT_FUTURE_S = 8
SNAPSHOT_TRAIL_S = 3
FORMATION_INSET_EXPAND = 1.95
FORMATION_ARROW_START_M = 0.0
FORMATION_ARROW_END_M = 52.0
FORMATION_INSET_VIEW_SPAN_M = 230.0
FORMATION_OUTLINE_PAD_M = 12.0
FORMATION_CENTROID_ARROW_LENGTH_M = 52.0
FORMATION_CENTROID_PLAYER_MIN_SEP_M = 16.0
FORMATION_PLAYER_ARROW_MIN_SEP_M = 28.0
FORMATION_PLAYER_ARROW_SPREAD_ITER = 60
PANEL_C_CONNECTOR_LW = 0.85
PANEL_C_CONNECTOR_ALPHA = 0.90
PANEL_C_CONNECTOR_DASH = (0, (2.3, 2.3))
PANEL_C_CONNECTOR_ANCHOR_SPREAD = 0.40
PAPER_FONT_BASE = 11.5
FIGURE_SIZE = (12.4, 7.15)
FIGURE_SUBPLOT_ADJUST = {
    "left": 0.040,
    "right": 0.992,
    "bottom": 0.075,
    "top": 0.955,
}
FIGURE_GRID_HSPACE = 0.20
FIGURE_GRID_WSPACE = 0.02
FIGURE_TOP_PANEL_GAP_ADJUST = -0.050
PANEL_C_GRID_WSPACE = 0.12
PANEL_LABEL_XY = (-0.002, 1.010)
PANEL_LABEL_FONTSIZE = 15.0
PANEL_A_LEGEND_LOC = "upper right"
PANEL_A_LEGEND_ANCHOR = (0.810, 0.845)
PANEL_A_LEGEND_FONTSIZE = 9.6
PANEL_B_PITCH_MARGIN_M = 9.0
PANEL_B_PITCH_FILL_COLOR = "#f7f6f2"
PANEL_B_OUTER_FILL_COLOR = "#d8d8d3"
PANEL_B_OUTER_FILL_ALPHA = 0.72
PANEL_B_RUN_LABEL_NUDGES = [
    (0.045, 0.025),
    (-0.065, 0.001),
    (0.006, 0.040),
    (-0.055, -0.030),
]
PANEL_B_ANGLE_LABEL_RADIUS_SCALE = 1.45
PANEL_B_ANGLE_LABEL_OFFSET = (0.010, 0.007)
PANEL_B_START_LABEL_OFFSET = (0.000, -0.075)
PANEL_B_END_LABEL_OFFSET = (0.000, -0.070)
PANEL_C_LEGEND_LOC = "lower center"
PANEL_C_LEGEND_ANCHOR = (0.50, 1.02)
PANEL_C_LEGEND_FONTSIZE = 9.4
PANEL_C_LEGEND_NCOL = 2
PANEL_A_PITCH_MARGIN_M = 9.0
PANEL_A_ACTIVE_TRAJECTORY_COLOR = "#111111"
PANEL_A_ACTIVE_TRAJECTORY_LW = 0.82
PANEL_A_ACTIVE_TRAJECTORY_ALPHA = 0.62
PANEL_C_INSET_BOUNDS = {
    "high": [0.085, 0.020, 0.300, 0.530],
    "low": [0.795, 0.075, 0.225, 0.510],
}
SHOW_FIGURE_TITLES = False
SHOW_PANEL_C_SNAPSHOT_LABELS = False
SHOW_PANEL_C_SNAPSHOT_SPANS = False
POLARISATION_PROFILE_S = 20
POLARISATION_CONTEXT_S = 15
POLARISATION_ROLLING_WINDOW_S = 8
POLARISATION_MIN_GAP_S = 25
POLARISATION_MAX_GAP_S = 85
TRAJECTORY_FILE = "trajectory_long_2020_2021_all_teams_pitchfix_sticky_active.parquet"
RUNS_FILE = "centroid_order_runs_2020_2021_all_teams_pitchfix_sticky_active.parquet"
PMV_FILE = "df_pmv_2020_2021_all_teams_pitchfix_sticky_active.parquet"
META_FILE = "multiseason_raw_pitch_metadata_2020_2021_all_teams_pitchfix_sticky_active.parquet"
RAW_STATUS_CACHE_DIR = DATA_ROOT / "active_filter_tuning_raw_cache"
PANEL_A_SEASON = "2021"
PANEL_A_MATCH_ID = "2021-10-16"
PANEL_A_MATCH_PHASE = "2H"
PANEL_A_TEAM = "rosenborg"
PANEL_A_SOURCE_KEY = "A"
PANEL_A_T0 = "2021-10-16T16:00:00"
PANEL_A_T1 = "2021-10-16T16:00:20"
PANEL_B_MATCH_ID = "2021-05-29"
PANEL_B_MATCH_PHASE = "1H"
PANEL_B_TEAM = "rosenborg"
PANEL_B_SOURCE_KEY = "A"
PANEL_B_T0 = "2021-05-29T16:25:08"
PANEL_B_T1 = "2021-05-29T16:26:07"
PANEL_C_MATCH_ID = None
PANEL_C_MATCH_PHASE = None
PANEL_C_TEAM = None
PANEL_C_SOURCE_KEY = None

COLORS = {
    "black": "#111111",
    "grey": "#5f666a",
    "dark_grey": "#383d40",
    "red": "#d7191c",
    "centroid": "#b43c3c",
    "bench": "#8b1a1a",
    "blue": "#0072B2",
    "green": "#009E73",
    "purple": "#7A3EB1",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "magenta": "#CC79A7",
    "teal": "#00A6B2",
    "navy": "#1f4e79",
    "steel": "#6f88a8",
    "gold": "#b8a365",
    "slate": "#4f5b62",
    "paper": "#f7f6f2",
    "outer": "#d8d8d3",
    "pitch_green": "#00f078",
    "active_green": "#00f078",
    "bench_red": "#ff6b6b",
}

FIGURE_TEXT = {
    "figure_caption": (
        "Figure 1. Real-match schematic of player trajectories, centroid-run segmentation, "
        "and collective polarisation."
    ),
    "panel_a_caption": "Real 20 s player and centroid trajectories with active and bench-player states.",
    "panel_b_caption": "Real centroid trajectory partitioned into runs at large changes in direction.",
    "panel_c_caption": "Polarisation time profile with high- and low-polarisation formation snapshots.",
    "panel_a_letter": "A.",
    "panel_b_letter": "B.",
    "panel_c_letter": "C.",
    "panel_a_player_trajectories_label": "Player trajectories",
    "panel_a_centroid_trajectory_label": "Centroid trajectory",
    "panel_a_bench_players_label": "Bench players",
    "panel_b_run_label_template": "run {n}",
    "panel_b_start_label": "start",
    "panel_b_end_label": "end",
    "panel_b_angle_label_template": r"$\Delta\theta={angle:.0f}^\circ>\theta_c$",
    "panel_b_title": "centroid run partition",
    "panel_c_iqr_label_template": "Local variability ({window_s} s IQR)",
    "panel_c_raw_label": "Polarisation trajectory",
    "panel_c_player_direction_label": "Player direction",
    "panel_c_centroid_direction_label": "Centroid direction",
    "panel_c_low_centroid_direction_label": "Centroid direction (low p)",
    "panel_c_high_centroid_direction_label": "Centroid direction (high p)",
    "panel_c_xlabel_template": "Time from kick-off (s)",
    "panel_c_ylabel": r"Polarisation $p(t)$",
    "panel_c_title": "polarisation time profile with matched real frames",
    "panel_c_snapshot_label_template": "{state} frame\np={p:.3f}",
    "panel_c_inset_title_template": "{state}-polarisation frame",
    "panel_c_profile_inset_xlabel": "s",
    "panel_c_profile_inset_ylabel": "p",
    "panel_c_profile_inset_title": "polarisation profile",
    "legacy_higher_polarisation_title": "higher polarisation",
    "legacy_lower_polarisation_title": "lower polarisation",
    "legacy_run_segmentation_title": "run segmentation",
}


def ascii_safe(value: str) -> str:
    text = str(value)
    try:
        text = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        pass
    text = (
        text.replace("ø", "o")
        .replace("Ø", "O")
        .replace("å", "a")
        .replace("Å", "A")
        .replace("æ", "ae")
        .replace("Æ", "Ae")
    )
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def norm_label(value: str) -> str:
    return " ".join(ascii_safe(str(value)).lower().split())


def fig_text(key: str, default: str | None = None, **kwargs) -> str:
    text = FIGURE_TEXT.get(key, default if default is not None else key)
    if text is None:
        return ""
    text = str(text)
    if not kwargs:
        return text
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return text


def load_pitch_registry() -> dict:
    return json.loads((METADATA_DIR / "pitches" / "toppserien_pitches.json").read_text(encoding="utf-8-sig"))


def load_schedule_meta(season: str, match_id: str, team: str) -> dict:
    schedule_path = METADATA_DIR / "schedules" / ("kamper_2021.xlsx" if str(season) == "2021" else "kamper_2020.xlsx")
    if not schedule_path.exists():
        return {"stadium_name": "Koteng Arena", "home": "", "away": "", "score": "", "scheduled_stadium": ""}
    sched = pd.read_excel(schedule_path).rename(
        columns={
            "Dato": "date",
            "Tid": "time",
            "Hjemmelag": "home",
            "Bortelag": "away",
            "Resultat": "score",
            "Bane": "stadium",
        }
    )
    sched["date"] = pd.to_datetime(sched["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sched["home_norm"] = sched["home"].map(norm_label)
    sched["away_norm"] = sched["away"].map(norm_label)
    team_norm = norm_label(team)
    rows = sched.loc[
        sched["date"].astype(str).eq(str(match_id))
        & (sched["home_norm"].eq(team_norm) | sched["away_norm"].eq(team_norm))
    ]
    if rows.empty:
        return {"stadium_name": "Koteng Arena", "home": "", "away": "", "score": "", "scheduled_stadium": ""}
    row = rows.iloc[0]
    return {
        "stadium_name": str(row.get("stadium", "Koteng Arena") or "Koteng Arena"),
        "scheduled_stadium": str(row.get("stadium", "") or ""),
        "home": str(row.get("home", "")),
        "away": str(row.get("away", "")),
        "score": str(row.get("score", "")),
    }


def load_pitch_xy(stadium: str = "Koteng Arena") -> list[tuple[float, float]]:
    pitches = load_pitch_registry()
    if stadium not in pitches:
        stadium = "Koteng Arena"
    coords = pitches[stadium]["coords"]
    c_lat, c_lon = polygon_latlon_centroid(coords)
    p = polygon_latlon_to_xy(coords, c_lat, c_lon)
    centered = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    v0, v1 = vt[0], vt[1]
    if v0[0] < 0:
        v0, v1 = -v0, -v1
    rot = np.column_stack((v0, v1))
    return [tuple(pt) for pt in p @ rot]


def pitch_bounds(pitch_xy: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    p = np.asarray(pitch_xy, dtype=float)
    xmin, ymin = p.min(axis=0)
    xmax, ymax = p.max(axis=0)
    return xmin, xmax, ymin, ymax


def setup_animation_pitch(ax, pitch_xy, *, scale_bar: bool = False) -> None:
    draw_pitch(
        ax,
        pitch_xy,
        margin_m=6.0,
        show_outer_box=True,
        active_depth_m=ACTIVE_DEPTH_M,
        show_active_zone=True,
        show_scale_bar=scale_bar,
        scale_bar_length_m=10.0,
    )


def setup_active_bench_pitch(ax, pitch_xy, *, scale_bar: bool = False, margin_m: float = 6.0) -> None:
    draw_pitch(
        ax,
        pitch_xy,
        margin_m=margin_m,
        show_outer_box=True,
        active_depth_m=ACTIVE_DEPTH_M,
        show_active_zone=True,
        show_scale_bar=scale_bar,
        scale_bar_length_m=10.0,
        show_outer_fill=True,
        outer_fill_color=COLORS["outer"],
        outer_fill_alpha=0.70,
        pitch_fill_color=COLORS["pitch_green"],
        bench_zone_color=COLORS["bench_red"],
        bench_zone_alpha=1.0,
        active_zone_color=COLORS["active_green"],
        active_zone_alpha=1.0,
    )


def setup_neutral_pitch(ax, pitch_xy, *, scale_bar: bool = False, margin_m: float = 5.0) -> None:
    draw_pitch(
        ax,
        pitch_xy,
        margin_m=margin_m,
        show_outer_box=True,
        active_depth_m=None,
        show_active_zone=False,
        show_scale_bar=scale_bar,
        scale_bar_length_m=10.0,
        show_outer_fill=True,
        outer_fill_color=PANEL_B_OUTER_FILL_COLOR,
        outer_fill_alpha=PANEL_B_OUTER_FILL_ALPHA,
        pitch_fill_color=PANEL_B_PITCH_FILL_COLOR,
    )


def arrow(ax, start, end, *, color, lw=1.4, ms=12, z=6) -> None:
    if not np.all(np.isfinite([start[0], start[1], end[0], end[1]])):
        return
    if np.hypot(end[0] - start[0], end[1] - start[1]) < 0.15:
        return
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def curved_arrow(ax, start, end, *, color, lw=1.2, ms=9, bend=0.15, alpha=1.0, z=6) -> None:
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    delta = end_arr - start_arr
    length = float(np.linalg.norm(delta))
    if length < 0.15:
        return
    normal = np.asarray([-delta[1], delta[0]], dtype=float) / (length + 1e-9)
    control = 0.5 * (start_arr + end_arr) + bend * length * normal
    path = MplPath(
        [tuple(start_arr), tuple(control), tuple(end_arr)],
        [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
    )
    ax.add_patch(PathPatch(path, fill=False, color=color, lw=lw, alpha=alpha, capstyle="round", zorder=z))
    tangent = end_arr - control
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    arrow_start = end_arr - min(4.0, 0.28 * length) * tangent
    arrow(ax, tuple(arrow_start), tuple(end_arr), color=color, lw=lw, ms=ms, z=z + 0.2)


def draw_panel_label(ax, label: str) -> None:
    key = f"panel_{str(label).strip().lower()}_letter"
    text = fig_text(key, f"{label}.")
    ax.text(
        PANEL_LABEL_XY[0],
        PANEL_LABEL_XY[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="normal",
        zorder=30,
        clip_on=False,
    )


def adjust_axis_x(ax, dx: float) -> None:
    if abs(float(dx)) < 1e-12:
        return
    pos = ax.get_position()
    ax.set_position([pos.x0 + float(dx), pos.y0, pos.width, pos.height])


def apply_top_panel_gap_adjust(fig, ax_a, ax_b) -> None:
    """Move the rendered top-row pitch axes after equal-aspect sizing."""
    if abs(float(FIGURE_TOP_PANEL_GAP_ADJUST)) < 1e-12:
        return
    fig.canvas.draw()
    adjust_axis_x(ax_a, -0.5 * FIGURE_TOP_PANEL_GAP_ADJUST)
    adjust_axis_x(ax_b, 0.5 * FIGURE_TOP_PANEL_GAP_ADJUST)


def draw_bench_excluded_markers(ax, pitch_xy) -> int:
    xmin, xmax, ymin, ymax = pitch_bounds(pitch_xy)
    w = xmax - xmin
    h = ymax - ymin
    marker_xy = [
        (xmin + 0.12 * w, ymax - 0.035 * h),
        (xmin + 0.30 * w, ymax - 0.035 * h),
        (xmin + 0.68 * w, ymax - 0.035 * h),
        (xmax - 0.10 * w, ymax - 0.035 * h),
        (xmin + 0.10 * w, ymin + 0.035 * h),
        (xmin + 0.28 * w, ymin + 0.035 * h),
        (xmax - 0.28 * w, ymin + 0.035 * h),
        (xmax - 0.12 * w, ymin + 0.035 * h),
        (xmin + 0.045 * w, ymin + 0.53 * h),
        (xmax - 0.045 * w, ymin + 0.47 * h),
    ]
    xy = np.asarray(marker_xy, dtype=float)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        marker="x",
        s=20,
        color=COLORS["bench"],
        linewidths=1.0,
        zorder=12,
    )
    ax.text(
        xmin + 0.015 * w,
        ymax - 0.040 * h,
        "bench/excluded",
        ha="left",
        va="center",
        fontsize=8.0,
        color=COLORS["bench"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.0),
        zorder=13,
    )
    return int(len(marker_xy))


def select_window_if_needed(traj: pd.DataFrame, duration_s: int = WINDOW_DURATION_S) -> dict:
    traj = traj.copy()
    traj["t"] = pd.to_datetime(traj["t"])
    best: dict | None = None
    for keys, g in traj.groupby(["match_id", "match_phase", "team", "source_key"], sort=False):
        match_id, phase, team, source = keys
        cent = g[g["track_type"].eq("centroid")].drop_duplicates("t").sort_values("t").reset_index(drop=True)
        players = g[g["track_type"].eq("player_abs")]
        if len(cent) <= duration_s or players.empty:
            continue
        candidates = []
        for i in range(0, len(cent) - duration_s):
            t0 = cent.loc[i, "t"]
            t1 = cent.loc[i + duration_s, "t"]
            if (t1 - t0).total_seconds() != duration_s:
                continue
            d = float(np.hypot(cent.loc[i + duration_s, "x_m"] - cent.loc[i, "x_m"], cent.loc[i + duration_s, "y_m"] - cent.loc[i, "y_m"]))
            candidates.append((d, pd.Timestamp(t0)))
        for _, t0 in sorted(candidates, reverse=True)[:120]:
            t1 = t0 + pd.Timedelta(seconds=duration_s)
            pwin = players[(players["t"] >= t0) & (players["t"] <= t1)]
            counts = pwin.groupby("player_name")["t"].nunique()
            keep = counts[counts >= duration_s - 1].index.tolist()
            if len(keep) < 10:
                continue
            cwin = cent[(cent["t"] >= t0) & (cent["t"] <= t1)]
            d = float(np.hypot(cwin.iloc[-1]["x_m"] - cwin.iloc[0]["x_m"], cwin.iloc[-1]["y_m"] - cwin.iloc[0]["y_m"]))
            score = d + 0.15 * len(keep)
            candidate = {
                "match_id": str(match_id),
                "match_phase": str(phase),
                "team": str(team),
                "source_key": str(source),
                "t0": t0,
                "t1": t1,
                "n_players": int(len(keep)),
                "centroid_displacement_m": d,
                "selection_score": score,
            }
            if best is None or candidate["selection_score"] > best["selection_score"]:
                best = candidate
    if best is None:
        raise RuntimeError("Could not find a suitable real-match trajectory window.")
    return best


def load_panel_a_status_window(window: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict] | None:
    raw_path = RAW_STATUS_CACHE_DIR / f"{PANEL_A_SEASON}_{PANEL_A_MATCH_ID}_{PANEL_A_SOURCE_KEY}_1hz.parquet"
    if not raw_path.exists():
        return None

    raw = pd.read_parquet(raw_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "lat", "lon", "player_name"]).copy()
    if raw.empty:
        return None

    schedule_meta = load_schedule_meta(PANEL_A_SEASON, PANEL_A_MATCH_ID, PANEL_A_TEAM)
    stadium_name, center_latlon, rotation, pitch_xy = calibrate_pitch_from_df(
        raw,
        load_pitch_registry(),
        preferred_stadium=schedule_meta.get("scheduled_stadium", ""),
    )
    t0 = pd.Timestamp(window["t0"])
    t1 = pd.Timestamp(window["t1"])
    raw_window = raw.loc[raw["timestamp"].between(t0, t1, inclusive="both")].copy()
    if raw_window.empty:
        return None

    xy = attach_xy_from_pitch(raw_window, center_latlon, rotation, stadium_name=stadium_name)
    if xy.empty:
        return None
    labeled = label_active_players(
        xy,
        pitch_xy=pitch_xy,
        active_depth_m=ACTIVE_DEPTH_M,
        activate_s=ACTIVATE_S,
        bench_off_s=BENCH_OFF_S,
        active_method=ACTIVE_PLAYER_METHOD,
        max_active_players=MAX_ACTIVE_PLAYERS,
        keep_debug_cols=True,
    )
    labeled["t"] = pd.to_datetime(labeled["timestamp"])
    labeled["match_id"] = PANEL_A_MATCH_ID
    labeled["match_phase"] = PANEL_A_MATCH_PHASE
    labeled["team"] = PANEL_A_TEAM
    labeled["source_key"] = PANEL_A_SOURCE_KEY
    labeled["track_type"] = "player_state"
    labeled = labeled[(labeled["t"] >= t0) & (labeled["t"] <= t1)].copy()
    if labeled.empty:
        return None

    active = labeled[labeled["player_status"].astype(str).eq("active")].copy()
    active_counts = active.groupby("t", observed=True)["player_name"].nunique()
    if active_counts.empty or int(active_counts.min()) != int(MAX_ACTIVE_PLAYERS):
        return None

    centroid = (
        active.groupby("t", observed=True)[["x_m", "y_m"]]
        .mean()
        .reset_index()
        .sort_values("t")
    )
    centroid["track_type"] = "centroid"
    centroid["player_name"] = "__centroid__"
    centroid["match_id"] = PANEL_A_MATCH_ID
    centroid["match_phase"] = PANEL_A_MATCH_PHASE
    centroid["team"] = PANEL_A_TEAM
    centroid["source_key"] = PANEL_A_SOURCE_KEY

    if len(centroid) >= 2:
        window["centroid_displacement_m"] = float(
            np.hypot(
                centroid.iloc[-1]["x_m"] - centroid.iloc[0]["x_m"],
                centroid.iloc[-1]["y_m"] - centroid.iloc[0]["y_m"],
            )
        )
        window["selection_score"] = float(window["centroid_displacement_m"] + 0.15 * int(MAX_ACTIVE_PLAYERS))

    start_rows = labeled[labeled["t"].eq(t0)]
    status_counts = start_rows["player_status"].astype(str).value_counts().to_dict()
    window["n_players"] = int(MAX_ACTIVE_PLAYERS)
    window["n_players_plotted"] = int(MAX_ACTIVE_PLAYERS)
    window["n_active_state_players"] = int(MAX_ACTIVE_PLAYERS)
    window["n_bench_state_players"] = int(status_counts.get("bench", 0))
    window["n_rejected_hoverer_players"] = int(status_counts.get("rejected_hoverer", 0))
    window["active_player_method"] = str(ACTIVE_PLAYER_METHOD)
    window["panel_A_state_source"] = str(raw_path)

    meta_info = {
        "stadium_name": stadium_name,
        "home": schedule_meta.get("home", ""),
        "away": schedule_meta.get("away", ""),
        "score": schedule_meta.get("score", ""),
    }
    return labeled, centroid, window, meta_info


def load_real_match_window() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    traj_path = DATA_DIR / TRAJECTORY_FILE
    audit_path = OUT_DIR / "real_match_20s_player_centroid_trajectories_audit.csv"
    meta_path = DATA_DIR / META_FILE

    traj = pd.read_parquet(
        traj_path,
        columns=["match_id", "match_phase", "team", "source_key", "track_type", "player_name", "t", "x_m", "y_m"],
    )
    traj = traj[traj["track_type"].isin(["player_abs", "centroid"])].copy()
    traj["t"] = pd.to_datetime(traj["t"])

    use_cached_window = False
    if audit_path.exists() and use_cached_window:
        row = pd.read_csv(audit_path).iloc[0].to_dict()
        window = {
            "match_id": str(row["match_id"]),
            "match_phase": str(row["match_phase"]),
            "team": str(row["team"]),
            "source_key": str(row["source_key"]),
            "t0": pd.Timestamp(row["t0"]),
            "t1": pd.Timestamp(row["t1"]),
            "n_players": int(row.get("n_players_plotted", row.get("n_players", 0))),
            "centroid_displacement_m": float(row.get("centroid_displacement_m", np.nan)),
            "selection_score": float(row.get("selection_score", np.nan)),
        }
    else:
        preferred = traj[
            traj["match_id"].astype(str).eq(PANEL_A_MATCH_ID)
            & traj["match_phase"].astype(str).eq(PANEL_A_MATCH_PHASE)
            & traj["team"].astype(str).eq(PANEL_A_TEAM)
            & traj["source_key"].astype(str).eq(PANEL_A_SOURCE_KEY)
        ].copy()
        if len(preferred) and PANEL_A_T0 and PANEL_A_T1:
            window = {
                "match_id": PANEL_A_MATCH_ID,
                "match_phase": PANEL_A_MATCH_PHASE,
                "team": PANEL_A_TEAM,
                "source_key": PANEL_A_SOURCE_KEY,
                "t0": pd.Timestamp(PANEL_A_T0),
                "t1": pd.Timestamp(PANEL_A_T1),
                "n_players": 0,
                "centroid_displacement_m": np.nan,
                "selection_score": np.nan,
            }
        else:
            window = select_window_if_needed(preferred if len(preferred) else traj)

    mask = (
        traj["match_id"].astype(str).eq(window["match_id"])
        & traj["match_phase"].astype(str).eq(window["match_phase"])
        & traj["team"].astype(str).eq(window["team"])
        & traj["source_key"].astype(str).eq(window["source_key"])
        & (traj["t"] >= window["t0"])
        & (traj["t"] <= window["t1"])
    )
    sub = traj.loc[mask].copy()
    player_sub = sub[sub["track_type"].eq("player_abs")].copy()
    counts = player_sub.groupby("player_name")["t"].nunique()
    keep_players = counts[counts >= WINDOW_DURATION_S - 1].index.tolist()
    player_sub = player_sub[player_sub["player_name"].isin(keep_players)].copy()
    centroid_sub = sub[sub["track_type"].eq("centroid")].sort_values("t").copy()
    if len(centroid_sub) >= 2:
        window["centroid_displacement_m"] = float(
            np.hypot(
                centroid_sub.iloc[-1]["x_m"] - centroid_sub.iloc[0]["x_m"],
                centroid_sub.iloc[-1]["y_m"] - centroid_sub.iloc[0]["y_m"],
            )
        )
        window["selection_score"] = float(window["centroid_displacement_m"] + 0.15 * len(keep_players))
    window["n_players"] = int(len(keep_players))

    meta_info = {"stadium_name": "Koteng Arena", "home": "", "away": "", "score": ""}
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        m = meta[
            meta["match_id"].astype(str).eq(window["match_id"])
            & meta["source_key"].astype(str).eq(window["source_key"])
            & meta["team"].astype(str).eq(window["team"])
        ]
        if len(m):
            row = m.iloc[0]
            meta_info = {
                "stadium_name": str(row.get("stadium_name", "Koteng Arena")),
                "home": str(row.get("home", "")),
                "away": str(row.get("away", "")),
                "score": str(row.get("score", "")),
            }
    window["n_players_plotted"] = int(len(keep_players))

    status_window = load_panel_a_status_window(window)
    if status_window is not None:
        player_sub, centroid_sub, window, meta_info = status_window

    return player_sub, centroid_sub, traj, window, meta_info


def draw_real_match_panel(ax, player_sub: pd.DataFrame, centroid_sub: pd.DataFrame, window: dict, meta: dict, pitch_xy) -> None:
    setup_active_bench_pitch(ax, pitch_xy, scale_bar=False, margin_m=PANEL_A_PITCH_MARGIN_M)
    xmin, xmax, ymin, ymax = pitch_bounds(pitch_xy)
    has_status_state = "player_status" in player_sub.columns
    if has_status_state:
        active_sub = player_sub[player_sub["player_status"].astype(str).eq("active")].copy()
        inactive_sub = player_sub[~player_sub["player_status"].astype(str).eq("active")].copy()
        n_bench_markers = int(inactive_sub["player_name"].nunique())
    else:
        active_sub = player_sub.copy()
        inactive_sub = player_sub.iloc[0:0].copy()
        n_bench_markers = draw_bench_excluded_markers(ax, pitch_xy)

    players = sorted(active_sub["player_name"].dropna().unique().tolist())
    player_colors = {player: PANEL_A_ACTIVE_TRAJECTORY_COLOR for player in players}

    if len(inactive_sub):
        for player, pg in inactive_sub.groupby("player_name", sort=False):
            pg = pg.sort_values("t")
            status = str(pg.iloc[-1].get("player_status", "bench"))
            color = COLORS["bench"] if status == "bench" else COLORS["dark_grey"]
            if len(pg) >= 2:
                ax.plot(pg["x_m"], pg["y_m"], color=color, lw=0.55, alpha=0.34, zorder=3)
            ax.scatter(
                pg.iloc[-1]["x_m"],
                pg.iloc[-1]["y_m"],
                marker="x",
                s=24,
                color=COLORS["black"],
                linewidths=1.05,
                alpha=0.95,
                zorder=12,
            )

    for player, pg in active_sub.groupby("player_name", sort=False):
        pg = pg.sort_values("t")
        color = player_colors[player]
        ax.plot(pg["x_m"], pg["y_m"], color=color, lw=PANEL_A_ACTIVE_TRAJECTORY_LW, alpha=PANEL_A_ACTIVE_TRAJECTORY_ALPHA, zorder=4)
        ax.scatter(pg.iloc[0]["x_m"], pg.iloc[0]["y_m"], s=12, facecolors="white", edgecolors=color, lw=0.65, zorder=5)
        ax.scatter(pg.iloc[-1]["x_m"], pg.iloc[-1]["y_m"], s=18, facecolors=COLORS["black"], edgecolors="white", lw=0.25, zorder=6)
        if len(pg) >= 2:
            p0 = pg.iloc[-2]
            p1 = pg.iloc[-1]
            arrow(ax, (p0["x_m"], p0["y_m"]), (p1["x_m"], p1["y_m"]), color=color, lw=0.72, ms=6, z=6)

    ax.plot(centroid_sub["x_m"], centroid_sub["y_m"], color=COLORS["centroid"], lw=2.45, zorder=8)
    if len(centroid_sub) >= 2:
        c0 = centroid_sub.iloc[-2]
        c1 = centroid_sub.iloc[-1]
        arrow(ax, (c0["x_m"], c0["y_m"]), (c1["x_m"], c1["y_m"]), color=COLORS["centroid"], lw=1.95, ms=13, z=9)
    ax.scatter(centroid_sub.iloc[0]["x_m"], centroid_sub.iloc[0]["y_m"], s=50, facecolors="white", edgecolors=COLORS["centroid"], lw=1.15, zorder=9)
    ax.scatter(centroid_sub.iloc[-1]["x_m"], centroid_sub.iloc[-1]["y_m"], s=58, color=COLORS["centroid"], edgecolors=COLORS["black"], linewidths=0.45, zorder=10)

    match_label = f"{ascii_safe(meta.get('home', ''))} vs {ascii_safe(meta.get('away', ''))}".strip(" vs")
    title_bits = [match_label or f"{ascii_safe(window['team']).title()} match", str(window["match_id"]), str(window["match_phase"]), ascii_safe(meta.get("score", "")).strip()]
    if SHOW_FIGURE_TITLES:
        ax.set_title(" | ".join([bit for bit in title_bits if bit]), fontsize=9.8, pad=3)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["black"],
            lw=0.95,
            markerfacecolor=COLORS["black"],
            markeredgecolor="white",
            markersize=4,
            label=fig_text("panel_a_player_trajectories_label"),
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["centroid"],
            lw=2.0,
            markerfacecolor=COLORS["centroid"],
            markeredgecolor=COLORS["black"],
            markersize=5,
            label=fig_text("panel_a_centroid_trajectory_label"),
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            ls="none",
            color=COLORS["black"],
            markersize=4.5,
            markeredgewidth=1.0,
            label=fig_text("panel_a_bench_players_label"),
        ),
    ]
    legend_kwargs = {
        "handles": handles,
        "loc": PANEL_A_LEGEND_LOC,
        "frameon": False,
        "fontsize": PANEL_A_LEGEND_FONTSIZE,
        "borderpad": 0.1,
        "borderaxespad": 0.25,
        "labelspacing": 0.15,
        "handlelength": 1.25,
        "handletextpad": 0.75,
    }
    if PANEL_A_LEGEND_ANCHOR is not None:
        legend_kwargs["bbox_to_anchor"] = PANEL_A_LEGEND_ANCHOR
    ax.legend(**legend_kwargs)
    window["n_bench_markers"] = n_bench_markers
    draw_panel_label(ax, "A")


def dashed_link(ax, c, p) -> None:
    ax.plot([c[0], p[0]], [c[1], p[1]], ls=(0, (3, 3)), lw=0.75, color=COLORS["black"], alpha=0.78, zorder=4)


def draw_players(ax, centroid, players, heading_vectors, *, high: bool) -> None:
    bends_hi = [0.13, -0.10, 0.11, 0.08, -0.10, 0.12, -0.08, 0.10, -0.11]
    bends_lo = [0.32, -0.25, 0.28, -0.35, 0.22, -0.30, 0.33, -0.18, 0.24]
    bends = bends_hi if high else bends_lo
    for idx, (pos, vec) in enumerate(zip(players, heading_vectors)):
        dashed_link(ax, centroid, pos)
        ax.scatter(*pos, s=25, color=COLORS["black"], zorder=7)
        start = (pos[0] - 0.36 * vec[0], pos[1] - 0.36 * vec[1])
        end = (pos[0] + vec[0], pos[1] + vec[1])
        curved_arrow(
            ax,
            start,
            end,
            color=COLORS["grey"],
            lw=1.05,
            ms=8,
            bend=bends[idx % len(bends)],
            alpha=0.92,
            z=6,
        )
    ax.scatter(*centroid, s=55, color=COLORS["red"], zorder=9)


def red_centroid_path(ax, points, *, curved: bool = False) -> None:
    if not curved:
        ax.plot([p[0] for p in points], [p[1] for p in points], color=COLORS["red"], lw=1.9, zorder=8)
        arrow(ax, points[-2], points[-1], color=COLORS["red"], lw=1.9, ms=13, z=9)
        return
    path = MplPath([points[0], points[1], points[2], points[3]], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, color=COLORS["red"], lw=1.9, zorder=8))
    tangent = np.asarray(points[3]) - np.asarray(points[2])
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    end = np.asarray(points[3])
    arrow(ax, tuple(end - 4.0 * tangent), tuple(end), color=COLORS["red"], lw=1.9, ms=13, z=9)


def draw_polarisation_panel(ax, pitch_xy, *, high: bool, label: str) -> None:
    setup_animation_pitch(ax, pitch_xy, scale_bar=False)
    xmin, xmax, ymin, ymax = pitch_bounds(pitch_xy)
    w = xmax - xmin
    h = ymax - ymin
    c = (xmin + 0.37 * w, ymin + 0.50 * h)
    players = [
        (xmin + 0.20 * w, ymin + 0.70 * h),
        (xmin + 0.15 * w, ymin + 0.56 * h),
        (xmin + 0.16 * w, ymin + 0.38 * h),
        (xmin + 0.31 * w, ymin + 0.79 * h),
        (xmin + 0.32 * w, ymin + 0.29 * h),
        (xmin + 0.49 * w, ymin + 0.79 * h),
        (xmin + 0.49 * w, ymin + 0.29 * h),
        (xmin + 0.58 * w, ymin + 0.56 * h),
        (xmin + 0.58 * w, ymin + 0.40 * h),
    ]
    hi_vec = [(0.10 * w, 0.0), (0.08 * w, 0.0), (0.08 * w, 0.02 * h), (0.09 * w, 0.0), (0.09 * w, 0.0), (0.09 * w, 0.0), (0.09 * w, 0.0), (0.09 * w, 0.0), (0.09 * w, 0.0)]
    lo_vec = [
        (-0.08 * w, 0.12 * h),
        (-0.09 * w, 0.04 * h),
        (-0.08 * w, -0.12 * h),
        (0.07 * w, 0.13 * h),
        (-0.06 * w, -0.14 * h),
        (0.07 * w, 0.13 * h),
        (0.07 * w, -0.13 * h),
        (0.10 * w, 0.00 * h),
        (0.09 * w, -0.02 * h),
    ]
    draw_players(ax, c, players, hi_vec if high else lo_vec, high=high)
    if high:
        ax.add_patch(Ellipse((xmin + 0.62 * w, ymin + 0.49 * h), 0.22 * w, 0.24 * h, fill=False, ec=COLORS["dark_grey"], lw=1.0, zorder=5))
        red_centroid_path(ax, [c, (xmin + 0.74 * w, ymin + 0.51 * h), (xmin + 0.86 * w, ymin + 0.51 * h)])
        title = fig_text("legacy_higher_polarisation_title")
    else:
        ax.add_patch(Ellipse((xmin + 0.62 * w, ymin + 0.50 * h), 0.21 * w, 0.18 * h, angle=12, fill=False, ec=COLORS["dark_grey"], lw=1.0, zorder=5))
        red_centroid_path(ax, [c, (xmin + 0.55 * w, ymin + 0.52 * h), (xmin + 0.70 * w, ymin + 0.45 * h), (xmin + 0.86 * w, ymin + 0.50 * h)], curved=True)
        title = fig_text("legacy_lower_polarisation_title")
    ax.set_title(title, fontsize=9.5, style="italic", pad=4)
    draw_panel_label(ax, label)


def bezier_run(ax, p0, c1, c2, p1, color, label, label_pos) -> None:
    path = MplPath([p0, c1, c2, p1], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, color=color, lw=2.0, capstyle="round", zorder=6))
    tangent = np.asarray(p1) - np.asarray(c2)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    end = np.asarray(p1)
    arrow(ax, tuple(end - 4.3 * tangent), tuple(end), color=color, lw=2.0, ms=12, z=7)
    ax.text(*label_pos, label, color=color, fontsize=8.8, ha="center", va="center", zorder=8)


def draw_segmentation_panel(ax, pitch_xy) -> None:
    setup_animation_pitch(ax, pitch_xy, scale_bar=False)
    xmin, xmax, ymin, ymax = pitch_bounds(pitch_xy)
    w = xmax - xmin
    h = ymax - ymin
    p0 = (xmin + 0.18 * w, ymin + 0.75 * h)
    p1 = (xmin + 0.47 * w, ymin + 0.63 * h)
    p2 = (xmin + 0.54 * w, ymin + 0.31 * h)
    p3 = (xmin + 0.72 * w, ymin + 0.19 * h)
    p4 = (xmin + 0.86 * w, ymin + 0.37 * h)
    p5 = (xmin + 0.93 * w, ymin + 0.13 * h)

    bezier_run(ax, p0, (xmin + 0.25 * w, ymin + 0.68 * h), (xmin + 0.38 * w, ymin + 0.70 * h), p1, COLORS["blue"], "run 1", (xmin + 0.33 * w, ymin + 0.72 * h))
    bezier_run(ax, p1, (xmin + 0.59 * w, ymin + 0.55 * h), (xmin + 0.60 * w, ymin + 0.42 * h), p2, COLORS["green"], "run 2", (xmin + 0.60 * w, ymin + 0.49 * h))
    bezier_run(ax, p2, (xmin + 0.38 * w, ymin + 0.25 * h), (xmin + 0.36 * w, ymin + 0.15 * h), p3, COLORS["purple"], "run 3", (xmin + 0.43 * w, ymin + 0.26 * h))
    bezier_run(ax, p3, (xmin + 0.66 * w, ymin + 0.08 * h), (xmin + 0.76 * w, ymin + 0.13 * h), p4, COLORS["orange"], "run 4", (xmin + 0.75 * w, ymin + 0.20 * h))
    bezier_run(ax, p4, (xmin + 0.92 * w, ymin + 0.30 * h), (xmin + 0.91 * w, ymin + 0.20 * h), p5, COLORS["teal"], "run 5", (xmin + 0.94 * w, ymin + 0.26 * h))

    for p in [p0, p1, p2, p3, p4, p5]:
        ax.scatter(*p, s=35, color=COLORS["black"], zorder=9)
    ax.text(p0[0] - 0.02 * w, p0[1] + 0.08 * h, fig_text("panel_b_start_label"), ha="center", va="bottom", fontsize=9.6)
    ax.text(p5[0] + 0.02 * w, p5[1] - 0.06 * h, fig_text("panel_b_end_label"), ha="center", va="top", fontsize=9.6)
    extension = (p1[0] + 0.16 * w, p1[1] + 0.25 * h)
    ax.plot([p1[0], extension[0]], [p1[1], extension[1]], color=COLORS["black"], lw=0.9, ls=(0, (5, 4)), zorder=5)
    ax.add_patch(Arc(p1, 0.20 * w, 0.24 * h, angle=0, theta1=18, theta2=80, lw=0.9, color=COLORS["black"], zorder=6))
    ax.text(
        p1[0] + 0.14 * w,
        p1[1] + 0.03 * h,
        fig_text("panel_b_angle_label_template", r"$\Delta\theta>\theta_c$", angle=TURN_THRESHOLD_DEG),
        fontsize=10.0,
        ha="left",
        va="center",
    )
    ax.set_title(fig_text("legacy_run_segmentation_title"), fontsize=10.5, pad=4)
    draw_panel_label(ax, "B")


def row_key(row) -> dict:
    return {
        "match_id": str(row["match_id"]),
        "match_phase": str(row["match_phase"]),
        "team": str(row["team"]),
        "source_key": str(row["source_key"]),
    }


def key_mask(df: pd.DataFrame, key: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, value in key.items():
        if col in df.columns:
            mask &= df[col].astype(str).eq(str(value))
    return mask


def vector_angle_deg(v0, v1) -> float:
    a = np.asarray(v0, dtype=float)
    b = np.asarray(v1, dtype=float)
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return np.nan
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def segment_direction(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([np.nan, np.nan], dtype=float)
    return np.asarray(points[-1], dtype=float) - np.asarray(points[0], dtype=float)


def local_tangent(points: np.ndarray, *, at_end: bool, samples: int = LOCAL_TANGENT_SAMPLES) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.array([np.nan, np.nan], dtype=float)
    step = min(max(int(samples), 1), len(pts) - 1)
    if at_end:
        return pts[-1] - pts[-1 - step]
    return pts[step] - pts[0]


def select_segmentation_sequence(
    runs: pd.DataFrame,
    traj: pd.DataFrame,
    preferred: dict,
    preferred_t0: str | None = None,
    preferred_t1: str | None = None,
) -> pd.DataFrame:
    d = runs[runs["track_type"].astype(str).eq("centroid")].copy()
    d["t_start"] = pd.to_datetime(d["t_start"])
    d["t_end"] = pd.to_datetime(d["t_end"])
    d["duration_s"] = pd.to_numeric(d["duration_s"], errors="coerce")
    d["run_length_m"] = pd.to_numeric(d["run_length_m"], errors="coerce")

    if preferred_t0 and preferred_t1:
        t0 = pd.Timestamp(preferred_t0)
        t1 = pd.Timestamp(preferred_t1)
        exact = d.loc[
            key_mask(d, preferred)
            & d["t_start"].ge(t0)
            & d["t_end"].le(t1)
            & d["duration_s"].ge(SEGMENT_MIN_DURATION_S)
        ].sort_values("t_start")
        if len(exact) in SEGMENT_SEQUENCE_RUNS:
            return exact.reset_index(drop=True)

    preferred_rows = d.loc[key_mask(d, preferred)].copy()
    pools = [preferred_rows, d] if len(preferred_rows) else [d]
    best_score = -np.inf
    best = None
    cent_all = traj[traj["track_type"].astype(str).eq("centroid")].copy()
    cent_all["t"] = pd.to_datetime(cent_all["t"])

    for pool in pools:
        for _, g in pool.groupby(["match_id", "match_phase", "team", "source_key"], sort=False):
            g = g.sort_values("t_start").reset_index(drop=True)
            group_key = row_key(g.iloc[0])
            cent = cent_all.loc[key_mask(cent_all, group_key)].sort_values("t")
            for n_runs in SEGMENT_SEQUENCE_RUNS:
                if len(g) < n_runs:
                    continue
                for start in range(0, len(g) - n_runs + 1):
                    seq = g.iloc[start : start + n_runs].copy()
                    duration = float((seq["t_end"].iloc[-1] - seq["t_start"].iloc[0]).total_seconds())
                    if not (28.0 <= duration <= 70.0):
                        continue
                    if (seq["duration_s"] < SEGMENT_MIN_DURATION_S).any():
                        continue
                    total_length = float(seq["run_length_m"].sum())
                    if total_length < 35.0:
                        continue

                    pts_all = []
                    end_tangents = []
                    start_tangents = []
                    for _, row in seq.iterrows():
                        pts = cent[(cent["t"] >= pd.Timestamp(row["t_start"])) & (cent["t"] <= pd.Timestamp(row["t_end"]))][["x_m", "y_m"]].to_numpy(float)
                        if len(pts) < 2:
                            break
                        pts_all.append(pts)
                        start_tangents.append(local_tangent(pts, at_end=False))
                        end_tangents.append(local_tangent(pts, at_end=True))
                    if len(pts_all) != len(seq):
                        continue
                    turn_angles = [vector_angle_deg(end_tangents[i], start_tangents[i + 1]) for i in range(len(end_tangents) - 1)]
                    if not turn_angles or np.nanmax(turn_angles) < TURN_THRESHOLD_DEG:
                        continue
                    all_pts = np.vstack(pts_all)
                    footprint = float((all_pts[:, 0].max() - all_pts[:, 0].min()) * (all_pts[:, 1].max() - all_pts[:, 1].min()))
                    footprint_penalty = 0.03 * footprint
                    angle_bonus = 0.20 * float(np.nanmax(turn_angles))
                    short_penalty = float(np.maximum(0.0, 8.0 - seq["duration_s"]).sum())
                    score = total_length + 0.30 * duration - 4.0 * short_penalty - footprint_penalty + angle_bonus
                    if score > best_score:
                        best_score = score
                        best = seq
        if best is not None:
            return best.reset_index(drop=True)

    raise RuntimeError("Could not find a real centroid sequence for segmentation panel.")


def draw_angle_marker(ax, point, v0, v1, *, pitch_w: float, pitch_h: float) -> None:
    if not np.all(np.isfinite([*point, *v0, *v1])):
        return
    angle = vector_angle_deg(v0, v1)
    if not np.isfinite(angle):
        return

    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v0 = v0 / (np.linalg.norm(v0) + 1e-9)
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    p = np.asarray(point, dtype=float)

    radius = 0.085 * min(pitch_w, pitch_h)
    extension_end = p + radius * 1.9 * v0
    new_dir_end = p + radius * 1.45 * v1
    ax.plot([p[0], extension_end[0]], [p[1], extension_end[1]], color=COLORS["black"], lw=0.9, ls=(0, (5, 4)), zorder=8)
    ax.plot([p[0], new_dir_end[0]], [p[1], new_dir_end[1]], color=COLORS["black"], lw=0.7, alpha=0.55, zorder=8)

    a0 = float(np.degrees(np.arctan2(v0[1], v0[0])))
    a1 = float(np.degrees(np.arctan2(v1[1], v1[0])))
    delta = (a1 - a0 + 180.0) % 360.0 - 180.0
    theta_a = a0
    theta_b = a0 + delta
    if theta_b < theta_a:
        theta_a, theta_b = theta_b, theta_a
    ax.add_patch(Arc(p, 2 * radius, 2 * radius, angle=0, theta1=theta_a, theta2=theta_b, lw=0.9, color=COLORS["black"], zorder=9))

    bisector = v0 + v1
    if np.linalg.norm(bisector) < 1e-9:
        bisector = np.asarray([-v0[1], v0[0]])
    bisector = bisector / (np.linalg.norm(bisector) + 1e-9)
    label_pos = p + radius * PANEL_B_ANGLE_LABEL_RADIUS_SCALE * bisector + np.asarray(
        [PANEL_B_ANGLE_LABEL_OFFSET[0] * pitch_w, PANEL_B_ANGLE_LABEL_OFFSET[1] * pitch_h]
    )
    ax.text(
        label_pos[0],
        label_pos[1],
        fig_text("panel_b_angle_label_template", angle=angle),
        fontsize=9.2,
        ha="left",
        va="center",
        transform=ax.transData + ScaledTranslation(-0.5 / 2.54, 0.0, ax.figure.dpi_scale_trans),
        zorder=12,
    )


def draw_real_segmentation_panel(ax, pitch_xy, traj: pd.DataFrame, sequence: pd.DataFrame) -> dict:
    setup_neutral_pitch(ax, pitch_xy, scale_bar=False, margin_m=PANEL_B_PITCH_MARGIN_M)
    xmin, xmax, ymin, ymax = pitch_bounds(pitch_xy)
    w = xmax - xmin
    h = ymax - ymin
    key = row_key(sequence.iloc[0])
    cent = traj.loc[key_mask(traj, key) & traj["track_type"].astype(str).eq("centroid")].copy()
    cent["t"] = pd.to_datetime(cent["t"])
    cent = cent.sort_values("t")

    colors = [COLORS["blue"], COLORS["vermillion"], COLORS["green"], COLORS["magenta"]]
    segment_records = []
    turn_records = []
    start_tangents = []
    end_tangents = []
    boundary_points = []

    for idx, run in sequence.reset_index(drop=True).iterrows():
        t0 = pd.Timestamp(run["t_start"])
        t1 = pd.Timestamp(run["t_end"])
        seg = cent[(cent["t"] >= t0) & (cent["t"] <= t1)].copy()
        if len(seg) < 2:
            continue
        color = colors[idx % len(colors)]
        pts = seg[["x_m", "y_m"]].to_numpy(float)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.25, alpha=0.96, solid_capstyle="round", zorder=6)
        arrow(ax, tuple(pts[-2]), tuple(pts[-1]), color=color, lw=2.0, ms=12, z=8)

        mid = pts[len(pts) // 2]
        label_nudges = [np.asarray([dx * w, dy * h]) for dx, dy in PANEL_B_RUN_LABEL_NUDGES]
        if idx < len(label_nudges):
            mid = mid + label_nudges[idx]
        else:
            direction = segment_direction(pts)
            normal = np.asarray([-direction[1], direction[0]], dtype=float)
            if np.linalg.norm(normal) > 1e-9:
                normal = normal / np.linalg.norm(normal)
                mid = mid + normal * (0.040 * h * (1 if idx % 2 == 0 else -1))
        ax.text(
            mid[0],
            mid[1],
            fig_text("panel_b_run_label_template", n=idx + 1),
            color=COLORS["black"],
            fontsize=9.2,
            ha="center",
            va="center",
            zorder=14,
        )
        start_tangents.append(local_tangent(pts, at_end=False))
        end_tangents.append(local_tangent(pts, at_end=True))
        boundary_points.append(tuple(pts[-1]))
        segment_records.append(
            {
                "run_number": int(idx + 1),
                "run_uid": str(run["run_uid"]),
                "t_start": pd.Timestamp(run["t_start"]).isoformat(),
                "t_end": pd.Timestamp(run["t_end"]).isoformat(),
                "duration_s": float(run["duration_s"]),
                "run_length_m": float(run["run_length_m"]),
            }
        )

    if segment_records:
        first = cent[cent["t"].eq(pd.Timestamp(sequence.iloc[0]["t_start"]))]
        last = cent[cent["t"].eq(pd.Timestamp(sequence.iloc[-1]["t_end"]))]
        if len(first):
            ax.scatter(first.iloc[0]["x_m"], first.iloc[0]["y_m"], s=40, facecolors="white", edgecolors=COLORS["black"], lw=1.0, zorder=11)
            start_label_xy = (
                first.iloc[0]["x_m"] + PANEL_B_START_LABEL_OFFSET[0] * w,
                first.iloc[0]["y_m"] + PANEL_B_START_LABEL_OFFSET[1] * h,
            )
            ax.text(
                start_label_xy[0],
                start_label_xy[1],
                fig_text("panel_b_start_label"),
                fontsize=9.2,
                ha="center",
                va="top",
                zorder=14,
            )
        if len(last):
            ax.scatter(last.iloc[0]["x_m"], last.iloc[0]["y_m"], s=44, color=COLORS["black"], zorder=11)
            end_label_xy = (
                last.iloc[0]["x_m"] + PANEL_B_END_LABEL_OFFSET[0] * w,
                last.iloc[0]["y_m"] + PANEL_B_END_LABEL_OFFSET[1] * h,
            )
            ax.text(
                end_label_xy[0],
                end_label_xy[1],
                fig_text("panel_b_end_label"),
                fontsize=9.2,
                ha="center",
                va="top",
                zorder=14,
            )

    boundary_angles = []
    for idx in range(len(end_tangents) - 1):
        angle = vector_angle_deg(end_tangents[idx], start_tangents[idx + 1])
        boundary_angles.append(angle)
        if np.isfinite(angle):
            turn_records.append(
                {
                    "boundary_after_run": int(idx + 1),
                    "turn_angle_deg": float(angle),
                    "angle_definition": "local end tangent of previous run vs local start tangent of next run",
                }
            )

    for point in boundary_points[:-1]:
        ax.scatter(*point, s=32, color=COLORS["black"], zorder=10)

    if len(boundary_angles):
        valid = [(i, a) for i, a in enumerate(boundary_angles) if np.isfinite(a)]
        if valid:
            marker_idx, _ = max(valid, key=lambda item: item[1])
            draw_angle_marker(ax, boundary_points[marker_idx], end_tangents[marker_idx], start_tangents[marker_idx + 1], pitch_w=w, pitch_h=h)

    start = pd.Timestamp(sequence.iloc[0]["t_start"])
    end = pd.Timestamp(sequence.iloc[-1]["t_end"])
    if SHOW_FIGURE_TITLES:
        ax.set_title(fig_text("panel_b_title"), fontsize=10.5, pad=4)
    draw_panel_label(ax, "B")

    return {
        **key,
        "t_start": start.isoformat(),
        "t_end": end.isoformat(),
        "n_runs_shown": int(len(segment_records)),
        "turn_threshold_deg": float(TURN_THRESHOLD_DEG),
        "angle_definition": "local end tangent of previous run vs local start tangent of next run",
        "local_tangent_samples": int(LOCAL_TANGENT_SAMPLES),
        "segments": segment_records,
        "turns": turn_records,
    }


def load_collective_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_parquet(DATA_DIR / RUNS_FILE)
    pmv = pd.read_parquet(DATA_DIR / PMV_FILE)
    return runs, pmv


def select_polarisation_snapshots(pmv: pd.DataFrame, traj: pd.DataFrame, preferred: dict) -> tuple[dict, dict]:
    pmv = pmv.copy()
    pmv["_t"] = pd.to_datetime(pmv["_t"])
    pmv["p_group"] = pd.to_numeric(pmv["p_group"], errors="coerce")
    pmv["n_players"] = pd.to_numeric(pmv["n_players"], errors="coerce")

    cent = traj[traj["track_type"].astype(str).eq("centroid")].copy()
    cent["t"] = pd.to_datetime(cent["t"])
    key_cols = ["match_id", "match_phase", "team", "source_key"]

    base = pmv[pmv["n_players"].ge(10) & np.isfinite(pmv["p_group"])].copy()
    base["future_t"] = base["_t"] + pd.Timedelta(seconds=SNAPSHOT_FUTURE_S)
    current = cent.rename(columns={"t": "_t", "x_m": "cx", "y_m": "cy"})[key_cols + ["_t", "cx", "cy"]]
    future = cent.rename(columns={"t": "future_t", "x_m": "fx", "y_m": "fy"})[key_cols + ["future_t", "fx", "fy"]]
    merged = base.merge(current, on=key_cols + ["_t"], how="inner").merge(future, on=key_cols + ["future_t"], how="inner")
    merged["future_disp_m"] = np.hypot(merged["fx"] - merged["cx"], merged["fy"] - merged["cy"])
    merged = merged[merged["future_disp_m"].ge(5.0)].copy()

    preferred_rows = merged.loc[key_mask(merged, preferred)].copy()
    pool = preferred_rows if len(preferred_rows) else merged
    if pool.empty:
        raise RuntimeError("Could not find high/low polarisation snapshots with future centroid displacement.")

    pool = pool.sort_values("_t").reset_index(drop=True)
    player_all = traj[traj["track_type"].astype(str).eq("player_abs")].copy()
    player_all["t"] = pd.to_datetime(player_all["t"])
    spacing_source = player_all.loc[key_mask(player_all, preferred)].copy() if len(preferred_rows) else player_all
    spacing_lookup = {}
    for group_key, players_now in spacing_source.groupby(key_cols + ["t"], sort=False):
        if len(players_now) < 10:
            continue
        pts = players_now[["x_m", "y_m"]].to_numpy(float)
        dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        dist[dist == 0] = np.nan
        nearest = np.nanmin(dist, axis=1)
        centroid = pts.mean(axis=0)
        radius = np.sqrt(((pts - centroid) ** 2).sum(axis=1))
        *group_values, t_value = group_key
        spacing_lookup[tuple(str(value) for value in group_values) + (pd.Timestamp(t_value),)] = (
            float(np.nanmedian(nearest)),
            float(np.nanmedian(radius)),
        )

    def formation_spacing(row) -> tuple[float, float]:
        key = tuple(str(row[col]) for col in key_cols) + (pd.Timestamp(row["_t"]),)
        return spacing_lookup.get(key, (0.0, 0.0))

    best_pair = None
    best_score = -np.inf
    for window_s in (70, 90, 110, 130):
        for _, row in pool.iterrows():
            t0 = pd.Timestamp(row["_t"])
            sub = pool[(pool["_t"] >= t0) & (pool["_t"] <= t0 + pd.Timedelta(seconds=window_s))]
            if len(sub) < 25:
                continue
            high_row = sub.loc[sub["p_group"].idxmax()]
            low_floor = float(sub["p_group"].min())
            low_candidates = sub[sub["p_group"].le(min(low_floor + 0.10, 0.35))]
            for _, low_row in low_candidates.iterrows():
                contrast = float(high_row["p_group"] - low_row["p_group"])
                gap_s = abs((pd.Timestamp(high_row["_t"]) - pd.Timestamp(low_row["_t"])).total_seconds())
                if contrast < 0.60 or gap_s < POLARISATION_MIN_GAP_S or gap_s > POLARISATION_MAX_GAP_S:
                    continue
                low_nearest, low_radius = formation_spacing(low_row)
                score = (
                    contrast
                    + 0.050 * low_nearest
                    + 0.010 * low_radius
                    + 0.002 * min(float(high_row["future_disp_m"]), 20.0)
                    + 0.002 * min(float(low_row["future_disp_m"]), 20.0)
                    - 0.0010 * abs(float(gap_s) - 55.0)
                    - 0.0008 * window_s
                )
                if score > best_score:
                    best_score = score
                    best_pair = (high_row.to_dict(), low_row.to_dict())
    if best_pair is not None:
        return best_pair

    high = pool.sort_values(["p_group", "future_disp_m"], ascending=[False, False]).iloc[0].to_dict()
    low = pool.sort_values(["p_group", "future_disp_m"], ascending=[True, False]).iloc[0].to_dict()
    return high, low


def time_slice(df: pd.DataFrame, t: pd.Timestamp, *, tolerance_s: float = 0.75) -> pd.DataFrame:
    exact = df[df["t"].eq(t)]
    if len(exact):
        return exact.copy()
    delta = (df["t"] - t).abs().dt.total_seconds()
    if delta.empty or float(delta.min()) > tolerance_s:
        return df.iloc[0:0].copy()
    nearest_t = df.loc[delta.idxmin(), "t"]
    return df[df["t"].eq(nearest_t)].copy()


def draw_polarisation_inset(ax, pmv: pd.DataFrame, snapshot: dict, *, color: str, bounds: list[float]) -> None:
    key = row_key(snapshot)
    t0 = pd.Timestamp(snapshot["_t"])
    prof = pmv.loc[key_mask(pmv, key)].copy()
    prof["_t"] = pd.to_datetime(prof["_t"])
    prof = prof[(prof["_t"] >= t0 - pd.Timedelta(seconds=POLARISATION_PROFILE_S)) & (prof["_t"] <= t0 + pd.Timedelta(seconds=POLARISATION_PROFILE_S))].sort_values("_t")
    if prof.empty:
        return

    ins = ax.inset_axes(bounds)
    x = (prof["_t"] - t0).dt.total_seconds().to_numpy(float)
    y = pd.to_numeric(prof["p_group"], errors="coerce").to_numpy(float)
    ins.plot(x, y, color=color, lw=1.15)
    ins.axvline(0, color=COLORS["red"], lw=0.9)
    ins.scatter([0], [float(snapshot["p_group"])], s=18, color=COLORS["red"], zorder=5)
    ins.set_xlim(-POLARISATION_PROFILE_S, POLARISATION_PROFILE_S)
    ins.set_ylim(0.0, 1.0)
    ins.set_xticks([-POLARISATION_PROFILE_S, 0, POLARISATION_PROFILE_S])
    ins.set_yticks([0.0, 0.5, 1.0])
    ins.tick_params(labelsize=6.2, length=2, pad=1)
    ins.set_xlabel(fig_text("panel_c_profile_inset_xlabel"), fontsize=7.3, labelpad=0)
    ins.set_ylabel(fig_text("panel_c_profile_inset_ylabel"), fontsize=7.3, labelpad=0)
    ins.set_title(fig_text("panel_c_profile_inset_title"), fontsize=7.8, pad=1.2)
    ins.set_facecolor((1, 1, 1, 0.88))
    for spine in ins.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#777777")


def draw_real_polarisation_panel(ax, pitch_xy, traj: pd.DataFrame, pmv: pd.DataFrame, snapshot: dict, *, high: bool, label: str) -> dict:
    setup_animation_pitch(ax, pitch_xy, scale_bar=False)
    xmin, xmax, ymin, ymax = pitch_bounds(pitch_xy)
    h = ymax - ymin
    key = row_key(snapshot)
    t0 = pd.Timestamp(snapshot["_t"])
    future_t = pd.Timestamp(snapshot["future_t"])
    sub = traj.loc[key_mask(traj, key)].copy()
    sub["t"] = pd.to_datetime(sub["t"])

    players_now = time_slice(sub[sub["track_type"].astype(str).eq("player_abs")], t0)
    cent = sub[sub["track_type"].astype(str).eq("centroid")].sort_values("t")
    cent_window = cent[(cent["t"] >= t0) & (cent["t"] <= future_t)].copy()
    current_cent = time_slice(cent, t0)
    future_cent = time_slice(cent, future_t)

    trace_start = t0 - pd.Timedelta(seconds=SNAPSHOT_TRAIL_S)
    player_traces = sub[(sub["track_type"].astype(str).eq("player_abs")) & (sub["t"] >= trace_start) & (sub["t"] <= t0)].copy()
    for player, pg in player_traces.groupby("player_name", sort=False):
        pg = pg.sort_values("t")
        if len(pg) >= 2:
            ax.plot(pg["x_m"], pg["y_m"], color=COLORS["grey"], lw=0.75, alpha=0.62, zorder=4)
            arrow(ax, (pg.iloc[-2]["x_m"], pg.iloc[-2]["y_m"]), (pg.iloc[-1]["x_m"], pg.iloc[-1]["y_m"]), color=COLORS["grey"], lw=0.75, ms=6, z=5)

    if len(current_cent):
        c = (float(current_cent.iloc[0]["x_m"]), float(current_cent.iloc[0]["y_m"]))
        for _, row in players_now.iterrows():
            dashed_link(ax, c, (float(row["x_m"]), float(row["y_m"])))

    if len(players_now):
        ax.scatter(players_now["x_m"], players_now["y_m"], s=27, color=COLORS["black"], edgecolors="white", linewidths=0.25, zorder=7)

    if len(cent_window) >= 2:
        pts = cent_window[["x_m", "y_m"]].to_numpy(float)
        ax.plot(pts[:, 0], pts[:, 1], color=COLORS["red"], lw=2.2, zorder=8)
        arrow(ax, tuple(pts[-2]), tuple(pts[-1]), color=COLORS["red"], lw=2.0, ms=14, z=9)
    elif len(current_cent) and len(future_cent):
        arrow(
            ax,
            (float(current_cent.iloc[0]["x_m"]), float(current_cent.iloc[0]["y_m"])),
            (float(future_cent.iloc[0]["x_m"]), float(future_cent.iloc[0]["y_m"])),
            color=COLORS["red"],
            lw=2.0,
            ms=14,
            z=9,
        )

    if len(current_cent):
        ax.scatter(current_cent.iloc[0]["x_m"], current_cent.iloc[0]["y_m"], s=58, color=COLORS["red"], edgecolors=COLORS["black"], linewidths=0.65, zorder=10)

    title = fig_text("legacy_higher_polarisation_title") if high else fig_text("legacy_lower_polarisation_title")
    ax.set_title(f"{title} | p={float(snapshot['p_group']):.3f}", fontsize=10.5, style="italic", pad=4)
    inset_bounds = [0.575, 0.055, 0.36, 0.285] if high else [0.075, 0.055, 0.36, 0.285]
    time_y = 0.92 if high else 0.045
    ax.text(
        0.985,
        time_y,
        f"{t0.strftime('%H:%M:%S')} -> +{SNAPSHOT_FUTURE_S}s centroid",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=1.4),
        zorder=25,
    )
    draw_polarisation_inset(ax, pmv, snapshot, color=COLORS["blue"] if high else COLORS["purple"], bounds=inset_bounds)
    draw_panel_label(ax, label)

    return {
        **key,
        "t": t0.isoformat(),
        "future_t": future_t.isoformat(),
        "p_group": float(snapshot["p_group"]),
        "n_players": int(snapshot["n_players"]),
        "future_disp_m": float(snapshot["future_disp_m"]),
        "players_plotted": int(len(players_now)),
    }


def draw_snapshot_pitch_inset(ax, pitch_xy, traj: pd.DataFrame, snapshot: dict, *, title: str, accent: str, compact: bool = False) -> dict:
    setup_neutral_pitch(ax, pitch_xy, scale_bar=False, margin_m=1.0)
    key = row_key(snapshot)
    t0 = pd.Timestamp(snapshot["_t"])
    future_t = pd.Timestamp(snapshot["future_t"])
    sub = traj.loc[key_mask(traj, key)].copy()
    sub["t"] = pd.to_datetime(sub["t"])

    players_now = time_slice(sub[sub["track_type"].astype(str).eq("player_abs")], t0)
    cent = sub[sub["track_type"].astype(str).eq("centroid")].sort_values("t")
    cent_window = cent[(cent["t"] >= t0) & (cent["t"] <= future_t)].copy()
    current_cent = time_slice(cent, t0)

    trace_start = t0 - pd.Timedelta(seconds=SNAPSHOT_TRAIL_S)
    traces = sub[(sub["track_type"].astype(str).eq("player_abs")) & (sub["t"] >= trace_start) & (sub["t"] <= t0)]
    player_arrows = []
    for _, pg in traces.groupby("player_name", sort=False):
        pg = pg.sort_values("t")
        if len(pg) >= 2:
            ax.plot(pg["x_m"], pg["y_m"], color="#9da3a6", lw=0.62, alpha=0.42, zorder=4)
            p_now = pg.iloc[-1][["x_m", "y_m"]].to_numpy(float)
            p_then = pg.iloc[0][["x_m", "y_m"]].to_numpy(float)
            direction = p_now - p_then
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 0.20:
                unit = direction / direction_norm
                arrow_start = p_now + 0.35 * unit
                arrow_end = p_now + 8.6 * unit
            else:
                arrow_start = pg.iloc[-2][["x_m", "y_m"]].to_numpy(float)
                arrow_end = p_now
            player_arrows.append((tuple(arrow_start), tuple(arrow_end)))

    if len(current_cent):
        c = (float(current_cent.iloc[0]["x_m"]), float(current_cent.iloc[0]["y_m"]))
        for _, row in players_now.iterrows():
            ax.plot([c[0], row["x_m"]], [c[1], row["y_m"]], ls=(0, (2.0, 2.4)), lw=0.42, color="#303030", alpha=0.24, zorder=5)

    if len(players_now):
        ax.scatter(players_now["x_m"], players_now["y_m"], s=28, color=COLORS["black"], edgecolors="white", linewidths=0.34, zorder=7)

    for start, end in player_arrows:
        arrow(ax, start, end, color="white", lw=2.7, ms=15, z=8)
        arrow(ax, start, end, color="#252a2d", lw=1.25, ms=12, z=9)

    if len(cent_window) >= 2:
        pts = cent_window[["x_m", "y_m"]].to_numpy(float)
        ax.plot(pts[:, 0], pts[:, 1], color="white", lw=4.7, solid_capstyle="round", zorder=10)
        ax.plot(pts[:, 0], pts[:, 1], color=accent, lw=2.7, solid_capstyle="round", zorder=11)
        arrow(ax, tuple(pts[-2]), tuple(pts[-1]), color="white", lw=4.8, ms=21, z=12)
        arrow(ax, tuple(pts[-2]), tuple(pts[-1]), color=accent, lw=2.65, ms=17, z=13)

    if len(current_cent):
        ax.scatter(
            current_cent.iloc[0]["x_m"],
            current_cent.iloc[0]["y_m"],
            s=64,
            facecolors=accent,
            edgecolors=COLORS["black"],
            linewidths=0.65,
            zorder=14,
        )

    if not compact:
        ax.set_title(title, fontsize=8.2, pad=1.8)
        ax.text(
            0.98,
            0.04,
            f"{t0.strftime('%H:%M:%S')} | p={float(snapshot['p_group']):.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.9),
            zorder=25,
        )

    return {
        **key,
        "t": t0.isoformat(),
        "future_t": future_t.isoformat(),
        "p_group": float(snapshot["p_group"]),
        "n_players": int(snapshot["n_players"]),
        "future_disp_m": float(snapshot["future_disp_m"]),
        "players_plotted": int(len(players_now)),
    }


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) <= 1:
        return pts
    pts = np.unique(np.round(pts, 6), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def molded_outline_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return pts
    hull = convex_hull(pts)
    if len(hull) < 3:
        return hull

    centre = hull.mean(axis=0)
    shaped = []
    for i, p0 in enumerate(hull):
        p1 = hull[(i + 1) % len(hull)]

        radial = p0 - centre
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1e-6:
            unit = np.asarray([1.0, 0.0])
        else:
            unit = radial / radial_norm
        angle = float(np.arctan2(unit[1], unit[0]))
        vertex_pad = FORMATION_OUTLINE_PAD_M * (
            1.00
            + 0.12 * np.sin(2.8 * angle + 0.35)
            + 0.07 * np.cos(5.3 * angle - 0.70)
            + 0.04 * np.sin(1.7 * i + 0.20)
        )
        shaped.append(p0 + vertex_pad * unit)

        edge_mid = 0.5 * (p0 + p1)
        edge_radial = edge_mid - centre
        edge_radial_norm = float(np.linalg.norm(edge_radial))
        if edge_radial_norm < 1e-6:
            edge_unit = unit
        else:
            edge_unit = edge_radial / edge_radial_norm
        edge = p1 - p0
        edge_norm = float(np.linalg.norm(edge))
        tangent = edge / edge_norm if edge_norm > 1e-6 else np.asarray([0.0, 1.0])
        mid_pad = FORMATION_OUTLINE_PAD_M * (0.82 + 0.16 * np.cos(1.9 * i + 0.55))
        wiggle = FORMATION_OUTLINE_PAD_M * 0.10 * np.sin(2.4 * i + 0.15)
        shaped.append(edge_mid + mid_pad * edge_unit + wiggle * tangent)
    return np.asarray(shaped, dtype=float)


def spread_arrow_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    min_sep: float | None = None,
    iterations: int | None = None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if not segments:
        return []
    if min_sep is None:
        min_sep = FORMATION_PLAYER_ARROW_MIN_SEP_M
    if iterations is None:
        iterations = FORMATION_PLAYER_ARROW_SPREAD_ITER
    starts = np.asarray([s for s, _ in segments], dtype=float)
    ends = np.asarray([e for _, e in segments], dtype=float)
    if len(starts) < 2:
        return [(tuple(starts[0]), tuple(ends[0]))]

    sample_alpha = np.linspace(0.00, 1.00, 7)
    for _ in range(max(0, int(iterations))):
        shifts = np.zeros_like(starts)
        samples = starts[:, None, :] + (ends - starts)[:, None, :] * sample_alpha[None, :, None]
        for i in range(len(starts) - 1):
            for j in range(i + 1, len(starts)):
                diffs = samples[i, :, None, :] - samples[j, None, :, :]
                d2 = np.sum(diffs * diffs, axis=2)
                idx = np.unravel_index(int(np.argmin(d2)), d2.shape)
                dist = float(np.sqrt(d2[idx]))
                if dist >= min_sep:
                    continue
                direction = diffs[idx]
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm < 1e-6:
                    angle = 2.399963229728653 * (i + 1) + 0.9185586535436918 * (j + 1)
                    direction = np.asarray([np.cos(angle), np.sin(angle)])
                else:
                    direction = direction / direction_norm
                step = 0.5 * (min_sep - dist) * direction
                shifts[i] += step
                shifts[j] -= step
        max_shift = float(np.nanmax(np.linalg.norm(shifts, axis=1)))
        starts += 0.75 * shifts
        ends += 0.75 * shifts
        if max_shift < 0.05:
            break

    return [(tuple(starts[i]), tuple(ends[i])) for i in range(len(starts))]


def nudge_segment_away_from_segments(
    segment: tuple[np.ndarray, np.ndarray],
    blockers: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    min_sep: float | None = None,
    iterations: int | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not blockers:
        return tuple(segment[0]), tuple(segment[1])
    if min_sep is None:
        min_sep = FORMATION_CENTROID_PLAYER_MIN_SEP_M
    if iterations is None:
        iterations = max(8, FORMATION_PLAYER_ARROW_SPREAD_ITER // 2)

    start = np.asarray(segment[0], dtype=float).copy()
    end = np.asarray(segment[1], dtype=float).copy()
    block_starts = np.asarray([s for s, _ in blockers], dtype=float)
    block_ends = np.asarray([e for _, e in blockers], dtype=float)
    sample_alpha = np.linspace(0.00, 1.00, 7)
    axis = end - start
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return tuple(start), tuple(end)
    normal = np.asarray([-axis[1], axis[0]], dtype=float) / axis_norm

    for _ in range(max(0, int(iterations))):
        ref_samples = start[None, :] + (end - start)[None, :] * sample_alpha[:, None]
        blocker_samples = block_starts[:, None, :] + (block_ends - block_starts)[:, None, :] * sample_alpha[None, :, None]
        shift_scalar = 0.0
        for i in range(len(blockers)):
            diffs = ref_samples[:, None, :] - blocker_samples[i, None, :, :]
            d2 = np.sum(diffs * diffs, axis=2)
            idx = np.unravel_index(int(np.argmin(d2)), d2.shape)
            dist = float(np.sqrt(d2[idx]))
            if dist >= min_sep:
                continue
            direction = diffs[idx]
            sign = float(np.sign(np.dot(direction, normal)))
            if sign == 0.0:
                ref_mid = 0.5 * (start + end)
                block_mid = 0.5 * (block_starts[i] + block_ends[i])
                sign = float(np.sign(np.dot(ref_mid - block_mid, normal))) or 1.0
            shift_scalar += sign * (min_sep - dist)
        if abs(shift_scalar) < 0.05:
            break
        shift = 0.70 * shift_scalar * normal
        start += shift
        end += shift
    return tuple(start), tuple(end)


def closed_catmull_rom_curve(points: np.ndarray, samples_per_edge: int = 18) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return pts

    samples_per_edge = max(4, int(samples_per_edge))
    samples = []
    t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
    t2 = t * t
    t3 = t2 * t
    for i in range(len(pts)):
        p0 = pts[(i - 1) % len(pts)]
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        p3 = pts[(i + 2) % len(pts)]
        segment = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3[:, None]
        )
        samples.append(segment)
    return np.vstack(samples)


def draw_molded_inset_outline(ax, outline_points: np.ndarray) -> np.ndarray:
    expanded = np.asarray(outline_points, dtype=float)
    expanded = expanded[np.isfinite(expanded).all(axis=1)]
    if len(expanded) < 3:
        return expanded
    outline_curve = closed_catmull_rom_curve(expanded)
    verts = [tuple(outline_curve[0])]
    codes = [MplPath.MOVETO]
    verts.extend([tuple(p) for p in outline_curve[1:]])
    codes.extend([MplPath.LINETO] * (len(outline_curve) - 1))
    verts.append(tuple(outline_curve[0]))
    codes.append(MplPath.CLOSEPOLY)
    path = MplPath(verts, codes)
    ax.add_patch(
        PathPatch(
            path,
            fill=True,
            fc=(1.0, 1.0, 1.0, 0.94),
            lw=0,
            ec="none",
            capstyle="round",
            joinstyle="round",
            zorder=5,
            clip_on=False,
        )
    )
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            ec=COLORS["black"],
            lw=1.02,
            capstyle="round",
            joinstyle="round",
            zorder=30,
            clip_on=False,
        )
    )
    return outline_curve


def optional_string(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return str(value)


def optional_anchor(value) -> tuple[float, float] | tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"", "none", "auto"}:
            return None
        raise ValueError("Legend anchors must be None or a tuple/list of axes coordinates.")
    anchor = tuple(float(v) for v in value)
    if len(anchor) not in {2, 4}:
        raise ValueError("Legend anchors must have length 2 or 4.")
    return anchor


def apply_figure1_text(text: dict | None = None) -> None:
    if text:
        FIGURE_TEXT.update({str(key): value for key, value in text.items()})


def apply_figure1_selection(panel_a: dict, panel_b: dict, panel_c: dict | None = None) -> None:
    global PANEL_A_SEASON, PANEL_A_MATCH_ID, PANEL_A_MATCH_PHASE, PANEL_A_TEAM, PANEL_A_SOURCE_KEY, PANEL_A_T0, PANEL_A_T1
    global PANEL_B_MATCH_ID, PANEL_B_MATCH_PHASE, PANEL_B_TEAM, PANEL_B_SOURCE_KEY, PANEL_B_T0, PANEL_B_T1
    global PANEL_C_MATCH_ID, PANEL_C_MATCH_PHASE, PANEL_C_TEAM, PANEL_C_SOURCE_KEY

    PANEL_A_SEASON = str(panel_a.get("season", ""))
    PANEL_A_MATCH_ID = str(panel_a["match_id"])
    PANEL_A_MATCH_PHASE = str(panel_a["match_phase"])
    PANEL_A_TEAM = str(panel_a["team"])
    PANEL_A_SOURCE_KEY = str(panel_a.get("source_key", "A"))
    PANEL_A_T0 = optional_string(panel_a.get("t0"))
    PANEL_A_T1 = optional_string(panel_a.get("t1"))

    PANEL_B_MATCH_ID = str(panel_b["match_id"])
    PANEL_B_MATCH_PHASE = str(panel_b["match_phase"])
    PANEL_B_TEAM = str(panel_b["team"])
    PANEL_B_SOURCE_KEY = str(panel_b.get("source_key", "A"))
    PANEL_B_T0 = optional_string(panel_b.get("t0"))
    PANEL_B_T1 = optional_string(panel_b.get("t1"))

    panel_c = panel_c or {"use_panel_a_game": True}
    if panel_c.get("use_panel_a_game", True):
        PANEL_C_MATCH_ID = None
        PANEL_C_MATCH_PHASE = None
        PANEL_C_TEAM = None
        PANEL_C_SOURCE_KEY = None
    else:
        PANEL_C_MATCH_ID = optional_string(panel_c.get("match_id"))
        PANEL_C_MATCH_PHASE = optional_string(panel_c.get("match_phase"))
        PANEL_C_TEAM = optional_string(panel_c.get("team"))
        PANEL_C_SOURCE_KEY = optional_string(panel_c.get("source_key"))


def apply_figure1_style(style: dict) -> None:
    global PAPER_FONT_BASE, FIGURE_SIZE, FIGURE_SUBPLOT_ADJUST, FIGURE_GRID_HSPACE, FIGURE_GRID_WSPACE, FIGURE_TOP_PANEL_GAP_ADJUST, PANEL_C_GRID_WSPACE
    global PANEL_LABEL_XY, PANEL_LABEL_FONTSIZE
    global PANEL_A_PITCH_MARGIN_M, PANEL_A_ACTIVE_TRAJECTORY_COLOR, PANEL_A_ACTIVE_TRAJECTORY_LW, PANEL_A_ACTIVE_TRAJECTORY_ALPHA
    global PANEL_A_LEGEND_LOC, PANEL_A_LEGEND_ANCHOR, PANEL_A_LEGEND_FONTSIZE
    global PANEL_B_PITCH_MARGIN_M, PANEL_B_PITCH_FILL_COLOR, PANEL_B_OUTER_FILL_COLOR, PANEL_B_OUTER_FILL_ALPHA, PANEL_B_RUN_LABEL_NUDGES
    global PANEL_B_ANGLE_LABEL_RADIUS_SCALE, PANEL_B_ANGLE_LABEL_OFFSET, PANEL_B_START_LABEL_OFFSET, PANEL_B_END_LABEL_OFFSET
    global PANEL_C_INSET_BOUNDS, PANEL_C_LEGEND_LOC, PANEL_C_LEGEND_ANCHOR, PANEL_C_LEGEND_FONTSIZE, PANEL_C_LEGEND_NCOL
    global SHOW_FIGURE_TITLES, SHOW_PANEL_C_SNAPSHOT_LABELS, SHOW_PANEL_C_SNAPSHOT_SPANS
    global FORMATION_INSET_EXPAND, FORMATION_ARROW_START_M, FORMATION_ARROW_END_M, FORMATION_CENTROID_ARROW_LENGTH_M, FORMATION_INSET_VIEW_SPAN_M
    global FORMATION_OUTLINE_PAD_M, FORMATION_CENTROID_PLAYER_MIN_SEP_M, FORMATION_PLAYER_ARROW_MIN_SEP_M, FORMATION_PLAYER_ARROW_SPREAD_ITER
    global PANEL_C_CONNECTOR_LW, PANEL_C_CONNECTOR_ALPHA, PANEL_C_CONNECTOR_ANCHOR_SPREAD, SNAPSHOT_FUTURE_S, SNAPSHOT_TRAIL_S
    global POLARISATION_CONTEXT_S, POLARISATION_ROLLING_WINDOW_S, POLARISATION_MIN_GAP_S, POLARISATION_MAX_GAP_S

    apply_figure1_text(style.get("text"))
    COLORS.update(style.get("colors", {}))
    PAPER_FONT_BASE = float(style["paper_font_base"])
    FIGURE_SIZE = tuple(style["figure_size"])
    FIGURE_SUBPLOT_ADJUST = dict(style["figure_subplot_adjust"])
    FIGURE_GRID_HSPACE = float(style["figure_grid_hspace"])
    FIGURE_GRID_WSPACE = float(style["figure_grid_wspace"])
    FIGURE_TOP_PANEL_GAP_ADJUST = float(style["top_panel_gap_adjust"])
    PANEL_C_GRID_WSPACE = float(style["panel_c_grid_wspace"])
    PANEL_LABEL_XY = tuple(style["panel_label_xy"])
    PANEL_LABEL_FONTSIZE = float(style["panel_label_fontsize"])

    PANEL_A_PITCH_MARGIN_M = float(style["panel_a_pitch_margin_m"])
    PANEL_A_ACTIVE_TRAJECTORY_COLOR = str(style["panel_a_active_trajectory_color"])
    PANEL_A_ACTIVE_TRAJECTORY_LW = float(style["panel_a_active_trajectory_lw"])
    PANEL_A_ACTIVE_TRAJECTORY_ALPHA = float(style["panel_a_active_trajectory_alpha"])
    PANEL_A_LEGEND_LOC = str(style["panel_a_legend_loc"])
    PANEL_A_LEGEND_ANCHOR = optional_anchor(style.get("panel_a_legend_anchor"))
    PANEL_A_LEGEND_FONTSIZE = float(style["panel_a_legend_fontsize"])

    PANEL_B_PITCH_MARGIN_M = float(style.get("panel_b_pitch_margin_m", PANEL_B_PITCH_MARGIN_M))
    PANEL_B_PITCH_FILL_COLOR = str(style["panel_b_pitch_fill_color"])
    PANEL_B_OUTER_FILL_COLOR = str(style["panel_b_outer_fill_color"])
    PANEL_B_OUTER_FILL_ALPHA = float(style["panel_b_outer_fill_alpha"])
    PANEL_B_RUN_LABEL_NUDGES = [tuple(v) for v in style["panel_b_run_label_nudges"]]
    PANEL_B_ANGLE_LABEL_RADIUS_SCALE = float(style["panel_b_angle_label_radius_scale"])
    PANEL_B_ANGLE_LABEL_OFFSET = tuple(style["panel_b_angle_label_offset"])
    PANEL_B_START_LABEL_OFFSET = tuple(style.get("panel_b_start_label_offset", PANEL_B_START_LABEL_OFFSET))
    PANEL_B_END_LABEL_OFFSET = tuple(style.get("panel_b_end_label_offset", PANEL_B_END_LABEL_OFFSET))

    PANEL_C_INSET_BOUNDS = {k: list(v) for k, v in style["panel_c_inset_bounds"].items()}
    PANEL_C_LEGEND_LOC = str(style["panel_c_legend_loc"])
    PANEL_C_LEGEND_ANCHOR = optional_anchor(style.get("panel_c_legend_anchor"))
    PANEL_C_LEGEND_FONTSIZE = float(style["panel_c_legend_fontsize"])
    PANEL_C_LEGEND_NCOL = int(style.get("panel_c_legend_ncol", PANEL_C_LEGEND_NCOL))
    SHOW_FIGURE_TITLES = bool(style["show_figure_titles"])
    SHOW_PANEL_C_SNAPSHOT_LABELS = bool(style["show_panel_c_snapshot_labels"])
    SHOW_PANEL_C_SNAPSHOT_SPANS = bool(style["show_panel_c_snapshot_spans"])

    FORMATION_INSET_EXPAND = float(style["formation_inset_expand"])
    FORMATION_ARROW_START_M = float(style["formation_player_arrow_start_m"])
    FORMATION_ARROW_END_M = float(style["formation_player_arrow_end_m"])
    FORMATION_CENTROID_ARROW_LENGTH_M = float(style["formation_centroid_arrow_length_m"])
    FORMATION_INSET_VIEW_SPAN_M = float(style["formation_inset_view_span_m"])
    FORMATION_OUTLINE_PAD_M = float(style["formation_outline_pad_m"])
    FORMATION_CENTROID_PLAYER_MIN_SEP_M = float(style["formation_centroid_player_min_sep_m"])
    FORMATION_PLAYER_ARROW_MIN_SEP_M = float(style["formation_player_arrow_min_sep_m"])
    FORMATION_PLAYER_ARROW_SPREAD_ITER = int(style["formation_player_arrow_spread_iter"])

    PANEL_C_CONNECTOR_LW = float(style["panel_c_connector_lw"])
    PANEL_C_CONNECTOR_ALPHA = float(style["panel_c_connector_alpha"])
    PANEL_C_CONNECTOR_ANCHOR_SPREAD = float(style.get("panel_c_connector_anchor_spread", PANEL_C_CONNECTOR_ANCHOR_SPREAD))
    SNAPSHOT_FUTURE_S = int(style["snapshot_future_s"])
    SNAPSHOT_TRAIL_S = int(style["snapshot_trail_s"])
    POLARISATION_CONTEXT_S = int(style["polarisation_context_s"])
    POLARISATION_ROLLING_WINDOW_S = int(style["polarisation_rolling_window_s"])
    POLARISATION_MIN_GAP_S = int(style["polarisation_min_gap_s"])
    POLARISATION_MAX_GAP_S = int(style["polarisation_max_gap_s"])


def plot_figure1_transport_mechanism(
    panel_a: dict,
    panel_b: dict,
    panel_c: dict | None,
    style: dict,
    *,
    text: dict | None = None,
    outdir: Path | str | None = None,
    close_figure: bool = True,
) -> dict:
    global OUT_DIR, SHARE_DIR
    apply_figure1_selection(panel_a, panel_b, panel_c)
    apply_figure1_style(style)
    apply_figure1_text(text)
    if outdir is not None:
        OUT_DIR = Path(outdir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARE_DIR = ROOT / "analysis" / "levy_paper" / "chatgpt_share"
    SHARE_DIR.mkdir(parents=True, exist_ok=True)

    fig = main(close_figure=close_figure)
    png = OUT_DIR / "real_match_transport_mechanism_4panel.png"
    pdf = OUT_DIR / "real_match_transport_mechanism_4panel.pdf"
    audit_path = OUT_DIR / "real_match_transport_mechanism_4panel_audit.json"
    caption_path = OUT_DIR / "real_match_transport_mechanism_4panel_caption.txt"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return {
        "figure": fig,
        "png": png,
        "pdf": pdf,
        "caption_path": caption_path,
        "audit_path": audit_path,
        "audit": audit,
    }


def load_trajectory_for_browser() -> pd.DataFrame:
    traj = pd.read_parquet(
        DATA_DIR / TRAJECTORY_FILE,
        columns=["match_id", "match_phase", "team", "source_key", "track_type", "player_name", "t", "x_m", "y_m"],
    )
    traj["t"] = pd.to_datetime(traj["t"])
    return traj


def available_games(limit: int = 30) -> pd.DataFrame:
    traj = load_trajectory_for_browser()
    cent = traj.loc[traj["track_type"].eq("centroid")].copy()
    games = (
        cent.groupby(["match_id", "match_phase", "team", "source_key"], as_index=False)
        .agg(t_start=("t", "min"), t_end=("t", "max"), n_frames=("t", "nunique"))
        .sort_values(["match_id", "match_phase", "team", "source_key"])
        .reset_index(drop=True)
    )
    return games.head(limit)


def panel_a_candidates(match_id=None, match_phase=None, team=None, source_key=None, n: int = 12) -> pd.DataFrame:
    traj = load_trajectory_for_browser()
    mask = pd.Series(True, index=traj.index)
    for col, value in {"match_id": match_id, "match_phase": match_phase, "team": team, "source_key": source_key}.items():
        if value is not None:
            mask &= traj[col].astype(str).eq(str(value))
    traj = traj.loc[mask].copy()

    records = []
    duration_s = int(WINDOW_DURATION_S)
    for keys, group in traj.groupby(["match_id", "match_phase", "team", "source_key"], sort=False):
        cent = group.loc[group["track_type"].eq("centroid")].drop_duplicates("t").sort_values("t").reset_index(drop=True)
        players = group.loc[group["track_type"].eq("player_abs")].copy()
        if len(cent) <= duration_s or players.empty:
            continue
        local_windows = []
        for i in range(0, len(cent) - duration_s):
            t0 = cent.loc[i, "t"]
            t1 = cent.loc[i + duration_s, "t"]
            if (t1 - t0).total_seconds() != duration_s:
                continue
            disp = float(np.hypot(cent.loc[i + duration_s, "x_m"] - cent.loc[i, "x_m"], cent.loc[i + duration_s, "y_m"] - cent.loc[i, "y_m"]))
            local_windows.append((disp, pd.Timestamp(t0), pd.Timestamp(t1)))
        for disp, t0, t1 in sorted(local_windows, reverse=True)[:160]:
            pwin = players.loc[(players["t"] >= t0) & (players["t"] <= t1)]
            counts = pwin.groupby("player_name")["t"].nunique()
            n_players = int((counts >= duration_s - 1).sum())
            if n_players < 10:
                continue
            match_id_i, phase_i, team_i, source_i = [str(v) for v in keys]
            records.append(
                {
                    "match_id": match_id_i,
                    "match_phase": phase_i,
                    "team": team_i,
                    "source_key": source_i,
                    "t0": t0.isoformat(),
                    "t1": t1.isoformat(),
                    "n_players": n_players,
                    "centroid_displacement_m": disp,
                    "selection_score": disp + 0.15 * n_players,
                }
            )
            break

    return pd.DataFrame(records).sort_values("selection_score", ascending=False).head(n).reset_index(drop=True)


def panel_a_from_candidate(candidates: pd.DataFrame, row: int = 0) -> dict:
    rec = candidates.iloc[int(row)].to_dict()
    return {
        "match_id": rec["match_id"],
        "match_phase": rec["match_phase"],
        "team": rec["team"],
        "source_key": rec["source_key"],
        "t0": rec["t0"],
        "t1": rec["t1"],
    }


def connect_profile_marker_to_inset(ax, ins, marker_xy: tuple[float, float], accent: str) -> None:
    outline = getattr(ins, "_fig1_outline_shape", None)
    if outline is None:
        return
    outline = np.asarray(outline, dtype=float)
    outline = outline[np.isfinite(outline).all(axis=1)]
    if len(outline) < 3:
        return
    marker_display = ax.transData.transform(marker_xy)
    marker_as_inset_data = ins.transData.inverted().transform(marker_display)
    centre = outline.mean(axis=0)
    direction = marker_as_inset_data - centre
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-6:
        nearest = outline[int(np.argmin(np.sum((outline - marker_as_inset_data) ** 2, axis=1)))]
        direction = nearest - centre
        direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-6:
        return
    span = float(np.nanmax(np.ptp(outline, axis=0))) if len(outline) else 0.0
    target_spread = float(np.clip(PANEL_C_CONNECTOR_ANCHOR_SPREAD, 0.08, 0.48))
    min_anchor_sep = max(10.0, target_spread * span)
    idx = int(np.argmin(np.sum((outline - marker_as_inset_data) ** 2, axis=1)))
    n_outline = len(outline)
    anchors = [outline[idx], outline[idx]]
    offset_fracs = [
        target_spread * 0.28,
        target_spread * 0.36,
        target_spread * 0.46,
        target_spread * 0.58,
        target_spread * 0.72,
        target_spread * 0.90,
    ]
    for offset_frac in offset_fracs:
        offset = max(8, int(round(offset_frac * n_outline)))
        pair = [outline[(idx - offset) % n_outline], outline[(idx + offset) % n_outline]]
        if np.linalg.norm(pair[0] - pair[1]) >= min_anchor_sep:
            anchors = pair
            break

    for anchor in anchors:
        connector = ConnectionPatch(
            xyA=marker_xy,
            xyB=tuple(anchor),
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ins,
            color=accent,
            lw=PANEL_C_CONNECTOR_LW,
            alpha=PANEL_C_CONNECTOR_ALPHA,
            linestyle=PANEL_C_CONNECTOR_DASH,
            zorder=48,
        )
        connector.set_clip_on(False)
        ax.figure.add_artist(connector)


def draw_snapshot_formation_inset(ax, traj: pd.DataFrame, snapshot: dict, *, title: str, accent: str) -> dict:
    key = row_key(snapshot)
    t0 = pd.Timestamp(snapshot["_t"])
    future_t = pd.Timestamp(snapshot["future_t"])
    sub = traj.loc[key_mask(traj, key)].copy()
    sub["t"] = pd.to_datetime(sub["t"])

    players_now = time_slice(sub[sub["track_type"].astype(str).eq("player_abs")], t0)
    cent = sub[sub["track_type"].astype(str).eq("centroid")].sort_values("t")
    cent_window = cent[(cent["t"] >= t0) & (cent["t"] <= future_t)].copy()
    current_cent = time_slice(cent, t0)
    if len(current_cent):
        origin = np.asarray([float(current_cent.iloc[0]["x_m"]), float(current_cent.iloc[0]["y_m"])])
    elif len(players_now):
        origin = players_now[["x_m", "y_m"]].median().to_numpy(float)
    else:
        origin = np.asarray([0.0, 0.0])

    rel_points = [np.asarray([[0.0, 0.0]])]
    outline_points = []
    player_arrows = []
    player_tracks = sub[sub["track_type"].astype(str).eq("player_abs")].copy()
    group_cols = [col for col in ("player_name", "path_uid") if col in player_tracks.columns]
    if not group_cols:
        group_cols = ["player_name"]
    for _, pg in player_tracks.groupby(group_cols, sort=False, dropna=False):
        pg = pg.sort_values("t")
        prev_rows = pg[pg["t"].le(t0)]
        if len(prev_rows) >= 2:
            current = prev_rows.iloc[-1]
            previous = prev_rows.iloc[-2]
            lag_s = (pd.Timestamp(current["t"]) - pd.Timestamp(previous["t"])).total_seconds()
            offset_s = abs((pd.Timestamp(current["t"]) - t0).total_seconds())
            if not (0.0 < lag_s <= 2.25 and offset_s <= 0.75):
                continue
            p_now = (current[["x_m", "y_m"]].to_numpy(float) - origin) * FORMATION_INSET_EXPAND
            p_then = (previous[["x_m", "y_m"]].to_numpy(float) - origin) * FORMATION_INSET_EXPAND
            direction = p_now - p_then
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 0.20:
                unit = direction / direction_norm
                arrow_start = p_now + FORMATION_ARROW_START_M * unit
                arrow_end = p_now + FORMATION_ARROW_END_M * unit
                player_arrows.append((arrow_start, arrow_end))

    player_arrows = spread_arrow_segments(player_arrows)
    for start, end in player_arrows:
        rel_points.append(np.vstack([start, end]))
        outline_points.append(np.vstack([start, end]))
    for start, end in player_arrows:
        arrow(ax, start, end, color="white", lw=2.6, ms=22, z=8)
    for start, end in player_arrows:
        arrow(ax, start, end, color="#171b1e", lw=0.95, ms=16, z=9)

    if len(cent_window) >= 2:
        pts = (cent_window[["x_m", "y_m"]].to_numpy(float) - origin) * FORMATION_INSET_EXPAND
        direction = pts[-1] - pts[0]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 0.20:
            unit = direction / direction_norm
            start_arr = pts[0]
            end_arr = start_arr + FORMATION_CENTROID_ARROW_LENGTH_M * unit
            start, end = nudge_segment_away_from_segments((start_arr, end_arr), player_arrows)
            start_arr = np.asarray(start, dtype=float)
            end_arr = np.asarray(end, dtype=float)
            rel_points.append(np.vstack([start_arr, end_arr]))
            outline_points.append(np.vstack([start_arr, end_arr]))
            arrow(ax, start, end, color="white", lw=5.8, ms=29, z=18)
            arrow(ax, start, end, color=accent, lw=3.05, ms=22, z=19)

    all_pts = np.vstack(rel_points)
    outline_source = np.vstack(outline_points) if outline_points else all_pts
    outline_shape = molded_outline_points(outline_source)
    outline_curve = closed_catmull_rom_curve(outline_shape) if len(outline_shape) else outline_shape
    if len(outline_shape):
        all_pts = np.vstack([all_pts, outline_shape])
    if len(outline_curve):
        all_pts = np.vstack([all_pts, outline_curve])
    xmin, ymin = np.nanmin(all_pts, axis=0)
    xmax, ymax = np.nanmax(all_pts, axis=0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = FORMATION_INSET_VIEW_SPAN_M
    pad = max(2.0, 0.030 * span)
    ax.set_xlim(cx - 0.5 * span - pad, cx + 0.5 * span + pad)
    ax.set_ylim(cy - 0.5 * span - pad, cy + 0.5 * span + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_visible(False)
    drawn_outline = draw_molded_inset_outline(ax, outline_shape)
    ax._fig1_outline_shape = drawn_outline
    return {
        **key,
        "t": t0.isoformat(),
        "future_t": future_t.isoformat(),
        "p_group": float(snapshot["p_group"]),
        "n_players": int(snapshot["n_players"]),
        "future_disp_m": float(snapshot["future_disp_m"]),
        "players_plotted": int(len(players_now)),
    }


def draw_polarisation_profile_panel(
    ax,
    pitch_xy,
    traj: pd.DataFrame,
    pmv: pd.DataFrame,
    high_snapshot: dict,
    low_snapshot: dict,
    snapshot_axes: dict[str, plt.Axes] | None = None,
) -> dict:
    key = row_key(high_snapshot)
    prof = pmv.loc[key_mask(pmv, key)].copy()
    prof["_t"] = pd.to_datetime(prof["_t"])
    prof["p_group"] = pd.to_numeric(prof["p_group"], errors="coerce")
    prof = prof.dropna(subset=["_t", "p_group"]).sort_values("_t")

    high_t = pd.Timestamp(high_snapshot["_t"])
    low_t = pd.Timestamp(low_snapshot["_t"])
    start_t = max(prof["_t"].min(), min(high_t, low_t) - pd.Timedelta(seconds=POLARISATION_CONTEXT_S))
    end_t = min(prof["_t"].max(), max(high_t, low_t) + pd.Timedelta(seconds=POLARISATION_CONTEXT_S))
    prof = prof[(prof["_t"] >= start_t) & (prof["_t"] <= end_t)].copy()

    x_s = (prof["_t"] - start_t).dt.total_seconds().to_numpy(float)
    p_raw = prof["p_group"].to_numpy(float)
    rolling = pd.Series(p_raw)
    p_lo = rolling.rolling(POLARISATION_ROLLING_WINDOW_S, center=True, min_periods=3).quantile(0.25).bfill().ffill().to_numpy(float)
    p_hi = rolling.rolling(POLARISATION_ROLLING_WINDOW_S, center=True, min_periods=3).quantile(0.75).bfill().ffill().to_numpy(float)

    ax.fill_between(
        x_s,
        p_lo,
        p_hi,
        color=COLORS["slate"],
        alpha=0.23,
        lw=0,
        label=fig_text("panel_c_iqr_label_template", window_s=POLARISATION_ROLLING_WINDOW_S),
    )
    ax.plot(x_s, p_raw, color=COLORS["black"], lw=1.35, label=fig_text("panel_c_raw_label"))
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlim(float(np.nanmin(x_s)), float(np.nanmax(x_s)))
    ax.set_xlabel(fig_text("panel_c_xlabel_template", time=start_t.strftime("%H:%M:%S")))
    ax.set_ylabel(fig_text("panel_c_ylabel"))
    if SHOW_FIGURE_TITLES:
        ax.set_title(fig_text("panel_c_title"), fontsize=8.8, pad=4)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.8, length=2.8)

    snapshot_audits = {}
    snapshot_markers = []
    embedded_insets = snapshot_axes is None
    if embedded_insets:
        embedded_side = {"low": "left", "high": "right"} if low_t <= high_t else {"high": "left", "low": "right"}
    else:
        embedded_side = {}
    low_accent = COLORS["blue"]
    high_accent = COLORS["red"]
    for snapshot, label, accent, label_y in [
        (low_snapshot, "low", low_accent, 0.18),
        (high_snapshot, "high", high_accent, 0.79),
    ]:
        t = pd.Timestamp(snapshot["_t"])
        x = (t - start_t).total_seconds()
        y = float(np.interp(x, x_s, p_raw))
        if SHOW_PANEL_C_SNAPSHOT_SPANS:
            ax.axvspan(x - 1.5, x + 1.5, color=accent, alpha=0.08, lw=0)
        ax.scatter([x], [y], s=38, color=accent, edgecolors=COLORS["black"], linewidths=0.50, zorder=8)
        snapshot_markers.append((label, snapshot, accent, x, y))
        text_x = x
        text_ha = "center"
        text_transform = ax.get_xaxis_transform()
        if embedded_insets:
            side = embedded_side[label]
            text_transform = ax.transAxes
            text_x = 0.31 if side == "left" else 0.69
            text_ha = "center"
            label_y = 0.605 if label == "low" else 0.84
        if SHOW_PANEL_C_SNAPSHOT_LABELS:
            ax.text(
                text_x,
                label_y,
                fig_text("panel_c_snapshot_label_template", state=label, p=y),
                transform=text_transform,
                ha=text_ha,
                va="center",
                fontsize=8.0,
                color=COLORS["black"],
                linespacing=0.88,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
                zorder=9,
            )

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    player_direction_handle = Line2D(
        [0, 1],
        [0, 0],
        color=COLORS["black"],
        lw=1.0,
        marker=">",
        markevery=[1],
        markersize=4.8,
    )
    low_centroid_handle = Line2D([0, 1], [0, 0], color=low_accent, lw=1.5, marker=">", markevery=[1], markersize=5.4)
    high_centroid_handle = Line2D([0, 1], [0, 0], color=high_accent, lw=1.5, marker=">", markevery=[1], markersize=5.4)
    legend_handles.extend([player_direction_handle, (low_centroid_handle, high_centroid_handle)])
    legend_labels.extend([fig_text("panel_c_player_direction_label"), fig_text("panel_c_centroid_direction_label")])
    legend_kwargs = {
        "handles": legend_handles,
        "labels": legend_labels,
        "loc": PANEL_C_LEGEND_LOC,
        "frameon": False,
        "ncol": PANEL_C_LEGEND_NCOL,
        "fontsize": PANEL_C_LEGEND_FONTSIZE,
        "handlelength": 1.35,
        "borderaxespad": 0.2,
        "columnspacing": 0.9,
        "labelspacing": 0.22,
        "handler_map": {tuple: HandlerTuple(ndivide=None, pad=0.18)},
    }
    if PANEL_C_LEGEND_ANCHOR is not None:
        legend_kwargs["bbox_to_anchor"] = PANEL_C_LEGEND_ANCHOR
    ax.legend(**legend_kwargs)
    draw_panel_label(ax, "C")

    x_lo, x_hi = ax.get_xlim()
    for label, snapshot, accent, x, y in snapshot_markers:
        if snapshot_axes and label in snapshot_axes:
            ins = snapshot_axes[label]
        else:
            if embedded_insets:
                x0, y0, width, height = PANEL_C_INSET_BOUNDS[label]
            else:
                width = 0.220
                height = 0.410
                x_frac = (x - x_lo) / (x_hi - x_lo + 1e-9)
                x0 = float(np.clip(x_frac - width / 2.0, 0.045, 1.0 - width - 0.045))
                if y >= 0.65:
                    y0 = float(np.clip(y - height - 0.15, 0.08, 1.0 - height - 0.045))
                else:
                    y0 = float(np.clip(y + 0.12, 0.08, 1.0 - height - 0.045))
            ins = ax.inset_axes([x0, y0, width, height])
            ins.set_zorder(40)
            snapshot_audits[label] = draw_snapshot_formation_inset(
                ins,
                traj,
                snapshot,
                title=fig_text("panel_c_inset_title_template", state=label),
                accent=accent,
            )
            connect_profile_marker_to_inset(ax, ins, (float(x), float(y)), accent)
            continue
        snapshot_audits[label] = draw_snapshot_pitch_inset(
            ins,
            pitch_xy,
            traj,
            snapshot,
            title=fig_text("panel_c_inset_title_template", state=label),
            accent=accent,
            compact=True,
        )

    return {
        **key,
        "profile_start": start_t.isoformat(),
        "profile_end": end_t.isoformat(),
        "n_profile_samples": int(len(prof)),
        "snapshots": snapshot_audits,
    }


def build_figure_caption() -> str:
    lines = []
    main_caption = fig_text("figure_caption").strip()
    if main_caption:
        lines.append(main_caption)
    for letter_key, caption_key in [
        ("panel_a_letter", "panel_a_caption"),
        ("panel_b_letter", "panel_b_caption"),
        ("panel_c_letter", "panel_c_caption"),
    ]:
        caption = fig_text(caption_key).strip()
        if caption:
            lines.append(f"{fig_text(letter_key).strip()} {caption}".strip())
    return "\n".join(lines).strip() + "\n"


def json_ready(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def main(*, close_figure: bool = True) -> plt.Figure:
    player_sub, centroid_sub, traj, window, meta = load_real_match_window()
    runs, pmv = load_collective_tables()
    panel_c_key = {
        "match_id": PANEL_C_MATCH_ID or window["match_id"],
        "match_phase": PANEL_C_MATCH_PHASE or window["match_phase"],
        "team": PANEL_C_TEAM or window["team"],
        "source_key": PANEL_C_SOURCE_KEY or window["source_key"],
    }
    segmentation_key = {
        "match_id": PANEL_B_MATCH_ID,
        "match_phase": PANEL_B_MATCH_PHASE,
        "team": PANEL_B_TEAM,
        "source_key": PANEL_B_SOURCE_KEY,
    }
    segmentation_sequence = select_segmentation_sequence(runs, traj, segmentation_key, PANEL_B_T0, PANEL_B_T1)
    high_snapshot, low_snapshot = select_polarisation_snapshots(pmv, traj, panel_c_key)
    pitch_xy = load_pitch_xy(meta.get("stadium_name", "Koteng Arena"))

    style_source = "analysis.levy_paper.util.paper_utils.configure_paper_plotting"
    if configure_paper_plotting is not None:
        configure_paper_plotting(base=PAPER_FONT_BASE)
    else:
        style_source = "local rcParams fallback"
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["CMU Serif", "Computer Modern", "STIX", "DejaVu Serif"],
                "font.size": PAPER_FONT_BASE,
                "axes.titlesize": PAPER_FONT_BASE * 1.15,
                "axes.labelsize": PAPER_FONT_BASE,
                "xtick.labelsize": PAPER_FONT_BASE * 0.9,
                "ytick.labelsize": PAPER_FONT_BASE * 0.9,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.72], hspace=FIGURE_GRID_HSPACE, wspace=FIGURE_GRID_WSPACE)
    fig.subplots_adjust(**FIGURE_SUBPLOT_ADJUST)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    gs_c = gs[1, :].subgridspec(
        1,
        5,
        width_ratios=[0.045, 0.405, 0.10, 0.405, 0.045],
        wspace=PANEL_C_GRID_WSPACE,
    )
    ax_c = fig.add_subplot(gs_c[0, 1:4])
    snapshot_axes = None

    draw_real_match_panel(ax_a, player_sub, centroid_sub, window, meta, pitch_xy)
    segmentation_audit = draw_real_segmentation_panel(ax_b, pitch_xy, traj, segmentation_sequence)
    profile_audit = draw_polarisation_profile_panel(ax_c, pitch_xy, traj, pmv, high_snapshot, low_snapshot, snapshot_axes=snapshot_axes)
    apply_top_panel_gap_adjust(fig, ax_a, ax_b)

    audit = {
        **window,
        "t0": window["t0"].isoformat(),
        "t1": window["t1"].isoformat(),
        "stadium_name": meta.get("stadium_name", ""),
        "home": meta.get("home", ""),
        "away": meta.get("away", ""),
        "score": meta.get("score", ""),
        "window_duration_s": WINDOW_DURATION_S,
        "active_depth_m": ACTIVE_DEPTH_M,
        "panel_mapping": "A=real 20s match trajectories with active/bench zones; B=clean real centroid sequence partitioned into angle-defined runs; C=short polarisation time profile with embedded real high/low formation insets",
        "pitch_style": "panel A uses bright active-zone green and red bench strip; B uses neutral pitch; C uses formation-only insets without mini-pitches",
        "panel_A_style": "bright active/bench pitch fill, sticky-hierarchical active-XI state, actual bench/excluded markers from raw labelled state when available, muted centroid accent",
        "panel_B": segmentation_audit,
        "panel_C_selection_key": panel_c_key,
        "panel_B_preselected_window": f"{PANEL_B_T0} to {PANEL_B_T1}",
        "panel_C": profile_audit,
        "neutral_pitch_panels": "B uses neutral pitch styling; C snapshots are embedded formation-only insets",
        "panel_C_player_traces": "player direction arrows use one-step unit displacements ending at the marked frame, matching the polarisation definition",
        "panel_C_formation_inset_expand": FORMATION_INSET_EXPAND,
        "panel_C_future_centroid_arrow_s": SNAPSHOT_FUTURE_S,
        "panel_C_centroid_arrow_length_m": FORMATION_CENTROID_ARROW_LENGTH_M,
        "panel_C_inset_bounds": PANEL_C_INSET_BOUNDS,
        "panel_C_profile_context_s": POLARISATION_CONTEXT_S,
        "panel_C_profile_rolling_window_s": POLARISATION_ROLLING_WINDOW_S,
        "panel_C_centroid_player_min_sep_m": FORMATION_CENTROID_PLAYER_MIN_SEP_M,
        "panel_C_player_arrow_min_sep_m": FORMATION_PLAYER_ARROW_MIN_SEP_M,
        "panel_C_player_arrow_spread_iter": FORMATION_PLAYER_ARROW_SPREAD_ITER,
        "panel_C_connector_lw": PANEL_C_CONNECTOR_LW,
        "panel_C_connector_alpha": PANEL_C_CONNECTOR_ALPHA,
        "figure_text": FIGURE_TEXT,
        "figure_caption": build_figure_caption().strip(),
        "style_source": style_source,
    }
    audit_path = OUT_DIR / "real_match_transport_mechanism_4panel_audit.json"
    audit_path.write_text(json.dumps(json_ready(audit), indent=2), encoding="utf-8")
    caption_path = OUT_DIR / "real_match_transport_mechanism_4panel_caption.txt"
    caption_path.write_text(build_figure_caption(), encoding="utf-8")

    player_disp_source = (
        player_sub[player_sub["player_status"].astype(str).eq("active")].copy()
        if "player_status" in player_sub.columns
        else player_sub.copy()
    )
    player_disp = pd.Series(
        [
            float(np.hypot(g.iloc[-1]["x_m"] - g.iloc[0]["x_m"], g.iloc[-1]["y_m"] - g.iloc[0]["y_m"]))
            for _, g in player_disp_source.sort_values("t").groupby("player_name")
            if len(g)
        ]
    )
    audit_csv = OUT_DIR / "real_match_20s_player_centroid_trajectories_audit.csv"
    pd.DataFrame(
        [
            {
                "match_id": window["match_id"],
                "match_phase": window["match_phase"],
                "team": window["team"],
                "source_key": window["source_key"],
                "t0": window["t0"].isoformat(),
                "t1": window["t1"].isoformat(),
                "n_players": window["n_players"],
                "centroid_displacement_m": window["centroid_displacement_m"],
                "median_player_displacement_m": float(player_disp.median()) if len(player_disp) else np.nan,
                "selection_score": window["selection_score"],
                "stadium_name": meta.get("stadium_name", ""),
                "home": meta.get("home", ""),
                "away": meta.get("away", ""),
                "score": meta.get("score", ""),
                "trajectory_source": str(DATA_DIR / TRAJECTORY_FILE),
                "n_players_plotted": window["n_players_plotted"],
                "window_duration_s": WINDOW_DURATION_S,
                "active_depth_m": ACTIVE_DEPTH_M,
                "pitch_style": "bright active-zone green and red bench-strip draw_pitch styling with outer box",
            }
        ]
    ).to_csv(audit_csv, index=False)

    png = OUT_DIR / "real_match_transport_mechanism_4panel.png"
    pdf = OUT_DIR / "real_match_transport_mechanism_4panel.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if close_figure:
        plt.close(fig)

    share_copy_errors = []
    for path in [png, pdf, caption_path, audit_path, audit_csv]:
        try:
            shutil.copy2(path, SHARE_DIR / path.name)
        except OSError as exc:
            share_copy_errors.append({"path": str(path), "error": str(exc)})

    print(f"SAVED {png}")
    print(f"SAVED {pdf}")
    print(f"SAVED {caption_path}")
    print(f"SAVED {audit_path}")
    print(f"SAVED {audit_csv}")
    if share_copy_errors:
        print(f"SKIPPED_LOCKED_SHARE_COPY {SHARE_DIR}")
        print(json.dumps(share_copy_errors, indent=2))
    else:
        print(f"COPIED outputs to {SHARE_DIR}")
    print(json.dumps(audit, indent=2))
    return fig


if __name__ == "__main__":
    main()
