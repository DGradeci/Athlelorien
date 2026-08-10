"""
Pitch calibration utilities: lat/lon -> (x, y) in metres with pitch-aligned axes.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_STADIUM_ALIASES: Dict[str, str] = {
    "Avaldsnes 1": "Avaldsnes Idrettssenter",
    "Brann stadion": "Brann Stadion",
    "Klepp stadion": "Klepp Stadion",
    "Kringsjå kunstgress": "Kringsjå kunstgress",
    "Røa kunstgress": "Røa kunstgress",
    "Stemmemyren kunstgress": "Stemmemyren",
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth (in metres)."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polygon_latlon_centroid(coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    lats = [float(c[0]) for c in coords]
    lons = [float(c[1]) for c in coords]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _latlon_to_xy(
    lat_deg: pd.Series | np.ndarray,
    lon_deg: pd.Series | np.ndarray,
    lat0: float,
    lon0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate lat/lon differences as metres near (lat0, lon0)."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    lat_arr = np.asarray(lat_deg, dtype=float)
    lon_arr = np.asarray(lon_deg, dtype=float)

    x_m = (lon_arr - lon0) * m_per_deg_lon
    y_m = (lat_arr - lat0) * m_per_deg_lat
    return x_m, y_m


def polygon_latlon_to_xy(
    coords: List[Tuple[float, float]], origin_lat: float, origin_lon: float
) -> np.ndarray:
    lats = np.array([float(c[0]) for c in coords], dtype=float)
    lons = np.array([float(c[1]) for c in coords], dtype=float)
    x_m, y_m = _latlon_to_xy(lats, lons, origin_lat, origin_lon)
    return np.column_stack((x_m, y_m))


def _auto_scale_degrees(
    lat_raw: pd.Series, lon_raw: pd.Series
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Ensure lat/lon are in decimal degrees.

    If values look like 'microdegrees' (e.g. ~5.994e7 for 59.94°),
    we divide by 1e6. Otherwise we leave them alone.

    Returns
    -------
    lat_deg, lon_deg, scale
      where scale is the divisor applied to the original numbers.
    """
    lat = pd.to_numeric(lat_raw, errors="coerce")
    lon = pd.to_numeric(lon_raw, errors="coerce")

    typical = np.nanmedian(np.abs(lat))

    # crude heuristic: if it's way bigger than 180, assume microdegrees
    if typical > 180.0:
        scale = 1e6
        lat = lat / scale
        lon = lon / scale
    else:
        scale = 1.0

    return lat, lon, scale


def _norm_stadium_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def resolve_pitch_name(
    stadium_name: Any,
    pitches: Dict[str, Dict[str, Any]],
    aliases: Dict[str, str] | None = None,
) -> str | None:
    """Resolve a schedule stadium name to a pitch-registry key."""
    if not stadium_name:
        return None

    name = str(stadium_name).strip()
    if name in pitches:
        return name

    combined_aliases = dict(DEFAULT_STADIUM_ALIASES)
    if aliases:
        combined_aliases.update(aliases)

    norm_to_pitch = {_norm_stadium_name(k): k for k in pitches}
    norm_to_alias = {_norm_stadium_name(k): v for k, v in combined_aliases.items()}
    norm = _norm_stadium_name(name)

    if norm in norm_to_alias:
        target = norm_to_alias[norm]
        if target in pitches:
            return target
        return norm_to_pitch.get(_norm_stadium_name(target))

    return norm_to_pitch.get(norm)


def _pitch_geometry_from_info(
    name: str,
    info: Dict[str, Any],
) -> Tuple[str, Tuple[float, float], np.ndarray, List[Tuple[float, float]]]:
    pitch_coords_ll = info["coords"]
    c_lat, c_lon = polygon_latlon_centroid(pitch_coords_ll)
    P = polygon_latlon_to_xy(
        pitch_coords_ll, c_lat, c_lon
    )  # metres, origin at pitch centre

    C = P - P.mean(axis=0)  # just to stabilise SVD
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    v0, v1 = Vt[0], Vt[1]  # principal axes (unit vectors)
    if v0[0] < 0:
        v0, v1 = -v0, -v1
    R = np.column_stack((v0, v1))  # 2x2 rotation matrix

    pitch_xy_rot = P @ R
    pitch_xy_rot_list = [tuple(pt) for pt in pitch_xy_rot]
    return name, (c_lat, c_lon), R, pitch_xy_rot_list


def calibrate_pitch_from_df(
    df_1hz: pd.DataFrame,
    pitches: Dict[str, Dict[str, Any]],
    lat_col: str = "lat",
    lon_col: str = "lon",
    preferred_stadium: str | None = None,
    stadium_aliases: Dict[str, str] | None = None,
    warn_distance_m: float | None = 300.0,
    allow_fallback_to_nearest: bool = True,
) -> Tuple[str, Tuple[float, float], np.ndarray, List[Tuple[float, float]]]:
    """
    Finds the pitch in `pitches` used for GPS calibration.

    If preferred_stadium is provided, it is resolved against the pitch registry
    and used first. Otherwise, or if no registry entry/alias is available, the
    nearest stadium to the first valid GPS sample is used as a fallback.

    Parameters
    ----------
    df_1hz : DataFrame
        Must have lat_col, lon_col in (approx) decimal degrees.
    pitches : dict
        Dictionary as loaded from toppserien_pitches.json:
        {
          "Stemmemyren": {
             "way_id": ...,
             "tags": {...},
             "coords": [(lat1, lon1), ...]
          },
          ...
        }
    lat_col, lon_col : str
        Column names for positions in df_1hz.
    preferred_stadium : str, optional
        Scheduled venue name to prefer over nearest-GPS matching.
    stadium_aliases : dict, optional
        Extra aliases mapping schedule venue names to pitch-registry keys.
    warn_distance_m : float, optional
        Print a warning when the first GPS sample is farther than this from the
        selected pitch centre.
    allow_fallback_to_nearest : bool
        If False, unresolved preferred_stadium raises instead of falling back.

    Returns
    -------
    stadium_name : str
    center_latlon : (lat, lon)
    R : 2x2 np.ndarray
        Rotation matrix to align long axis horizontally.
    pitch_xy_rot_list : list[(x, y)]
        Rotated pitch polygon in metres, centred at (0, 0).
    """
    lat_raw = pd.to_numeric(df_1hz[lat_col], errors="coerce")
    lon_raw = pd.to_numeric(df_1hz[lon_col], errors="coerce")
    lat, lon, _ = _auto_scale_degrees(lat_raw, lon_raw)
    valid = lat.notna() & lon.notna() & (lat != 0) & (lon != 0)
    if not valid.any():
        raise ValueError("No valid lat/lon rows in df_1hz for calibration.")

    lat0 = float(lat.loc[valid].iloc[0])
    lon0 = float(lon.loc[valid].iloc[0])

    best_name: str | None = None
    best_info: Dict[str, Any] | None = None
    best_d = float("inf")
    match_method = "nearest"

    resolved = resolve_pitch_name(preferred_stadium, pitches, aliases=stadium_aliases)
    if resolved is not None:
        best_name = resolved
        best_info = pitches[resolved]
        c_lat, c_lon = polygon_latlon_centroid(best_info["coords"])
        best_d = haversine(lat0, lon0, c_lat, c_lon)
        match_method = "scheduled"
    elif preferred_stadium and not allow_fallback_to_nearest:
        raise KeyError(f"Scheduled stadium not found in pitch registry: {preferred_stadium!r}")

    if best_info is None:
        # Nearest stadium by centroid.
        for name, info in pitches.items():
            coords = info["coords"]  # list of [lat, lon]
            c_lat, c_lon = polygon_latlon_centroid(coords)
            d = haversine(lat0, lon0, c_lat, c_lon)
            if d < best_d:
                best_name, best_info, best_d = name, info, d

    if best_info is None or best_name is None:
        raise RuntimeError("Could not match any stadium from the pitches dictionary.")

    best_name, center_latlon, R, pitch_xy_rot_list = _pitch_geometry_from_info(best_name, best_info)
    c_lat, c_lon = center_latlon

    print(
        f"[calibrate] Stadium='{best_name}', method={match_method}, centre=({c_lat:.6f}, {c_lon:.6f}), "
        f"first-sample distance ≈ {best_d:.1f} m"
    )
    if warn_distance_m is not None and best_d > float(warn_distance_m):
        print(
            f"[calibrate] WARNING: first GPS sample is {best_d:.1f} m from selected pitch "
            f"'{best_name}' (preferred_stadium={preferred_stadium!r})."
        )

    return best_name, center_latlon, R, pitch_xy_rot_list


def attach_xy_from_pitch(
    df_1hz: pd.DataFrame,
    center_latlon: Tuple[float, float],
    R: np.ndarray,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    stadium_name: str | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Adds x_m, y_m in metres (centre + rotated) to df_1hz, dropping invalid lat/lon rows.

    Parameters
    ----------
    df_1hz : DataFrame
        Must contain latitude/longitude columns.
    center_latlon : (lat, lon)
        Pitch centre in degrees.
    R : 2x2 np.ndarray
        Rotation matrix as returned by calibrate_pitch_from_df.
    lat_col, lon_col : str
        Column names for latitude/longitude.
    stadium_name : str, optional
        If provided, a 'stadium_name' column is added.
    copy : bool
        If True (default), operate on a copy. Otherwise modify in-place.

    Returns
    -------
    DataFrame with new columns 'x_m', 'y_m' (and optionally 'stadium_name').
    """
    df = df_1hz.copy() if copy else df_1hz

    lat_raw = pd.to_numeric(df[lat_col], errors="coerce")
    lon_raw = pd.to_numeric(df[lon_col], errors="coerce")
    lat_deg, lon_deg, _ = _auto_scale_degrees(lat_raw, lon_raw)

    bad = (
        ((lat_deg == 0) & (lon_deg == 0)) | (lat_deg.abs() > 90) | (lon_deg.abs() > 180)
    )
    ok = (~bad) & lat_deg.notna() & lon_deg.notna()
    if not ok.any():
        return df.iloc[0:0].copy()

    df = df.loc[ok].reset_index(drop=True)

    lat0, lon0 = center_latlon
    x_local, y_local = _latlon_to_xy(lat_deg.loc[ok], lon_deg.loc[ok], lat0, lon0)

    XY_local = np.column_stack(
        (np.asarray(x_local, dtype=float), np.asarray(y_local, dtype=float))
    )
    XY = XY_local @ R

    df["x_m"] = XY[:, 0]
    df["y_m"] = XY[:, 1]
    if stadium_name:
        df["stadium_name"] = stadium_name
    return df
