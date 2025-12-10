"""
Utilities for timestamp handling and fast 1 Hz downsampling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def shift_time_categories_fast(s: pd.Series, hours: int) -> pd.Series:
    """
    Shift 'time' strings (HH:MM | HH:MM:SS | HH:MM:SS.mmm) by whole hours (mod 24)
    by remapping ONLY the unique categories. Returns object dtype strings.
    Leaves unparsable categories unchanged.
    """
    if s.dtype == "O" or pd.api.types.is_string_dtype(s):
        cat = s.astype("category")
        cats = pd.Series(cat.cat.categories, dtype="string")
        # parse categories once (tiny array vs huge column)
        m = cats.str.extract(
            r"^\s*(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?\s*$",
            expand=True,
        )
        ok = m[0].notna() & m[1].notna()

        new_cats = cats.copy()
        if ok.any():
            hh = pd.to_numeric(m.loc[ok, 0], errors="coerce").fillna(0).astype(int)
            mm = pd.to_numeric(m.loc[ok, 1], errors="coerce").fillna(0).astype(int)
            ss = pd.to_numeric(m.loc[ok, 2], errors="coerce").fillna(0).astype(int)
            fr = m.loc[ok, 3].fillna("")

            hh = (hh + int(hours)) % 24
            base = hh.astype(str).str.zfill(2) + ":" + mm.astype(str).str.zfill(2)

            has_sec = m.loc[ok, 2].notna()
            base.loc[has_sec] = (
                base.loc[has_sec]
                + ":"
                + ss.loc[has_sec].astype(int).astype(str).str.zfill(2)
            )
            has_frac = fr.str.len() > 0
            if has_frac.any():
                idx = fr.index[has_frac]
                base.loc[idx] = base.loc[idx] + "." + fr.loc[idx]

            new_cats.loc[ok] = base

        out_cat = pd.Categorical.from_codes(cat.cat.codes, new_cats)
        return pd.Series(out_cat.astype("object"), index=s.index)
    else:
        # Non-string 'time' (rare). If datetime-like, add hours;
        # otherwise just return unchanged.
        if pd.api.types.is_datetime64_any_dtype(s):
            return (s + pd.Timedelta(hours=hours)).astype("datetime64[ns]")
        return s


def ensure_timestamp_fast(
    df: pd.DataFrame, time_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Ensure df has a datetime64[ns] column `time_col`.

    If:
      - `time_col` exists and is datetime-like -> return df unchanged.
      - `time_col` exists but is string -> parse with pd.to_datetime.
      - `time_col` missing but 'time' exists -> parse 'time' as HH:MM(:SS(.fff)) and
        build a Timestamp relative to a dummy date.
    """
    if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        return df

    out = df.copy()
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=False)
        return out

    if "time" in out.columns:
        s = out["time"].astype(str)
        # try HH:MM:SS.mmm then HH:MM:SS
        ts = pd.to_datetime(s, format="%H:%M:%S.%f", errors="coerce")
        m = ts.isna()
        if m.any():
            ts.loc[m] = pd.to_datetime(s[m], format="%H:%M:%S", errors="coerce")

        # anchor on arbitrary date but keep time-of-day
        base = pd.Timestamp("1970-01-01")
        out[time_col] = base + (ts - ts.dt.normalize())
        return out

    raise KeyError(f"Could not find a usable time column ('{time_col}' or 'time').")


def downsample_to_1hz_fast(
    df: pd.DataFrame,
    player_col: str = "player_name",
    time_col: str = "timestamp",
    keep: str = "first",  # or "last"
    assume_sorted: bool = True,
) -> pd.DataFrame:
    """
    Downsample to at most one row per second per player, using a NumPy-fast
    grouping on (player, floor(timestamp to seconds)).

    Parameters
    ----------
    df : DataFrame
        Must contain player_col and time_col.
    player_col : str
        Player identifier column.
    time_col : str
        Timestamp column. Will be created via ensure_timestamp_fast if needed.
    keep : {'first', 'last'}
        Which sample to keep within each (player, second) bin.
    assume_sorted : bool
        If True, assumes df is sorted by player_col then time_col for a faster path.

    Returns
    -------
    DataFrame with <= 1 sample per second per player.
    """
    if player_col not in df.columns:
        raise KeyError(f"Missing player column '{player_col}' in df.")

    df_ts = ensure_timestamp_fast(df, time_col=time_col)

    # --- convert to NumPy arrays (avoid pandas alignment overhead) ---
    p_codes = pd.factorize(df_ts[player_col], sort=False)[0].astype(
        np.int64, copy=False
    )
    t_ns = df_ts[time_col].astype("int64", copy=False).to_numpy()
    t_sec = (t_ns // 1_000_000_000).astype(np.int64, copy=False)

    if not assume_sorted:
        # sort by (player, second, timestamp)
        order = np.lexsort((t_ns, t_sec, p_codes))
        p_sorted = p_codes[order]
        s_sorted = t_sec[order]

        if keep == "first":
            keep_mask_sorted = np.empty_like(p_sorted, dtype=bool)
            keep_mask_sorted[0] = True
            keep_mask_sorted[1:] = (p_sorted[1:] != p_sorted[:-1]) | (
                s_sorted[1:] != s_sorted[:-1]
            )
        else:
            keep_mask_sorted = np.empty_like(p_sorted, dtype=bool)
            keep_mask_sorted[-1] = True
            keep_mask_sorted[:-1] = (p_sorted[:-1] != p_sorted[1:]) | (
                s_sorted[:-1] != s_sorted[1:]
            )

        kept_idx = df_ts.index.to_numpy()[order][keep_mask_sorted]
        return df_ts.loc[np.sort(kept_idx)]

    # fast path: already sorted by player then time
    p = p_codes
    s = t_sec
    if keep == "first":
        keep_mask = np.empty(len(df_ts), dtype=bool)
        keep_mask[0] = True
        keep_mask[1:] = (p[1:] != p[:-1]) | (s[1:] != s[:-1])
    else:
        keep_mask = np.empty(len(df_ts), dtype=bool)
        keep_mask[-1] = True
        keep_mask[:-1] = (p[:-1] != p[1:]) | (s[:-1] != s[1:])

    return df_ts.loc[keep_mask]
