"""
Utilities for loading and normalising match schedules from Excel / CSV.

This wraps your original procedural loader into a MatchesLoader class while
preserving behaviour and regex heuristics.
"""

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper functions – mostly direct lifts from your original script
# ---------------------------------------------------------------------------


def _clean_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s.replace("–", "-").replace("—", "-")


def _guess_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    norm = {c: re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in df.columns}
    for pat in patterns:
        rx = re.compile(pat)
        for c, n in norm.items():
            if rx.search(n):
                return c
    return None


def _split_match_col(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s = series.astype(str).map(_clean_str)
    m = s.str.extract(r"^(?P<home>.+?)\s*[-vV]\s*(?P<away>.+?)$")
    if m is not None and not m.empty:
        return m["home"].fillna(""), m["away"].fillna("")
    return (
        pd.Series([""] * len(s), index=s.index),
        pd.Series([""] * len(s), index=s.index),
    )


def _excel_serial_to_ts(vals: pd.Series) -> pd.Series:
    s = pd.to_numeric(vals, errors="coerce")
    base = pd.Timestamp("1899-12-30")
    ts = base + pd.to_timedelta(s, unit="D")
    ts[s.isna()] = pd.NaT
    return ts


def _parse_time_like(series: pd.Series) -> pd.Series:
    """
    Return HH:MM strings from textual times, Excel serial fractions,
    or e.g. 15.00 format, without using pd.to_datetime inference.
    """
    out = pd.Series([""] * len(series), index=series.index, dtype=object)

    # 1) Numeric (Excel fraction of a day)
    num = pd.to_numeric(series, errors="coerce")
    mask_num = num.notna()
    if mask_num.any():
        frac = num[mask_num] % 1.0
        secs = np.round(frac * 86400).astype(int)
        hh = (secs // 3600) % 24
        mm = (secs % 3600) // 60
        out.loc[mask_num] = [f"{int(h):02d}:{int(m):02d}" for h, m in zip(hh, mm)]

    # 2) Text forms
    s = series.astype(str)

    # normalise: drop "Kl.", spaces; convert dots to colons
    s_norm = (
        s.str.replace(r"(?i)^kl\.?\s*", "", regex=True)
        .str.replace(" ", "", regex=False)
        .str.replace(".", ":", regex=False)
    )

    m = s_norm.str.extract(r"^\s*(\d{1,2}):(\d{2})\s*$")
    mask_text = m[0].notna() & m[1].notna()
    if mask_text.any():
        h = m.loc[mask_text, 0].astype(int).clip(0, 23)
        mi = m.loc[mask_text, 1].astype(int).clip(0, 59)
        out.loc[mask_text] = [f"{int(H):02d}:{int(M):02d}" for H, M in zip(h, mi)]

    return out


def _ensure_date_time(
    date_s: pd.Series, time_s: Optional[pd.Series]
) -> Tuple[pd.Series, pd.Series]:
    """
    Standardise date and time columns.

    - Dates: strict parsing with dayfirst, then Excel serials, then ffill.
    - Times: parse from dedicated column if present, else from date fraction,
      then forward-fill blanks.
    """
    dt_try = pd.to_datetime(date_s, errors="coerce", dayfirst=True)
    if dt_try.isna().mean() > 0.3:
        dt_try = dt_try.fillna(_excel_serial_to_ts(date_s))
    dt_try = dt_try.ffill()
    date_out = dt_try.dt.date.astype(str).where(dt_try.notna(), "")

    if time_s is not None and time_s.notna().any():
        time_out = _parse_time_like(time_s)
    else:
        tod = dt_try.dt.strftime("%H:%M")
        time_out = tod.mask(tod == "00:00", "")
    time_out = time_out.replace("", np.nan).ffill().fillna("")

    return date_out, time_out


def _ffill_visual_merges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill 'merged' cells in Excel where only the top row has the date/time.
    """
    like_date = [
        c for c in df.columns if re.search(r"(dato|date|kampdato|dag)", str(c), re.I)
    ]
    like_time = [
        c
        for c in df.columns
        if re.search(r"(tid|time|klokkeslett|kickoff|avspark)", str(c), re.I)
    ]
    f = df.copy()
    for c in like_date + like_time:
        f[c] = f[c].replace("", np.nan).ffill()
    return f


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class MatchesLoader:
    """
    Loader for match schedules from Excel / CSV files.

    Usage
    -----
    loader = MatchesLoader("path/to/kamper.xlsx")
    matches = loader.load()
    """

    def __init__(self, src_path: str) -> None:
        self.src_path = src_path

    # ------------------------------
    # Public API
    # ------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load and normalise the matches table.

        Returns a DataFrame with columns:
        ['date', 'time', 'home', 'away', 'score']
        """
        raw = self._load_raw()

        if raw.empty:
            return pd.DataFrame(columns=["date", "time", "home", "away", "score"])

        # Remove all-empty columns
        raw = raw.loc[:, raw.notna().any(axis=0)]

        return self._parse_normalised(raw)

    # ------------------------------
    # Internal helpers
    # ------------------------------

    def _load_raw(self) -> pd.DataFrame:
        src_lower = self.src_path.lower()
        if src_lower.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(self.src_path)
            frames = []
            for sh in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sh)
                except Exception:
                    continue
                if not df.empty:
                    frames.append(_ffill_visual_merges(df))
            raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            raw = pd.read_csv(self.src_path)

        return raw

    def _parse_normalised(self, raw: pd.DataFrame) -> pd.DataFrame:
        # Guess likely columns
        date_col = _guess_col(raw, [r"(dato|date|kampdato|dag)$"])
        time_col = _guess_col(raw, [r"(tid|time|klokkeslett|kickoff|avspark)$"])
        home_col = _guess_col(raw, [r"(hjemmelag|home|vertslag|vert|lagh|lag1)$"])
        away_col = _guess_col(raw, [r"(bortelag|away|gjestelag|lagb|lag2)$"])
        match_col = _guess_col(raw, [r"(kamp|match|oppgjor|oppgjr)$"])
        score_col = _guess_col(raw, [r"(resultat|score|sluttresultat)$"])

        date_s = raw[date_col] if date_col else pd.Series([""] * len(raw))
        time_s = raw[time_col] if time_col else None

        # Home / away extraction
        if home_col and away_col:
            home = raw[home_col].astype(str).map(_clean_str)
            away = raw[away_col].astype(str).map(_clean_str)
        elif match_col:
            home, away = _split_match_col(raw[match_col])
        else:
            best: Optional[str] = None
            for c in raw.columns:
                s = raw[c].astype(str)
                if s.str.contains(r"\s[-–—vV]\s", regex=True).mean() > 0.5:
                    best = c
                    break
            if best is not None:
                home, away = _split_match_col(raw[best])
            else:
                home = pd.Series([""] * len(raw))
                away = pd.Series([""] * len(raw))

        # Score "x-y" from many variants
        if score_col:
            sc_txt = raw[score_col].astype(str).map(_clean_str)
        else:
            sc_txt = pd.Series([""] * len(raw))

        sc = sc_txt.str.extract(r"^\s*(\d+)\s*[-:\.]\s*(\d+)\s*$")
        score = (sc[0].fillna("") + "-" + sc[1].fillna("")).str.strip("-")

        date_out, time_out = _ensure_date_time(date_s, time_s)

        out = pd.DataFrame(
            {
                "date": date_out.map(_clean_str),
                "time": time_out.map(_clean_str),
                "home": home.map(_clean_str),
                "away": away.map(_clean_str),
                "score": score.map(_clean_str),
            }
        )

        out = out[(out["home"] != "") | (out["away"] != "")]
        out = out.reset_index(drop=True)
        return out
