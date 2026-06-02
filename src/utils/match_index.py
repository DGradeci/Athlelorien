# src/utils/match_index.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Union, List

import pandas as pd


# -----------------------------
# Constants / regex
# -----------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# -----------------------------
# String / date helpers
# -----------------------------
def _norm(s: str) -> str:
    """Normalise club names for robust matching."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _assert_date_str(date_str: str) -> None:
    if not _DATE_RE.match(str(date_str)):
        raise ValueError(f"date_str must be 'YYYY-MM-DD', got: {date_str}")


def build_day_prefix(bucket: str, source_base: str, date_str: str) -> str:
    """
    Build *DAY-level* S3 prefix:
      {bucket}/data/{source_base}_{YYYY}/{YYYY-MM}/{YYYY-MM-DD}
    """
    date_str = str(date_str)
    _assert_date_str(date_str)
    year = date_str[:4]
    month = date_str[:7]
    return f"{bucket}/data/{source_base}_{year}/{month}/{date_str}"


def _assert_is_day_prefix(prefix: str) -> None:
    """Ensure prefix ends in YYYY-MM-DD (exact day), not a month or year folder."""
    tail = prefix.rstrip("/").split("/")[-1]
    if not _DATE_RE.match(tail):
        raise ValueError(
            f"Expected DAY folder prefix ending with YYYY-MM-DD, got: {prefix}\n"
            f"Tip: prefix should look like .../YYYY-MM/YYYY-MM-DD"
        )


# -----------------------------
# Kamper helpers
# -----------------------------
def _all_clubs(matches: pd.DataFrame) -> List[str]:
    clubs = pd.concat([matches["home"], matches["away"]], ignore_index=True)
    clubs = clubs.dropna().astype(str).map(_norm)
    return sorted(clubs.unique())


def _club_dates(matches: pd.DataFrame, club_norm: str) -> Set[str]:
    home = matches["home"].astype(str).map(_norm)
    away = matches["away"].astype(str).map(_norm)
    mask = (home == club_norm) | (away == club_norm)
    return set(matches.loc[mask, "date"].astype(str))


def infer_two_clubs_date_only(matches: pd.DataFrame, dates_A: Set[str], dates_B: Set[str]) -> Dict[str, Any]:
    """
    Assign club names to Source A and Source B by maximising date overlap
    with each club's match dates in kamper.
    """
    clubs = _all_clubs(matches)
    club_to_dates = {c: _club_dates(matches, c) for c in clubs}

    best: Optional[Dict[str, Any]] = None
    for cA in clubs:
        sA = len(dates_A & club_to_dates[cA])
        for cB in clubs:
            if cB == cA:
                continue
            sB = len(dates_B & club_to_dates[cB])
            score = sA + sB
            if best is None or score > best["score"]:
                best = {
                    "club_A": cA,
                    "club_B": cB,
                    "A_overlap": sA,
                    "B_overlap": sB,
                    "score": score,
                }
    if best is None:
        raise RuntimeError("Could not infer clubs (kamper matches might be empty or malformed).")
    return best


# -----------------------------
# S3 discovery helpers
# -----------------------------
def list_available_dates(fs: Any, bucket: str, source_base: str, year: str = "2020") -> Set[str]:
    """
    Return available dates for a source by scanning:
      {bucket}/data/{source_base}_{year}/YYYY-MM/YYYY-MM-DD/
    """
    root = f"{bucket}/data/{source_base}_{year}"
    dates: Set[str] = set()

    # Use glob to reduce round-trips if available
    try:
        day_paths = fs.glob(f"{root}/*/*")
        for p in day_paths:
            d = str(p).rstrip("/").split("/")[-1]
            if _DATE_RE.match(d):
                dates.add(d)
        return dates
    except Exception:
        pass

    # Fallback: month-by-month listing
    for month_path in fs.ls(root):
        try:
            day_paths = fs.ls(month_path)
        except Exception:
            continue
        for day_path in day_paths:
            d = str(day_path).rstrip("/").split("/")[-1]
            if _DATE_RE.match(d):
                dates.add(d)

    return dates


# -----------------------------
# Main orchestration class
# -----------------------------
@dataclass
class SoccermonMatchIndex:
    """
    Orchestrates:
      kamper schedule (MatchesLoader output)  <->  S3 day folders (objective_* sources)

    It does NOT do pitch calibration, bench/active, or windowing to kickoff.
    Its job is just to:
      - infer club names for source A/B using date overlap
      - provide match lists (A, B, head-to-head)
      - load raw parquets for an *exact day* into df_raw
    """
    fs: Any
    bucket: str
    day_loader: Any                 # expects .load_day(prefix, ...) and optionally .load_day_1hz(prefix, ...)
    matches: pd.DataFrame           # from MatchesLoader().load()
    year: str = "2020"

    # Folder bases WITHOUT "_2020" suffix
    source_base_A: str = "objective_TEAM_A"
    source_base_B: str = "objective_team_B"

    # Filled by infer()
    club_A: Optional[str] = None
    club_B: Optional[str] = None
    catalog: Optional[pd.DataFrame] = None

    def infer(self) -> Dict[str, Any]:
        """
        Infer club name for source A and B and build an annotated catalog.
        """
        # 1) S3 date sets
        dates_A = list_available_dates(self.fs, self.bucket, self.source_base_A, year=self.year)
        dates_B = list_available_dates(self.fs, self.bucket, self.source_base_B, year=self.year)

        # 2) Assign clubs
        pair = infer_two_clubs_date_only(self.matches, dates_A, dates_B)
        self.club_A, self.club_B = pair["club_A"], pair["club_B"]

        # 3) Build catalog from kamper
        m = self.matches.copy()
        required = {"date", "home", "away"}
        if not required.issubset(set(m.columns)):
            raise ValueError(f"matches must contain columns: {sorted(required)}")

        m["home_norm"] = m["home"].astype(str).map(_norm)
        m["away_norm"] = m["away"].astype(str).map(_norm)

        m["has_A"] = (m["home_norm"] == self.club_A) | (m["away_norm"] == self.club_A)
        m["has_B"] = (m["home_norm"] == self.club_B) | (m["away_norm"] == self.club_B)
        m["head_to_head"] = m["has_A"] & m["has_B"]

        def _home_source(r):
            if r["home_norm"] == self.club_A:
                return "A"
            if r["home_norm"] == self.club_B:
                return "B"
            return None

        def _away_source(r):
            if r["away_norm"] == self.club_A:
                return "A"
            if r["away_norm"] == self.club_B:
                return "B"
            return None

        m["home_source"] = m.apply(_home_source, axis=1)
        m["away_source"] = m.apply(_away_source, axis=1)

        self.catalog = m

        return {
            "source_A_club": self.club_A,
            "source_B_club": self.club_B,
            "A_overlap": int(pair["A_overlap"]),
            "B_overlap": int(pair["B_overlap"]),
            "n_head_to_head": int(m["head_to_head"].sum()),
        }

    # -----------------------------
    # Catalog views
    # -----------------------------
    def matches_for_source(self, source_key: str) -> pd.DataFrame:
        if self.catalog is None:
            raise RuntimeError("Call infer() first.")
        if source_key == "A":
            return self.catalog[self.catalog["has_A"]].reset_index(drop=True)
        if source_key == "B":
            return self.catalog[self.catalog["has_B"]].reset_index(drop=True)
        raise ValueError("source_key must be 'A' or 'B'")

    def head_to_head(self) -> pd.DataFrame:
        if self.catalog is None:
            raise RuntimeError("Call infer() first.")
        return self.catalog[self.catalog["head_to_head"]].reset_index(drop=True)

    # -----------------------------
    # Prefix construction (DAY only)
    # -----------------------------
    def day_prefix(self, source_key: str, date_str: str) -> str:
        _assert_date_str(date_str)
        base = self.source_base_A if source_key == "A" else self.source_base_B
        prefix = build_day_prefix(self.bucket, base, date_str)
        _assert_is_day_prefix(prefix)
        return prefix

    # -----------------------------
    # Loaders (DAY only)
    # -----------------------------
    def load_day_source(
        self,
        date_str: str,
        source_key: str,
        one_hz: bool = False,
        max_files: Optional[int] = None,
        add_labels: bool = True,
        categorical_labels: bool = True,
    ) -> pd.DataFrame:
        """
        Load ALL parquets for a given DATE from one source ("A" or "B").

        Parameters
        ----------
        one_hz:
            If True, requires DayDataLoader.load_day_1hz(prefix, ...) to exist.
            This is the most reliable way to avoid MemoryError for head-to-head days.
        max_files:
            Optional dev-mode knob: load only the first N parquet files.
            Requires DayDataLoader.load_day(prefix, max_files=...) support (optional).
        add_labels:
            Add 'source_key' and 'team' columns.
        categorical_labels:
            Store 'source_key'/'team' as category to reduce memory.
        """
        if self.club_A is None or self.club_B is None:
            raise RuntimeError("Call infer() first (to set club names).")

        date_str = str(date_str)
        _assert_date_str(date_str)

        if source_key not in ("A", "B"):
            raise ValueError("source_key must be 'A' or 'B'")

        prefix = self.day_prefix(source_key, date_str)

        # IMPORTANT: NO .copy() here (can double memory for huge frames)
        if one_hz:
            if not hasattr(self.day_loader, "load_day_1hz"):
                raise AttributeError(
                    "DayDataLoader has no load_day_1hz(). "
                    "Implement load_day_1hz(prefix) to downsample while reading."
                )
            df = self.day_loader.load_day_1hz(prefix)
        else:
            # Try to pass max_files if DayDataLoader supports it
            if max_files is not None:
                try:
                    df = self.day_loader.load_day(prefix, max_files=max_files)
                except TypeError:
                    df = self.day_loader.load_day(prefix)
            else:
                df = self.day_loader.load_day(prefix)

        if add_labels:
            club = self.club_A if source_key == "A" else self.club_B
            df["source_key"] = source_key
            df["team"] = club

            if categorical_labels:
                # categories are tiny, saves memory vs repeating strings millions of times
                df["source_key"] = df["source_key"].astype("category")
                df["team"] = df["team"].astype("category")

        return df

    def load_team_match(
        self,
        row: Union[pd.Series, int],
        source_key: Optional[str] = None,
        one_hz: bool = False,
        max_files: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load ALL parquets for the DATE in the provided kamper row, from one source.
        (No windowing to kickoff; this is day-level raw data.)

        - row can be a pd.Series (from catalog/matches_for_source) or an int index into catalog.
        - source_key can be forced ("A"/"B") or inferred from home_source/away_source.
        """
        if self.catalog is None:
            raise RuntimeError("Call infer() first (to build catalog).")

        if isinstance(row, int):
            row = self.catalog.iloc[row]

        date_str = str(row["date"])
        _assert_date_str(date_str)

        if source_key is None:
            source_key = row.get("home_source") or row.get("away_source")

        if source_key not in ("A", "B"):
            raise ValueError("Could not infer source_key for this row; pass source_key='A' or 'B'.")

        return self.load_day_source(
            date_str=date_str,
            source_key=source_key,
            one_hz=one_hz,
            max_files=max_files,
        )

    def load_head_to_head(
        self,
        row: Union[pd.Series, int],
        one_hz: bool = False,
        max_files: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load ALL parquets for the DATE in the provided head-to-head row, from BOTH sources,
        concatenated with team labels.

        If you hit MemoryError:
          - use one_hz=True (requires DayDataLoader.load_day_1hz)
          - or use max_files=N to test pipeline
        """
        if self.catalog is None:
            raise RuntimeError("Call infer() first (to build catalog).")

        h2h = self.head_to_head()
        if isinstance(row, int):
            row = h2h.iloc[row]

        date_str = str(row["date"])
        _assert_date_str(date_str)

        dfA = self.load_day_source(date_str, "A", one_hz=one_hz, max_files=max_files)
        dfB = self.load_day_source(date_str, "B", one_hz=one_hz, max_files=max_files)

        # Concatenation itself can be memory-heavy if both are huge.
        return pd.concat([dfA, dfB], ignore_index=True)
