from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[3]
LEVY_DIR = ROOT / "analysis" / "levy_paper"
DATA_DIR = LEVY_DIR / "data"
METADATA_DIR = LEVY_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIT_DIR = LEVY_DIR / "figures" / "2026-07-02_multiseason_data_cache" / "audits"
MANIFEST_DIR = LEVY_DIR / "figures" / "2026-07-02_multiseason_data_cache" / "manifests"

for _path in (DATA_DIR, PROCESSED_DIR, AUDIT_DIR, MANIFEST_DIR):
    _path.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.levy_paper.util.paper_utils import (  # noqa: E402
    ACTIVE_DEPTH_M,
    ACTIVE_PLAYER_METHOD,
    ACTIVATE_S,
    BENCH_OFF_S,
    MAX_ACTIVE_PLAYERS,
    THETA_DEG,
)
from src.utils.collective_stats import build_collective_order_tables_from_transport  # noqa: E402
from src.utils.data_access import S3DataAccess  # noqa: E402
from src.utils.day_loader import DayDataLoader, _extract_hhmmss_as_timedelta, norway_offset  # noqa: E402
from src.utils.hazard_stats import build_centroid_hazard_intervals  # noqa: E402
from src.utils.match_index import SoccermonMatchIndex  # noqa: E402
from src.utils.match_loader import MatchesLoader  # noqa: E402
from src.utils.match_phases import label_match_phases  # noqa: E402
from src.utils.pitch_calibration import attach_xy_from_pitch, calibrate_pitch_from_df  # noqa: E402
from src.utils.player_status import label_active_players  # noqa: E402
from src.utils.player_utils import PlayerNameMapper  # noqa: E402
from src.utils.trajectory_stats import build_transport_tables_from_active_v2  # noqa: E402


@dataclass(frozen=True)
class SeasonConfig:
    year: str
    kamper_path: Path
    source_base_A: str
    source_base_B: str
    source_club_A: str | None = None
    source_club_B: str | None = None


SEASON_CONFIGS: dict[str, SeasonConfig] = {
    "2020": SeasonConfig(
        year="2020",
        kamper_path=METADATA_DIR / "schedules" / "kamper_2020.xlsx",
        source_base_A="objective_TEAM_A",
        source_base_B="objective_team_B",
        source_club_A="rosenborg",
        source_club_B="vålerenga",
    ),
    "2021": SeasonConfig(
        year="2021",
        kamper_path=METADATA_DIR / "schedules" / "kamper_2021.xlsx",
        source_base_A="objective_Team_A",
        source_base_B="objective_Team_B",
        source_club_A="rosenborg",
        source_club_B="vålerenga",
    ),
}


def _load_pitch_registry() -> dict[str, Any]:
    candidates = [
        METADATA_DIR / "pitches" / "toppserien_pitches.json",
        ROOT / "toppserien_pitches.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8-sig"))
    raise FileNotFoundError("Could not find toppserien_pitches.json")


def _save_parquet(df: pd.DataFrame, stem: str, *, overwrite: bool, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.parquet"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing cache: {path}")
    df.to_parquet(path, index=False)
    return path


def _save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _stamp(df: pd.DataFrame, *, season: str, source_key: str, team: str, match_id: str) -> pd.DataFrame:
    out = df.copy()
    out["season"] = str(season)
    out["source_key"] = str(source_key)
    out["team"] = str(team)
    out["match_id"] = str(match_id)
    return out


def _norm_club(s: Any) -> str:
    return " ".join(str(s).strip().lower().split())


def _apply_source_club_override(index: SoccermonMatchIndex, club_A: str, club_B: str) -> None:
    """Override fragile date-overlap inference with known source-club identities."""
    club_A_norm = _norm_club(club_A)
    club_B_norm = _norm_club(club_B)
    m = index.matches.copy()

    m["home_norm"] = m["home"].astype(str).map(_norm_club)
    m["away_norm"] = m["away"].astype(str).map(_norm_club)

    m["has_A"] = (m["home_norm"] == club_A_norm) | (m["away_norm"] == club_A_norm)
    m["has_B"] = (m["home_norm"] == club_B_norm) | (m["away_norm"] == club_B_norm)
    m["head_to_head"] = m["has_A"] & m["has_B"]

    def _home_source(r: pd.Series) -> str | None:
        if r["home_norm"] == club_A_norm:
            return "A"
        if r["home_norm"] == club_B_norm:
            return "B"
        return None

    def _away_source(r: pd.Series) -> str | None:
        if r["away_norm"] == club_A_norm:
            return "A"
        if r["away_norm"] == club_B_norm:
            return "B"
        return None

    m["home_source"] = m.apply(_home_source, axis=1)
    m["away_source"] = m.apply(_away_source, axis=1)

    index.club_A = club_A_norm
    index.club_B = club_B_norm
    index.catalog = m


def _kickoff_time(row: pd.Series) -> pd.Timestamp:
    return pd.Timestamp(f"{row['date']} {row['time']}")


def _load_team_match_1hz(
    index: SoccermonMatchIndex,
    row: pd.Series,
    *,
    source_key: str,
    max_files: int | None,
) -> pd.DataFrame:
    """Load one source/day at 1 Hz while preserving max_files for smoke tests."""
    date_str = str(row["date"])
    prefix = index.day_prefix(source_key, date_str)
    raw = index.day_loader.load_day_1hz(prefix, max_files=max_files)
    team = index.club_A if source_key == "A" else index.club_B
    raw["source_key"] = source_key
    raw["team"] = str(team)
    return raw


def _tracking_window_from_first_file(index: SoccermonMatchIndex, row: pd.Series, source_key: str) -> dict[str, Any]:
    """Cheap guard against same-date non-match/training folders."""
    date_str = str(row["date"])
    out: dict[str, Any] = {
        "date": date_str,
        "source_key": source_key,
        "folder_exists": False,
        "n_files": 0,
        "sample_file": "",
        "track_start": pd.NaT,
        "track_end": pd.NaT,
        "covers_match_window": False,
        "coverage_error": "",
    }
    try:
        prefix = index.day_prefix(source_key, date_str)
        files = sorted([f for f in index.fs.ls(prefix) if str(f).endswith(".parquet")])
    except Exception as exc:
        out["coverage_error"] = f"missing_or_unreadable_source_folder: {exc!r}"
        return out

    out["folder_exists"] = True
    out["n_files"] = len(files)
    if not files:
        out["coverage_error"] = "no_parquet_files"
        return out

    out["sample_file"] = str(files[0])
    try:
        time_col = pd.read_parquet(files[0], filesystem=index.fs, columns=["time"])["time"]
        td = _extract_hhmmss_as_timedelta(time_col).dropna()
        if td.empty:
            out["coverage_error"] = "sample_file_has_no_parseable_time_values"
            return out
        match_date = pd.Timestamp(date_str).date()
        base = pd.Timestamp(date_str)
        offset = norway_offset(match_date)
        track_start = base + td.min() + pd.Timedelta(hours=offset)
        track_end = base + td.max() + pd.Timedelta(hours=offset)
        kickoff = _kickoff_time(row)
        out["track_start"] = track_start
        out["track_end"] = track_end
        out["minutes_start_before_kickoff"] = (kickoff - track_start).total_seconds() / 60.0
        out["minutes_end_after_kickoff"] = (track_end - kickoff).total_seconds() / 60.0
        out["covers_match_window"] = bool(
            (track_end >= kickoff) and (track_start <= kickoff + pd.Timedelta(minutes=105))
        )
    except Exception as exc:
        out["coverage_error"] = f"time_coverage_error: {exc!r}"
    return out


def _make_index(
    cfg: SeasonConfig,
    *,
    fs: Any,
    day_loader: DayDataLoader,
    bucket: str,
) -> tuple[SoccermonMatchIndex, pd.DataFrame, dict[str, Any]]:
    matches = MatchesLoader(str(cfg.kamper_path)).load()
    index = SoccermonMatchIndex(
        fs=fs,
        bucket=bucket,
        day_loader=day_loader,
        matches=matches,
        year=cfg.year,
        source_base_A=cfg.source_base_A,
        source_base_B=cfg.source_base_B,
    )
    inferred_info = index.infer()
    info = dict(inferred_info)
    info["inferred_source_A_club"] = inferred_info.get("source_A_club")
    info["inferred_source_B_club"] = inferred_info.get("source_B_club")
    if cfg.source_club_A and cfg.source_club_B:
        _apply_source_club_override(index, cfg.source_club_A, cfg.source_club_B)
        info["source_A_club"] = index.club_A
        info["source_B_club"] = index.club_B
        info["source_club_override"] = True
    else:
        info["source_club_override"] = False
    return index, matches, info


def _catalog_audit(
    configs: list[SeasonConfig],
    *,
    fs: Any,
    day_loader: DayDataLoader,
    bucket: str,
) -> tuple[pd.DataFrame, dict[str, SoccermonMatchIndex]]:
    rows: list[dict[str, Any]] = []
    indexes: dict[str, SoccermonMatchIndex] = {}
    for cfg in configs:
        try:
            index, matches, info = _make_index(cfg, fs=fs, day_loader=day_loader, bucket=bucket)
            indexes[cfg.year] = index
            source_A = index.matches_for_source("A")
            source_B = index.matches_for_source("B")
            h2h = index.head_to_head()
            rows.append(
                {
                    "season": cfg.year,
                    "kamper_path": str(cfg.kamper_path),
                    "source_base_A": cfg.source_base_A,
                    "source_base_B": cfg.source_base_B,
                    "source_A_club": info.get("source_A_club"),
                    "source_B_club": info.get("source_B_club"),
                    "inferred_source_A_club": info.get("inferred_source_A_club"),
                    "inferred_source_B_club": info.get("inferred_source_B_club"),
                    "source_club_override": info.get("source_club_override"),
                    "A_overlap": info.get("A_overlap"),
                    "B_overlap": info.get("B_overlap"),
                    "schedule_rows": len(matches),
                    "source_A_fixture_rows": len(source_A),
                    "source_B_fixture_rows": len(source_B),
                    "head_to_head_rows": len(h2h),
                    "head_to_head_dates": ";".join(h2h["date"].astype(str).tolist()) if not h2h.empty else "",
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "season": cfg.year,
                    "kamper_path": str(cfg.kamper_path),
                    "source_base_A": cfg.source_base_A,
                    "source_base_B": cfg.source_base_B,
                    "status": "error",
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(rows), indexes


def _process_source_match(
    *,
    index: SoccermonMatchIndex,
    row: pd.Series,
    source_key: str,
    season: str,
    pitch_registry: dict[str, Any],
    max_files: int | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    date_str = str(row["date"])
    team = index.club_A if source_key == "A" else index.club_B
    if team is None:
        raise RuntimeError("Match index club labels are unavailable; call infer() first.")

    raw = _load_team_match_1hz(index, row, source_key=source_key, max_files=max_files)
    raw_cols = list(raw.columns)
    scheduled_stadium = str(row.get("stadium", "") or "")
    stadium_name, center_latlon, rotation, pitch_xy = calibrate_pitch_from_df(
        raw,
        pitch_registry,
        preferred_stadium=scheduled_stadium,
    )
    xy = attach_xy_from_pitch(raw, center_latlon, rotation, stadium_name=stadium_name)
    kickoff_time = _kickoff_time(row)
    phased = label_match_phases(xy, kickoff_time=kickoff_time)
    play = phased.loc[phased["match_phase"].isin(["1H", "2H"])].copy()
    if play.empty:
        raise ValueError(
            f"No rows in match phases for {date_str} source={source_key}; "
            f"tracking window may not overlap kickoff={kickoff_time}."
        )
    play = label_active_players(
        play,
        pitch_xy=pitch_xy,
        active_depth_m=ACTIVE_DEPTH_M,
        activate_s=ACTIVATE_S,
        bench_off_s=BENCH_OFF_S,
        active_method=ACTIVE_PLAYER_METHOD,
        max_active_players=MAX_ACTIVE_PLAYERS,
    )
    active = play.loc[play["player_status"] == "active"].copy()
    if active.empty:
        raise ValueError(f"No active rows after player-status labelling for {date_str} source={source_key}.")
    active["season"] = str(season)
    active["match_id"] = date_str
    active["source_key"] = source_key
    active["team"] = str(team)

    transport = build_transport_tables_from_active_v2(active, theta_deg=THETA_DEG)
    pmv, centroid_order = build_collective_order_tables_from_transport(transport, verbose=False)
    hazard = build_centroid_hazard_intervals(transport, pmv, verbose=False)

    tables = {
        "runs_long": _stamp(transport["runs_long"], season=season, source_key=source_key, team=str(team), match_id=date_str),
        "trajectory_long": _stamp(
            transport["trajectory_long"], season=season, source_key=source_key, team=str(team), match_id=date_str
        ),
        "msd_long": _stamp(transport["msd_long"], season=season, source_key=source_key, team=str(team), match_id=date_str),
        "df_pmv": _stamp(pmv, season=season, source_key=source_key, team=str(team), match_id=date_str),
        "centroid_order_runs": _stamp(
            centroid_order, season=season, source_key=source_key, team=str(team), match_id=date_str
        ),
        "hazard_intervals": _stamp(hazard, season=season, source_key=source_key, team=str(team), match_id=date_str),
    }

    pitch_x = [float(p[0]) for p in pitch_xy]
    pitch_y = [float(p[1]) for p in pitch_xy]
    meta = {
        "season": str(season),
        "match_id": date_str,
        "date": date_str,
        "time": str(row.get("time", "")),
        "home": str(row.get("home", "")),
        "away": str(row.get("away", "")),
        "score": str(row.get("score", "")),
        "source_key": source_key,
        "team": str(team),
        "scheduled_stadium": scheduled_stadium,
        "stadium_name": stadium_name,
        "raw_rows_1hz": len(raw),
        "raw_unique_timestamps": int(raw["timestamp"].nunique()) if "timestamp" in raw.columns else 0,
        "active_rows": len(active),
        "active_player_method": ACTIVE_PLAYER_METHOD,
        "max_active_players": int(MAX_ACTIVE_PLAYERS),
        "active_players": int(active["player_name"].nunique()) if "player_name" in active.columns else 0,
        "has_ball_raw": any("ball" in c.lower() for c in raw_cols),
        "has_possession_raw": any(("possess" in c.lower()) or ("possession" in c.lower()) for c in raw_cols),
        "raw_columns": ";".join(map(str, raw_cols)),
        "pitch_xmin_m": min(pitch_x),
        "pitch_xmax_m": max(pitch_x),
        "pitch_ymin_m": min(pitch_y),
        "pitch_ymax_m": max(pitch_y),
        "pitch_length_m": max(pitch_x) - min(pitch_x),
        "pitch_width_m": max(pitch_y) - min(pitch_y),
        "status": "ok",
        "error": "",
    }
    return tables, meta


def _concat_parts(parts: dict[str, list[pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, frames in parts.items():
        if frames:
            out[name] = pd.concat(frames, ignore_index=True)
    return out


def _filter_match_dates(matches: pd.DataFrame, match_dates: list[str] | None) -> pd.DataFrame:
    """Restrict a schedule-derived match table to explicit ISO match dates."""
    if not match_dates or matches.empty:
        return matches
    requested = {pd.Timestamp(value).strftime("%Y-%m-%d") for value in match_dates}
    normalized = pd.to_datetime(matches["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return matches.loc[normalized.isin(requested)].copy()


def _missing_schedule_inputs(configs: list[SeasonConfig]) -> list[Path]:
    return [cfg.kamper_path for cfg in configs if not cfg.kamper_path.exists()]


def build_multiseason_cache(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    bucket = os.getenv("S3_BUCKET", "ucl-ai-soccormon-dataset")
    cache_dir = PROCESSED_DIR / args.cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    configs = [SEASON_CONFIGS[y] for y in args.seasons]
    missing_schedules = _missing_schedule_inputs(configs)
    if missing_schedules:
        formatted = "\n".join(f" - {path}" for path in missing_schedules)
        raise FileNotFoundError(
            "AWS cache rebuild requires local NFF schedule inputs:\n"
            f"{formatted}\n"
            "See analysis/levy_paper/metadata/schedules/README.md."
        )
    s3 = S3DataAccess()
    mapper = PlayerNameMapper()
    day_loader = DayDataLoader(s3.fs, mapper)
    pitch_registry = _load_pitch_registry()
    catalog, indexes = _catalog_audit(configs, fs=s3.fs, day_loader=day_loader, bucket=bucket)
    catalog_path = _save_csv(catalog, AUDIT_DIR / f"multiseason_catalog_{args.suffix}.csv")
    print("CATALOG_AUDIT", catalog_path)
    print(catalog.to_string(index=False))
    if args.dry_run:
        return {
            "created": [str(catalog_path)],
            "catalog_path": str(catalog_path),
            "dry_run": True,
        }

    unavailable_seasons = [cfg.year for cfg in configs if cfg.year not in indexes]
    if unavailable_seasons:
        print(
            "UNAVAILABLE_SEASONS",
            ",".join(unavailable_seasons),
            "See the catalog audit above for the underlying AWS error.",
            flush=True,
        )
        configs = [cfg for cfg in configs if cfg.year in indexes]
    if not configs:
        raise RuntimeError(
            "AWS catalog indexing failed for every requested season. "
            "Check S3 credentials and s3:ListBucket access, then inspect "
            f"{catalog_path}."
        )

    single_parts = {
        "runs_long": [],
        "trajectory_long": [],
        "msd_long": [],
        "df_pmv": [],
        "centroid_order_runs": [],
        "hazard_intervals": [],
    }
    h2h_parts = {
        "runs_long_h2h": [],
        "trajectory_long_h2h": [],
        "msd_long_h2h": [],
        "df_pmv_h2h": [],
        "centroid_order_runs_h2h": [],
        "hazard_intervals_h2h": [],
    }
    pmv_team_parts = {"A": [], "B": []}
    meta_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    created: list[str] = [str(catalog_path)]

    for cfg in configs:
        index = indexes[cfg.year]

        if not args.skip_single:
            for single_source in args.single_sources:
                source_matches = index.matches_for_source(single_source)
                source_matches = _filter_match_dates(source_matches, args.match_dates)
                if args.max_single_matches is not None:
                    source_matches = source_matches.head(int(args.max_single_matches))
                for _, row in source_matches.iterrows():
                    date_str = str(row["date"])
                    print(f"SINGLE season={cfg.year} source={single_source} date={date_str}", flush=True)
                    coverage = _tracking_window_from_first_file(index, row, single_source)
                    coverage.update({"season": cfg.year, "pool": "single", "match_id": date_str})
                    coverage_rows.append(coverage)
                    if args.require_match_window and not coverage.get("covers_match_window", False):
                        reason = coverage.get("coverage_error") or "tracking_window_does_not_overlap_kickoff"
                        error_rows.append(
                            {
                                "pool": "single",
                                "season": cfg.year,
                                "match_id": date_str,
                                "source_key": single_source,
                                "error": f"skipped: {reason}",
                                "traceback": "",
                            }
                        )
                        print(f"SKIPPED single {cfg.year} {date_str} {single_source}: {reason}", flush=True)
                        continue
                    try:
                        tables, meta = _process_source_match(
                            index=index,
                            row=row,
                            source_key=single_source,
                            season=cfg.year,
                            pitch_registry=pitch_registry,
                            max_files=args.max_files,
                        )
                        for name in single_parts:
                            single_parts[name].append(tables[name])
                        meta["pool"] = "single"
                        meta_rows.append(meta)
                    except Exception as exc:
                        error_rows.append(
                            {
                                "pool": "single",
                                "season": cfg.year,
                                "match_id": date_str,
                                "source_key": single_source,
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        print(f"FAILED single {cfg.year} {date_str} {single_source}: {exc!r}", flush=True)

        if not args.skip_h2h:
            h2h = index.head_to_head()
            h2h = _filter_match_dates(h2h, args.match_dates)
            if args.max_h2h_matches is not None:
                h2h = h2h.head(int(args.max_h2h_matches))
            for _, row in h2h.iterrows():
                date_str = str(row["date"])
                h2h_coverage = {}
                for source_key in ("A", "B"):
                    coverage = _tracking_window_from_first_file(index, row, source_key)
                    coverage.update({"season": cfg.year, "pool": "h2h", "match_id": date_str})
                    coverage_rows.append(coverage)
                    h2h_coverage[source_key] = coverage
                if args.require_match_window and not all(
                    h2h_coverage[k].get("covers_match_window", False) for k in ("A", "B")
                ):
                    for source_key in ("A", "B"):
                        if not h2h_coverage[source_key].get("covers_match_window", False):
                            reason = (
                                h2h_coverage[source_key].get("coverage_error")
                                or "tracking_window_does_not_overlap_kickoff"
                            )
                            error_rows.append(
                                {
                                    "pool": "h2h",
                                    "season": cfg.year,
                                    "match_id": date_str,
                                    "source_key": source_key,
                                    "error": f"skipped: {reason}",
                                    "traceback": "",
                                }
                            )
                            print(f"SKIPPED h2h {cfg.year} {date_str} {source_key}: {reason}", flush=True)
                    continue
                h2h_results: dict[str, tuple[dict[str, pd.DataFrame], dict[str, Any]]] = {}
                for source_key in ("A", "B"):
                    print(f"H2H season={cfg.year} source={source_key} date={date_str}", flush=True)
                    try:
                        tables, meta = _process_source_match(
                            index=index,
                            row=row,
                            source_key=source_key,
                            season=cfg.year,
                            pitch_registry=pitch_registry,
                            max_files=args.max_files,
                        )
                        meta["pool"] = "h2h"
                        h2h_results[source_key] = (tables, meta)
                    except Exception as exc:
                        error_rows.append(
                            {
                                "pool": "h2h",
                                "season": cfg.year,
                                "match_id": date_str,
                                "source_key": source_key,
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        print(f"FAILED h2h {cfg.year} {date_str} {source_key}: {exc!r}", flush=True)
                if set(h2h_results) == {"A", "B"}:
                    for source_key in ("A", "B"):
                        tables, meta = h2h_results[source_key]
                        h2h_parts["runs_long_h2h"].append(tables["runs_long"])
                        h2h_parts["trajectory_long_h2h"].append(tables["trajectory_long"])
                        h2h_parts["msd_long_h2h"].append(tables["msd_long"])
                        h2h_parts["df_pmv_h2h"].append(tables["df_pmv"])
                        h2h_parts["centroid_order_runs_h2h"].append(tables["centroid_order_runs"])
                        h2h_parts["hazard_intervals_h2h"].append(tables["hazard_intervals"])
                        pmv_team_parts[source_key].append(tables["df_pmv"])
                        meta_rows.append(meta)
                else:
                    successful = ",".join(sorted(h2h_results)) or "none"
                    error_rows.append(
                        {
                            "pool": "h2h",
                            "season": cfg.year,
                            "match_id": date_str,
                            "source_key": "A+B",
                            "error": f"discarded incomplete h2h pair; successful_sources={successful}",
                            "traceback": "",
                        }
                    )
                    print(f"DISCARDED incomplete h2h {cfg.year} {date_str}: successful_sources={successful}", flush=True)

    for stem, df in _concat_parts(single_parts).items():
        path = _save_parquet(df, f"{stem}_{args.suffix}", overwrite=args.overwrite, output_dir=cache_dir)
        created.append(str(path))
        print("SAVED", path, df.shape)

    for stem, df in _concat_parts(h2h_parts).items():
        path = _save_parquet(df, f"{stem}_{args.suffix}", overwrite=args.overwrite, output_dir=cache_dir)
        created.append(str(path))
        print("SAVED", path, df.shape)

    for source_key, frames in pmv_team_parts.items():
        if frames:
            df = pd.concat(frames, ignore_index=True)
            path = _save_parquet(
                df,
                f"df_pmv_team{source_key}_{args.suffix}",
                overwrite=args.overwrite,
                output_dir=cache_dir,
            )
            created.append(str(path))
            print("SAVED", path, df.shape)

    meta = pd.DataFrame(meta_rows)
    errors = pd.DataFrame(error_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = _save_csv(
        coverage_df,
        AUDIT_DIR / f"multiseason_match_window_coverage_{args.suffix}.csv",
    )
    created.append(str(coverage_path))
    print("COVERAGE_AUDIT", coverage_path, coverage_df.shape)
    meta_path = _save_parquet(
        meta,
        f"multiseason_raw_pitch_metadata_{args.suffix}",
        overwrite=args.overwrite,
        output_dir=cache_dir,
    )
    errors_path = _save_parquet(
        errors,
        f"multiseason_cache_errors_{args.suffix}",
        overwrite=args.overwrite,
        output_dir=cache_dir,
    )
    created.extend([str(meta_path), str(errors_path)])
    print("SAVED", meta_path, meta.shape)
    print("SAVED", errors_path, errors.shape)

    summary_rows = []
    for path_str in created:
        path = Path(path_str)
        row = {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        if path.suffix == ".parquet" and path.exists():
            try:
                df = pd.read_parquet(path)
                row["rows"] = len(df)
                row["columns"] = len(df.columns)
                if "season" in df.columns:
                    row["seasons"] = ";".join(sorted(df["season"].dropna().astype(str).unique()))
                if "match_id" in df.columns:
                    row["n_matches"] = int(df["match_id"].dropna().astype(str).nunique())
            except Exception as exc:
                row["read_error"] = repr(exc)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = _save_csv(summary, AUDIT_DIR / f"multiseason_cache_summary_{args.suffix}.csv")
    created.append(str(summary_path))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "bucket": bucket,
        "seasons": args.seasons,
        "suffix": args.suffix,
        "cache_dir": str(cache_dir),
        "single_source": args.single_source,
        "single_sources": args.single_sources,
        "max_single_matches": args.max_single_matches,
        "max_h2h_matches": args.max_h2h_matches,
        "max_files": args.max_files,
        "match_dates": args.match_dates,
        "require_match_window": bool(args.require_match_window),
        "catalog_audit": str(catalog_path),
        "match_window_coverage_csv": str(coverage_path),
        "summary_csv": str(summary_path),
        "created": created,
        "errors": int(len(errors)),
        "notes": [
            "Existing 2020-only caches are not overwritten unless --overwrite is used.",
            "Source-club labels are configured explicitly because AWS daily folders include training/non-match dates and date-overlap inference is ambiguous.",
            "Configured source labels: 2020 A=Rosenborg, 2020 B=Vålerenga, 2021 A=Rosenborg, 2021 B=Vålerenga.",
            "2021 AWS source folders use objective_Team_A/objective_Team_B capitalization.",
            "Date/source folders are skipped when a sampled tracking window does not overlap the scheduled kickoff window.",
            "H2H caches are filtered to complete source A+B pairs only; partial one-team H2H outputs are excluded.",
        ],
    }
    manifest_path = MANIFEST_DIR / f"multiseason_data_cache_manifest_{args.suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    created.append(str(manifest_path))

    print("SUMMARY", summary_path)
    print("MANIFEST", manifest_path)
    print("ERRORS", len(errors))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build versioned 2020+2021 Levy-paper data caches.")
    parser.add_argument("--seasons", nargs="+", default=["2020", "2021"], choices=sorted(SEASON_CONFIGS))
    parser.add_argument("--suffix", default="2020_2021_all_teams_pitchfix_sticky_active")
    parser.add_argument("--cache-name", default=None)
    parser.add_argument("--single-source", default=None, choices=["A", "B"])
    parser.add_argument("--single-sources", nargs="+", choices=["A", "B"], default=None)
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-h2h", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-single-matches", type=int, default=None)
    parser.add_argument("--max-h2h-matches", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--match-dates",
        nargs="+",
        default=None,
        help="Optional ISO dates (YYYY-MM-DD) to process after schedule/source matching.",
    )
    parser.add_argument("--no-require-match-window", dest="require_match_window", action="store_false")
    parser.set_defaults(require_match_window=True)
    args = parser.parse_args()
    if args.single_sources is None:
        args.single_sources = [args.single_source] if args.single_source else ["A", "B"]
    else:
        args.single_sources = list(dict.fromkeys(args.single_sources))
    args.single_source = args.single_sources[0]
    if args.cache_name is None:
        if args.suffix == "2020_2021_all_teams_pitchfix_sticky_active":
            args.cache_name = "all_team_2020_2021_sticky_active"
        else:
            args.cache_name = args.suffix
    return args


if __name__ == "__main__":
    build_multiseason_cache(parse_args())
