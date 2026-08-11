"""Reusable helpers for the publication reproduction notebooks.

Figure-specific mathematics remains in the figure modules. This module keeps
path handling, provenance displays, and small descriptive summaries out of the
notebook cells so that the visible workflow stays concise and readable.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PRIMARY_CACHE_NAME = "all_team_2020_2021_sticky_active"
PRIMARY_CACHE_SUFFIX = "2020_2021_all_teams_pitchfix_sticky_active"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Return the repository root when called from a notebook or script."""
    current = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "analysis" / "levy_paper").is_dir() and (
            candidate / "requirements.txt"
        ).is_file():
            return candidate
    raise RuntimeError("Could not find the Athlelorien repository root.")


def relative_path(path: str | Path, root: str | Path) -> str:
    """Render *path* relative to *root* where possible."""
    path = Path(path)
    root = Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def publication_paths(root: str | Path) -> dict[str, Path]:
    """Return the canonical paper paths used by every publication notebook."""
    root = Path(root)
    levy_dir = root / "analysis" / "levy_paper"
    data_dir = levy_dir / "data"
    return {
        "root": root,
        "levy_dir": levy_dir,
        "data_dir": data_dir,
        "primary_cache_dir": data_dir / "processed" / PRIMARY_CACHE_NAME,
        "final_figures": levy_dir / "figures" / "main_figures",
        "source_data": levy_dir / "figures" / "source_data",
        "supplement": levy_dir / "figures" / "supplementary_material",
    }


def cache_path(
    cache_dir: str | Path,
    stem: str,
    suffix: str = PRIMARY_CACHE_SUFFIX,
) -> Path:
    """Return the canonical parquet path for a processed table."""
    return Path(cache_dir) / f"{stem}_{suffix}.parquet"


def load_processed_cache(
    cache_dir: str | Path,
    stem: str,
    suffix: str = PRIMARY_CACHE_SUFFIX,
    *,
    columns: Sequence[str] | None = None,
    filters=None,
) -> pd.DataFrame:
    """Load one processed parquet table with a clear missing-cache error."""
    path = cache_path(cache_dir, stem, suffix)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed cache: {path}. Build it from AWS or install "
            "the external cache bundle."
        )
    return pd.read_parquet(path, columns=columns, filters=filters)


def file_status(paths: Sequence[str | Path], root: str | Path) -> pd.DataFrame:
    """Summarise required files without reading their contents."""
    rows = []
    for value in paths:
        path = Path(value)
        rows.append(
            {
                "path": relative_path(path, root),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1024**2, 3)
                if path.exists() and path.is_file()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def csv_shapes(paths: Sequence[str | Path], root: str | Path) -> pd.DataFrame:
    """Read tracked CSV files and report their table dimensions."""
    rows = []
    for value in paths:
        path = Path(value)
        row = {
            "path": relative_path(path, root),
            "exists": path.exists(),
            "rows": np.nan,
            "columns": np.nan,
        }
        if path.exists():
            try:
                frame = pd.read_csv(path)
                row.update({"rows": len(frame), "columns": len(frame.columns)})
            except Exception as exc:
                row["error"] = f"{exc.__class__.__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def table_inventory(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarise loaded DataFrames by name."""
    return pd.DataFrame.from_dict(
        {
            name: {
                "rows": len(frame),
                "columns": len(frame.columns),
                "memory_mb": round(frame.memory_usage(deep=True).sum() / 1024**2, 2),
            }
            for name, frame in tables.items()
            if frame is not None
        },
        orient="index",
    )


def resolve_data_mode(
    requested_mode: str,
    required_cache_paths: Sequence[str | Path],
) -> str:
    """Resolve ``auto`` to ``cache`` or ``reviewer`` and validate cache mode."""
    valid = {"auto", "cache", "reviewer"}
    if requested_mode not in valid:
        raise ValueError(f"DATA_MODE must be one of {sorted(valid)}; got {requested_mode!r}.")
    paths = [Path(path) for path in required_cache_paths]
    cache_ready = bool(paths) and all(path.exists() for path in paths)
    if requested_mode == "cache" and not cache_ready:
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(f"Cache mode requested but files are missing: {missing}")
    if requested_mode == "auto":
        return "cache" if cache_ready else "reviewer"
    return requested_mode


def transport_run_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Return run counts and central transport statistics by track type."""
    work = runs.copy()
    for column in ("duration_s", "run_length_m"):
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    if "track_type" not in work:
        work["track_type"] = "all"
    aggregations: dict[str, tuple[str, object]] = {
        "runs": ("track_type", "size")
    }
    if "duration_s" in work:
        aggregations.update(
            median_duration_s=("duration_s", "median"),
            p95_duration_s=("duration_s", lambda x: x.quantile(0.95)),
        )
    if "run_length_m" in work:
        aggregations.update(
            median_length_m=("run_length_m", "median"),
            p95_length_m=("run_length_m", lambda x: x.quantile(0.95)),
        )
    return work.groupby("track_type", observed=True).agg(**aggregations)


def order_state_summary(
    runs: pd.DataFrame,
    state_col: str = "early_order_state",
) -> pd.DataFrame:
    """Summarise the sample represented by each low/mid/high order state."""
    metrics: dict[str, tuple[str, object]] = {
        "runs": (state_col, "size"),
        "median_duration_s": ("duration_s", "median"),
        "median_length_m": ("run_length_m", "median"),
    }
    speed_col = next(
        (column for column in ("v_mean_mps", "v_group_mean") if column in runs),
        None,
    )
    if speed_col:
        metrics["median_speed_mps"] = (speed_col, "median")
    return runs.groupby(state_col, observed=True).agg(**metrics)


def hazard_support_summary(panel_a: pd.DataFrame) -> pd.DataFrame:
    """Summarise empirical Figure 4 risk-set support and events."""
    support_col = next(
        (column for column in ("fitted_flag", "late_fit_support_flag") if column in panel_a),
        None,
    )
    sparse_col = "sparse_flag" if "sparse_flag" in panel_a else None
    if not support_col:
        return pd.DataFrame()
    group_cols = [support_col] + ([sparse_col] if sparse_col else [])
    aggregations: dict[str, tuple[str, object]] = {
        "age_bins": ("age_s", "size"),
        "min_age_s": ("age_s", "min"),
        "max_age_s": ("age_s", "max"),
    }
    if "n_at_risk" in panel_a:
        aggregations["total_interval_exposure"] = ("n_at_risk", "sum")
    if "n_terminated" in panel_a:
        aggregations["terminations"] = ("n_terminated", "sum")
    return (
        panel_a.groupby(group_cols, dropna=False, observed=True)
        .agg(**aggregations)
        .reset_index()
    )


def panel_inventory(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarise panel source tables before plotting."""
    return pd.DataFrame.from_dict(
        {
            name: {"rows": len(frame), "columns": len(frame.columns)}
            for name, frame in panels.items()
        },
        orient="index",
    )


def transition_row_sum_audit(transitions: pd.DataFrame) -> pd.DataFrame:
    """Check that each age-band/current-state transition row sums to one."""
    required = {"age_band", "current_state", "point_estimate"}
    if not required.issubset(transitions):
        return pd.DataFrame()
    named = {
        "probability_sum": ("point_estimate", "sum"),
        "transitions": ("point_estimate", "size"),
    }
    if "n_origin_state_transitions" in transitions:
        named["n_origin_state_transitions"] = (
            "n_origin_state_transitions",
            "max",
        )
    return (
        transitions.groupby(["age_band", "current_state"], observed=True)
        .agg(**named)
        .reset_index()
    )


def display_live_or_frozen(
    fig,
    frozen_path: str | Path,
    *,
    display_frozen: bool = True,
    width: int = 1100,
    dpi: int = 160,
) -> str:
    """Display a rendered live Matplotlib figure or a tracked frozen PNG."""
    from IPython.display import Image, Markdown, display

    if fig is not None:
        image_buffer = BytesIO()
        fig.savefig(
            image_buffer,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        display(Image(data=image_buffer.getvalue(), width=width))
        return "live_figure"
    frozen_path = Path(frozen_path)
    if display_frozen and frozen_path.exists():
        display(Image(filename=str(frozen_path), width=width))
        return "frozen_figure"
    display(Markdown(f"Missing figure: `{frozen_path}`"))
    return "missing_figure"
