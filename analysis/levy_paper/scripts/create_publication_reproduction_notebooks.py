from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
LEVY_DIR = ROOT / "analysis" / "levy_paper"
OUT_DIR = LEVY_DIR / "notebooks_publication"

KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}

COMMON_SETUP = r"""
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


# Locate the repository before importing its analysis package. This works when
# Jupyter starts from either the repo root or this notebook directory.
_start = Path.cwd()
ROOT = next(
    path for path in (_start, *_start.parents)
    if (path / "analysis" / "levy_paper").is_dir()
    and (path / "requirements.txt").is_file()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.levy_paper.util.publication_notebook_utils import (
    PRIMARY_CACHE_SUFFIX,
    cache_path as make_cache_path,
    csv_shapes,
    display_live_or_frozen,
    file_status,
    hazard_support_summary,
    load_processed_cache,
    order_state_summary,
    panel_inventory,
    publication_paths,
    relative_path,
    resolve_data_mode,
    table_inventory,
    transition_row_sum_audit,
    transport_run_summary,
)

PATHS = publication_paths(ROOT)
LEVY_DIR = PATHS["levy_dir"]
DATA_DIR = PATHS["data_dir"]
PRIMARY_CACHE_DIR = PATHS["primary_cache_dir"]
FINAL_FIGURES = PATHS["final_figures"]
SOURCE_DATA = PATHS["source_data"]
SUPPLEMENT = PATHS["supplement"]
CACHE_SUFFIX = PRIMARY_CACHE_SUFFIX

# DATA_MODE options:
#   "auto"     use processed caches when all required files exist;
#              otherwise use tracked reviewer tables/frozen figures
#   "cache"    require processed caches and fail clearly if they are missing
#   "reviewer" use only tracked public artefacts
DATA_MODE = "auto"
BUILD_FIGURE = True
SAVE_FIGURE_OUTPUTS = True
DISPLAY_FROZEN_OUTPUT = True
REBUILD_CACHE_FROM_AWS = False
REFIT_FIGURE4_FIGURE5_MODELS = False
USE_VERSIONED_FINAL_FIGURE4_FIT = True


def rel(path):
    return relative_path(path, ROOT)


def show_file_status(paths):
    return file_status(paths, ROOT)


def show_csv_shapes(paths):
    return csv_shapes(paths, ROOT)


def cache_path(stem):
    return make_cache_path(PRIMARY_CACHE_DIR, stem, CACHE_SUFFIX)


def show_figure(fig, frozen_path, width=1100):
    return display_live_or_frozen(
        fig,
        frozen_path,
        display_frozen=DISPLAY_FROZEN_OUTPUT,
        width=width,
    )
"""


def md(text: str) -> nbf.NotebookNode:
    source = textwrap.dedent(text).strip()
    cell = nbf.v4.new_markdown_cell(source)
    cell["id"] = _cell_id("markdown", source)
    return cell


def code(text: str) -> nbf.NotebookNode:
    source = textwrap.dedent(text).strip()
    cell = nbf.v4.new_code_cell(source)
    cell["id"] = _cell_id("code", source)
    return cell


def _cell_id(cell_type: str, source: str) -> str:
    """Return a stable Jupyter cell ID for reproducible notebook generation."""
    payload = f"{cell_type}\0{source}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def write_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = KERNEL
    notebook["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


def data_cache_notebook() -> list[nbf.NotebookNode]:
    return [
        md("""
        # 00 - Match selection, AWS access, and local cache

        Select seasons, tracked sources, and optional match dates here. Local NFF schedule exports are required only for an AWS rebuild and are not distributed in the public repository. The final build cell runs only when `REBUILD_CACHE_FROM_AWS = True`.
        """),
        code(COMMON_SETUP),
        md("## Analysis selection"),
        code(r"""
SELECTED_SEASONS = [2020, 2021]
SELECTED_SOURCES = ["A", "B"]
SELECTED_MATCH_DATES = None  # Example: ["2021-05-22", "2021-06-30"]
INCLUDE_SINGLE_TEAM_FILES = True
INCLUDE_HEAD_TO_HEAD_FILES = True
REQUIRE_MATCH_WINDOW = True

# Use a new name/suffix for exploratory subsets. The canonical names below
# reproduce the paper cache and should be overwritten only deliberately.
OUTPUT_CACHE_NAME = "all_team_2020_2021_sticky_active"
OUTPUT_SUFFIX = "2020_2021_all_teams_pitchfix_sticky_active"
OVERWRITE_EXISTING_CACHE = False
MAX_SINGLE_MATCHES = None
MAX_H2H_MATCHES = None
MAX_FILES_PER_MATCH = None

# Local load controls. The frame-level trajectory and full player-run tables
# are opt-in because they expand substantially in memory after parquet decoding.
LOAD_ANALYSIS_TABLES = True
LOAD_CENTROID_RUNS = True
LOAD_FULL_RUN_TABLE = False
LOAD_TRAJECTORY_TABLE = False
TRAJECTORY_COLUMNS = [
    "match_id", "match_phase", "team", "source_key", "track_type",
    "track_entity_uid", "player_name", "t", "x_m", "y_m", "season",
]
"""),
        md("## Schedule selection"),
        code(r"""
schedule_paths = {
    2020: LEVY_DIR / "metadata" / "schedules" / "kamper_2020.xlsx",
    2021: LEVY_DIR / "metadata" / "schedules" / "kamper_2021.xlsx",
}
schedule_frames = []
missing_schedule_paths = [
    schedule_paths[season]
    for season in SELECTED_SEASONS
    if not schedule_paths[season].exists()
]
if missing_schedule_paths:
    schedule = pd.DataFrame()
    print("local_schedule_inputs_missing")
    for path in missing_schedule_paths:
        print(" -", rel(path))
    print("required_only_for_aws_rebuild", rel(LEVY_DIR / "metadata" / "schedules" / "README.md"))
else:
    for season in SELECTED_SEASONS:
        path = schedule_paths[season]
        frame = pd.read_excel(path)
        frame["season"] = season
        frame["schedule_source"] = rel(path)
        schedule_frames.append(frame)
    schedule = pd.concat(schedule_frames, ignore_index=True)

if SELECTED_MATCH_DATES and not schedule.empty:
    requested_dates = {pd.Timestamp(value).strftime("%Y-%m-%d") for value in SELECTED_MATCH_DATES}
    date_columns = [column for column in schedule.columns if "date" in str(column).lower() or "dato" in str(column).lower()]
    if not date_columns:
        raise KeyError("No date-like schedule column found.")
    schedule_dates = pd.to_datetime(schedule[date_columns[0]], errors="coerce").dt.strftime("%Y-%m-%d")
    schedule = schedule.loc[schedule_dates.isin(requested_dates)].copy()

print("selected_schedule_rows", len(schedule))
display(schedule.head(20))
"""),
        md("## Pitch registry and processed cache"),
        code(r"""
pitch_registry = LEVY_DIR / "metadata" / "pitches" / "toppserien_pitches.json"
primary_cache_files = [
    cache_path("trajectory_long"),
    cache_path("runs_long"),
    cache_path("msd_long"),
    cache_path("df_pmv"),
    cache_path("centroid_order_runs"),
    cache_path("hazard_intervals"),
]
display(show_file_status([pitch_registry, *schedule_paths.values(), *primary_cache_files]))

cache_inventory = []
for path in primary_cache_files:
    row = {"file": rel(path), "exists": path.exists()}
    if path.exists():
        sample = pd.read_parquet(path).head(3)
        row.update({"columns": len(sample.columns), "sample_rows": len(sample)})
    cache_inventory.append(row)
display(pd.DataFrame(cache_inventory))
"""),
        md("## Build command"),
        code(r"""
builder = LEVY_DIR / "scripts" / "build_multiseason_data_cache.py"
build_command = [
    sys.executable, str(builder),
    "--seasons", *map(str, SELECTED_SEASONS),
    "--single-sources", *SELECTED_SOURCES,
    "--cache-name", OUTPUT_CACHE_NAME,
    "--suffix", OUTPUT_SUFFIX,
]
if not INCLUDE_SINGLE_TEAM_FILES:
    build_command.append("--skip-single")
if not INCLUDE_HEAD_TO_HEAD_FILES:
    build_command.append("--skip-h2h")
if not REQUIRE_MATCH_WINDOW:
    build_command.append("--no-require-match-window")
if SELECTED_MATCH_DATES:
    build_command.extend(["--match-dates", *SELECTED_MATCH_DATES])
if MAX_SINGLE_MATCHES is not None:
    build_command.extend(["--max-single-matches", str(MAX_SINGLE_MATCHES)])
if MAX_H2H_MATCHES is not None:
    build_command.extend(["--max-h2h-matches", str(MAX_H2H_MATCHES)])
if MAX_FILES_PER_MATCH is not None:
    build_command.extend(["--max-files", str(MAX_FILES_PER_MATCH)])
if OVERWRITE_EXISTING_CACHE:
    build_command.append("--overwrite")

print(" ".join(build_command))
"""),
        md("## Optional AWS cache build"),
        code(r"""
canonical_subset_risk = bool(SELECTED_MATCH_DATES) and (
    OUTPUT_CACHE_NAME == "all_team_2020_2021_sticky_active"
    or OUTPUT_SUFFIX == "2020_2021_all_teams_pitchfix_sticky_active"
)
if REBUILD_CACHE_FROM_AWS:
    if missing_schedule_paths:
        missing = "\n".join(f" - {rel(path)}" for path in missing_schedule_paths)
        raise FileNotFoundError(
            "AWS rebuild requires local NFF schedule inputs:\n"
            f"{missing}\n"
            f"See {rel(LEVY_DIR / 'metadata' / 'schedules' / 'README.md')}"
        )
    if canonical_subset_risk:
        raise ValueError("Choose a new OUTPUT_CACHE_NAME and OUTPUT_SUFFIX for a date subset.")
    subprocess.run(build_command, cwd=ROOT, check=True)
else:
    print("AWS build skipped")
"""),
        md("## Load processed data into the notebook"),
        code(r"""
analysis_tables = {}
selected_cache_dir = DATA_DIR / "processed" / OUTPUT_CACHE_NAME
default_analysis_paths = [
    selected_cache_dir / f"{name}_{OUTPUT_SUFFIX}.parquet"
    for name in ["df_pmv", "msd_long", "centroid_order_runs", "hazard_intervals", "runs_long"]
]
resolved_mode = resolve_data_mode(DATA_MODE, default_analysis_paths)
selected_cache_ready = resolved_mode == "cache"
print("resolved_data_mode", resolved_mode)

if LOAD_ANALYSIS_TABLES and selected_cache_ready:
    for table_name in ["df_pmv", "msd_long", "centroid_order_runs", "hazard_intervals"]:
        analysis_tables[table_name] = load_processed_cache(
            selected_cache_dir, table_name, OUTPUT_SUFFIX
        )

    # Named variables make the tables immediately usable in exploratory cells.
    df_pmv = analysis_tables["df_pmv"]
    msd_long = analysis_tables["msd_long"]
    centroid_order_runs = analysis_tables["centroid_order_runs"]
    hazard_intervals = analysis_tables["hazard_intervals"]

centroid_runs = None
if LOAD_CENTROID_RUNS and selected_cache_ready:
    centroid_runs = load_processed_cache(
        selected_cache_dir,
        "runs_long",
        OUTPUT_SUFFIX,
        filters=[("track_type", "==", "centroid")],
    )

runs_long = None
if LOAD_FULL_RUN_TABLE and selected_cache_ready:
    runs_long = load_processed_cache(selected_cache_dir, "runs_long", OUTPUT_SUFFIX)

trajectory_long = None
trajectory_path = selected_cache_dir / f"trajectory_long_{OUTPUT_SUFFIX}.parquet"
if LOAD_TRAJECTORY_TABLE and trajectory_path.exists():
    trajectory_long = load_processed_cache(
        selected_cache_dir,
        "trajectory_long",
        OUTPUT_SUFFIX,
        columns=TRAJECTORY_COLUMNS,
    )

loaded_summary = [
    {"variable": name, "rows": len(frame), "columns": len(frame.columns)}
    for name, frame in {
        **analysis_tables,
        "centroid_runs": centroid_runs,
        "runs_long": runs_long,
        "trajectory_long": trajectory_long,
    }.items()
    if frame is not None
]
display(pd.DataFrame(loaded_summary))
if not selected_cache_ready:
    print("Processed cache unavailable; dataframes were not loaded. Frozen/source-table figure mode remains available.")
"""),
        md("## Selected-data examples"),
        code(r"""
if centroid_runs is not None:
    display(centroid_runs.head())
if analysis_tables:
    display(hazard_intervals.head())
    display(centroid_order_runs.head())
"""),
    ]


def figure1_notebook() -> list[nbf.NotebookNode]:
    return [
        md("""
        # 01 - Figure 1: transport mechanism

        **Question.** How are player trajectories reduced to a team-centroid
        trajectory, segmented into runs, and related to collective order?

        | Panel | Construction |
        |---|---|
        | A | Real 20-second player and centroid trajectories on the calibrated pitch |
        | B | Direction-change segmentation of the centroid path |
        | C/D | High- and low-polarisation player configurations |

        The selected match windows and required caches are shown before the
        final plotting function is called.
        """),
        code(COMMON_SETUP),
        md("## 1. Data mode and required inputs"),
        code(r"""
from analysis.levy_paper.scripts import create_real_match_transport_mechanism_4panel as figure1

figure1_inputs = [cache_path("trajectory_long"), cache_path("runs_long"), cache_path("df_pmv")]
resolved_mode = resolve_data_mode(DATA_MODE, figure1_inputs)
cache_ready = resolved_mode == "cache"
print("resolved_data_mode", resolved_mode)
display(show_file_status(figure1_inputs))

"""),
        md("## 2. Available match/team trajectories"),
        code(r"""
games = pd.DataFrame()

if cache_ready:
    games = figure1.available_games(limit=30)
    display(games)
else:
    print("Processed trajectory cache unavailable; reviewer fallback will be used.")
"""),
        md("## 3. Selected real-match windows"),
        code(r"""
selection = pd.DataFrame([
    {"panel": "A", "match_id": figure1.PANEL_A_MATCH_ID, "phase": figure1.PANEL_A_MATCH_PHASE, "team": figure1.PANEL_A_TEAM, "source": figure1.PANEL_A_SOURCE_KEY, "t0": figure1.PANEL_A_T0},
    {"panel": "B", "match_id": figure1.PANEL_B_MATCH_ID, "phase": figure1.PANEL_B_MATCH_PHASE, "team": figure1.PANEL_B_TEAM, "source": figure1.PANEL_B_SOURCE_KEY, "t0": figure1.PANEL_B_T0},
    {"panel": "C", "match_id": figure1.PANEL_C_MATCH_ID, "phase": figure1.PANEL_C_MATCH_PHASE, "team": figure1.PANEL_C_TEAM, "source": figure1.PANEL_C_SOURCE_KEY, "t0": None},
])
display(selection)
"""),
        md("## 4. Construct the publication figure"),
        code(r"""
fig = None
if BUILD_FIGURE and cache_ready:
    fig = figure1.main(close_figure=False)
elif BUILD_FIGURE:
    print("Processed trajectory/run/order cache unavailable; using frozen Figure 1.")

display_mode = show_figure(fig, FINAL_FIGURES / "figure1_transport_mechanism_schematic.png")
print("figure_display_mode", display_mode)
"""),
    ]


def figure2_notebook() -> list[nbf.NotebookNode]:
    return [
        md("""
        # 02 - Figure 2: transport phenotype

        **Question.** What statistical transport phenotype distinguishes the
        team centroid from its constituent player trajectories?

        | Panel | Analysis |
        |---|---|
        | A | Run-duration survivor functions |
        | B | Run-length survivor functions |
        | C | Mean-squared displacement and fitted scaling ranges |
        | D | Player MSD decomposed into centroid translation and motion within the formation |
        """),
        code(COMMON_SETUP),
        md("## 1. Load processed transport data"),
        code(r"""
from analysis.levy_paper.scripts import create_figure2_transport_phenotype as figure2_base
from analysis.levy_paper.scripts import create_final_figure2_transport_phenotype as figure2

figure2_inputs = {
    "runs": cache_path("runs_long"),
    "msd": cache_path("msd_long"),
    "trajectory": cache_path("trajectory_long"),
}
resolved_mode = resolve_data_mode(DATA_MODE, list(figure2_inputs.values()))
cache_ready = resolved_mode == "cache"
print("resolved_data_mode", resolved_mode)
display(show_file_status(figure2_inputs.values()))

inputs = {}
if cache_ready:
    inputs = {name: pd.read_parquet(path) for name, path in figure2_inputs.items()}
    display(table_inventory(inputs))
    display(transport_run_summary(inputs["runs"]))
"""),
        md("## 2. Compute the common panel context"),
        code(r"""
context = None
if inputs:
    context = figure2_base.prepare_figure2_context(inputs=inputs, write_audits=False)
    context["completed_centroid_runs"] = figure2.completed_centroid_runs(inputs["runs"])
    display(panel_inventory({
        "duration survivor audit": context["duration_audit"],
        "length survivor audit": context["length_audit"],
        "MSD aggregate": context["msd_agg"],
        "MSD bootstrap": context["msd_boot"],
        "MSD decomposition": context["decomp"],
    }))
"""),
        md("## 3. Panels A and B - run survivor functions"),
        code(r"""
if context is not None:
    survivor_summary = pd.DataFrame([
        {
            "population": "centroid",
            "runs": len(context["centroid_runs"]),
            "median_duration_s": context["centroid_runs"]["duration_s"].median(),
            "median_length_m": context["centroid_runs"]["run_length_m"].median(),
        },
        {
            "population": "player",
            "runs": len(context["player_runs"]),
            "median_duration_s": context["player_runs"]["duration_s"].median(),
            "median_length_m": context["player_runs"]["run_length_m"].median(),
        },
    ])
    display(survivor_summary)
    display(context["duration_audit"].head(10))
    display(context["length_audit"].head(10))
"""),
        md("## 4. Panel C - MSD scaling"),
        code(r"""
if context is not None:
    display(context["scaling_audit"])
    display(context["local_slope_audit"].head(12))
"""),
        md("## 5. Panel D - centroid-frame MSD decomposition"),
        code(r"""
if context is not None:
    display(context["decomp"].head(12))
    display(context["decomp_components"].head(12))
"""),
        md("## 6. Duration-tail model used in Panel A"),
        code(r"""
tail_decision = None
if context is not None:
    tail_audit = figure2.load_tail_audit()
    tail_decision = figure2.broad_tail_decision(tail_audit)
    display(pd.DataFrame([tail_decision]))
"""),
        md("## 7. Construct the publication figure"),
        code(r"""
fig = None
source_tables = None
if BUILD_FIGURE and context is not None:
    fig, source_tables = figure2.plot_final_figure(context, tail_decision)
    if SAVE_FIGURE_OUTPUTS:
        figure2.save_figure_bundle(fig)
        figure2.write_source_data(source_tables)
elif BUILD_FIGURE:
    print("Processed run/MSD/trajectory cache unavailable; using frozen Figure 2.")

display_mode = show_figure(fig, FINAL_FIGURES / "figure2_transport_phenotype.png")
print("figure_display_mode", display_mode)
"""),
    ]


def figure3_notebook() -> list[nbf.NotebookNode]:
    return [
        md("""
        # 03 - Figure 3: order and transport

        **Question.** How is collective order associated with run persistence,
        displacement, and centroid speed?

        | Panel | Analysis |
        |---|---|
        | A | Duration survivor by early-order tercile |
        | B | Length survivor by early-order tercile |
        | C | Joint order-speed phenotype coloured by run duration |

        The state classification and bootstrap unit are made explicit below.
        """),
        code(COMMON_SETUP),
        md("## 1. Load the centroid-run population"),
        code(r"""
from analysis.levy_paper.scripts import create_final_figure3_order_transport as figure3

run_cache = cache_path("centroid_order_runs")
resolved_mode = resolve_data_mode(DATA_MODE, [run_cache])
cache_ready = resolved_mode == "cache"
print("resolved_data_mode", resolved_mode)
display(show_file_status([run_cache]))

raw_runs = None
runs = None
if cache_ready:
    raw_runs = pd.read_parquet(run_cache)
    runs = figure3.load_runs()
    display(pd.DataFrame([
        {"stage": "cached rows", "rows": len(raw_runs)},
        {"stage": "centroid, complete-case and physical-range filters", "rows": len(runs)},
    ]))
"""),
        md("## 2. Define early-order states"),
        code(r"""
if runs is not None:
    q1, q2 = runs["p_early_3s"].quantile([1 / 3, 2 / 3]).to_numpy(float)
    state_bounds = pd.DataFrame([
        {"state": "Low", "lower": runs["p_early_3s"].min(), "upper": q1},
        {"state": "Mid", "lower": q1, "upper": q2},
        {"state": "High", "lower": q2, "upper": runs["p_early_3s"].max()},
    ])
    display(state_bounds)
    display(order_state_summary(runs))
"""),
        md("## 3. Cluster bootstrap and survivor grids"),
        code(r"""
duration_grid = length_grid = None
if runs is not None:
    cluster_cols = figure3.best_cluster_cols(runs)
    duration_grid = np.arange(1.0, min(75.0, max(35.0, runs["duration_s"].quantile(0.995))) + 1.0, 2.0)
    length_grid = np.linspace(1.0, min(85.0, max(25.0, runs["run_length_m"].quantile(0.995))), 34)
    print("bootstrap_cluster_columns", cluster_cols)
    print("bootstrap_cluster_count", runs[cluster_cols].drop_duplicates().shape[0])
    display(pd.DataFrame({"duration_grid_s": pd.Series(duration_grid), "length_grid_m": pd.Series(length_grid)}).head(12))
"""),
        md("## 4. Inspect empirical survivor values by state"),
        code(r"""
if runs is not None:
    survivor_rows = []
    for state in figure3.STATE_ORDER:
        subset = runs.loc[runs["early_order_state"].astype(str).eq(state)]
        for age, estimate in zip(duration_grid[:10], figure3.ccdf(subset["duration_s"], duration_grid[:10])):
            survivor_rows.append({"state": state, "duration_s": age, "survival": estimate, "runs": len(subset)})
    display(pd.DataFrame(survivor_rows).head(15))
"""),
        md("## 5. Inspect the order-speed phenotype inputs"),
        code(r"""
if runs is not None:
    phenotype_summary = runs.groupby("early_order_state", observed=True).agg(
        runs=("duration_s", "size"),
        mean_polarisation=("p_mean", "mean"),
        mean_centroid_speed_mps=("v_mean_mps", "mean"),
        median_duration_s=("duration_s", "median"),
    )
    display(phenotype_summary)
"""),
        md("## 6. Construct the publication figure"),
        code(r"""
fig = None
created = []
if BUILD_FIGURE and runs is not None:
    figure3.configure_paper_plotting(base=figure3.PAPER_FONT_BASE)
    fig, created = figure3.build_publication_figure(
        runs, duration_grid, length_grid, cluster_cols,
        close_figure=False,
        save_outputs=SAVE_FIGURE_OUTPUTS,
    )
elif BUILD_FIGURE:
    print("Processed order-run cache unavailable; using frozen Figure 3.")

display_mode = show_figure(fig, FINAL_FIGURES / "figure3_order_transport.png")
print("figure_display_mode", display_mode)
"""),
    ]


def figure4_notebook() -> list[nbf.NotebookNode]:
    return [
        md("""
        # 04 - Figure 4: killed transport

        **Question.** How do run age, recent collective order, and a smooth
        late-time termination contribution combine to determine run survival?

        | Panel | Analysis |
        |---|---|
        | A | Empirical interval termination probability and fitted hazard components |
        | B | Termination probability stratified by recent order state |
        | C | Empirical and model-reconstructed duration survivor functions |

        Cache mode recomputes panel tables from interval and cross-fitted model
        caches. Set `REFIT_FIGURE4_FIGURE5_MODELS = True` only when a full local
        model refit is intended.
        """),
        code(COMMON_SETUP),
        md("## 1. Select the model-cache pathway"),
        code(r"""
model_cache = DATA_DIR / "processed" / "figure4_fig5_crossfitted" / "out_of_fold_interval_predictions.parquet"
if REFIT_FIGURE4_FIGURE5_MODELS:
    if DATA_MODE == "reviewer":
        raise ValueError("Model refitting is incompatible with reviewer mode.")
    if not cache_path("hazard_intervals").exists():
        raise FileNotFoundError("The hazard-interval cache is required for refitting.")
    subprocess.run([sys.executable, str(LEVY_DIR / "scripts" / "build_figure4_figure5_model_cache.py")], cwd=ROOT, check=True)
else:
    print("model refit skipped", rel(model_cache), model_cache.exists())

resolved_mode = resolve_data_mode(DATA_MODE, [model_cache])
print("resolved_data_mode", resolved_mode)
display(show_file_status([cache_path("hazard_intervals"), model_cache]))
"""),
        md("## 2. Derive the three panel tables"),
        code(r"""
from analysis.levy_paper.scripts import create_final_fig4_fig5_polished as figure45

if resolved_mode == "cache":
    if REFIT_FIGURE4_FIGURE5_MODELS or not USE_VERSIONED_FINAL_FIGURE4_FIT:
        rebuilt_panel_a_audit = model_cache.parent / "figure4_panelA_hazard_audit.csv"
        if not rebuilt_panel_a_audit.exists():
            raise FileNotFoundError(f"Missing rebuilt Figure 4 hazard audit: {rel(rebuilt_panel_a_audit)}")
        figure45.PANEL_A_AUDIT = rebuilt_panel_a_audit
    panel_bundle = figure45.build_main_panel_sources()
    panel_a = panel_bundle["figure4a"]
    panel_b = panel_bundle["figure4b"]
    panel_c = panel_bundle["figure4c"]
    hazard_parameters = panel_bundle["tables"]["hazard_parameters"]
    validation_metrics = panel_bundle["tables"]["validation_metrics"]
    fit_mode = "versioned_final_smooth_late_fit" if USE_VERSIONED_FINAL_FIGURE4_FIT and not REFIT_FIGURE4_FIGURE5_MODELS else "rebuilt_model_cache_fit"
    panel_source_mode = f"derived_from_interval_and_crossfit_caches_with_{fit_mode}"
else:
    panel_a = pd.read_csv(SOURCE_DATA / "figure4A_source_data.csv")
    panel_b = pd.read_csv(SOURCE_DATA / "figure4B_source_data.csv")
    panel_c = pd.read_csv(SOURCE_DATA / "figure4C_source_data.csv")
    hazard_parameters = pd.read_csv(SOURCE_DATA / "crossfit_hazard_parameters.csv")
    validation_metrics = pd.read_csv(SOURCE_DATA / "final_crossfitted_validation_metrics.csv")
    panel_source_mode = "tracked_publication_source_table_fallback"

print("panel_source_mode", panel_source_mode)
display(panel_inventory({"Panel A": panel_a, "Panel B": panel_b, "Panel C": panel_c}))
"""),
        md("## 3. Panel A - empirical risk sets and fitted support"),
        code(r"""
display(hazard_support_summary(panel_a))
panel_a_columns = [
    column for column in [
        "age_s", "n_at_risk", "n_terminated", "empirical_estimate",
        "ci_low", "ci_high", "age_only_baseline", "full_hazard_model",
        "sparse_flag", "fitted_flag",
    ] if column in panel_a
]
display(panel_a[panel_a_columns].head(15))
"""),
        md("## 4. Panel B - recent-order termination strata"),
        code(r"""
panel_b_summary = panel_b.groupby("state", observed=True).agg(
    age_bins=("age_s", "size"),
    interval_exposure=("n_at_risk", "sum"),
    terminations=("n_terminated", "sum"),
    min_age_s=("age_s", "min"),
    max_age_s=("age_s", "max"),
).reset_index()
display(panel_b_summary)
display(panel_b[[column for column in ["age_s", "state", "n_at_risk", "n_terminated", "empirical_estimate", "ci_low", "ci_high"] if column in panel_b]].head(15))
"""),
        md("## 5. Panel C - interval-consistent survival reconstruction"),
        code(r"""
panel_c_summary = panel_c.groupby("series", observed=True).agg(
    points=("age_s", "size"),
    min_age_s=("age_s", "min"),
    max_age_s=("age_s", "max"),
    min_survival=("estimate", "min"),
).reset_index()
display(panel_c_summary)
"""),
        md("## 6. Fitted parameters and held-out validation"),
        code(r"""
display(hazard_parameters.head(20))
display(validation_metrics)
"""),
        md("## 7. Construct the publication figure"),
        code(r"""
fig = None
if BUILD_FIGURE:
    figure45.configure_style()
    created = []
    fig = figure45.plot_figure4(
        panel_a, panel_b, panel_c, created,
        close_figure=False,
        save_outputs=SAVE_FIGURE_OUTPUTS,
    )
display_mode = show_figure(fig, FINAL_FIGURES / "figure4_killed_transport.png")
print("figure_display_mode", display_mode)
"""),
    ]


def figure5_notebook() -> list[nbf.NotebookNode]:
    return [
        md("""
        # 05 - Figure 5: state-structured survival

        **Question.** Does age-dependent switching among low-, mid-, and
        high-order states reproduce the state composition of surviving runs?

        | Panel | Analysis |
        |---|---|
        | A | Age-banded order-state transition probabilities |
        | B | Survivor-conditioned current-state composition |
        | C | Cumulative state exposure among surviving runs |

        Cache mode derives every panel from the cross-fitted model outputs;
        reviewer mode uses the tracked panel source tables.
        """),
        code(COMMON_SETUP),
        md("## 1. Select the model-cache pathway"),
        code(r"""
model_cache = DATA_DIR / "processed" / "figure4_fig5_crossfitted" / "out_of_fold_interval_predictions.parquet"
if REFIT_FIGURE4_FIGURE5_MODELS:
    if DATA_MODE == "reviewer":
        raise ValueError("Model refitting is incompatible with reviewer mode.")
    if not cache_path("hazard_intervals").exists():
        raise FileNotFoundError("The hazard-interval cache is required for refitting.")
    subprocess.run([sys.executable, str(LEVY_DIR / "scripts" / "build_figure4_figure5_model_cache.py")], cwd=ROOT, check=True)
else:
    print("model refit skipped", rel(model_cache), model_cache.exists())

resolved_mode = resolve_data_mode(DATA_MODE, [model_cache])
print("resolved_data_mode", resolved_mode)
display(show_file_status([cache_path("hazard_intervals"), model_cache]))
"""),
        md("## 2. Derive panel and transition tables"),
        code(r"""
from analysis.levy_paper.scripts import create_final_fig4_fig5_polished as figure45

if resolved_mode == "cache":
    panel_bundle = figure45.build_main_panel_sources()
    panel_a = panel_bundle["figure5a"]
    panel_b = panel_bundle["figure5b"]
    panel_c = panel_bundle["figure5c"]
    transitions = panel_bundle["tables"]["transition_matrices"]
    validation = panel_bundle["tables"]["validation_metrics"]
    panel_source_mode = "derived_from_processed_crossfit_model_cache"
else:
    panel_a = pd.read_csv(SOURCE_DATA / "figure5A_source_data.csv")
    panel_b = pd.read_csv(SOURCE_DATA / "figure5B_source_data.csv")
    panel_c = pd.read_csv(SOURCE_DATA / "figure5C_source_data.csv")
    transitions = pd.read_csv(SOURCE_DATA / "crossfit_transition_parameters.csv")
    validation = pd.read_csv(SOURCE_DATA / "crossfit_validation_metrics.csv")
    panel_source_mode = "tracked_publication_source_table_fallback"

print("panel_source_mode", panel_source_mode)
display(panel_inventory({
    "Panel A": panel_a,
    "Panel B": panel_b,
    "Panel C": panel_c,
    "transition parameters": transitions,
}))
"""),
        md("## 3. Panel A - age-dependent transition probabilities"),
        code(r"""
display(transition_row_sum_audit(transitions))
transition_columns = [
    column for column in [
        "age_band", "current_state", "next_state", "point_estimate",
        "lower_uncertainty_bound", "upper_uncertainty_bound",
        "n_origin_state_transitions",
    ] if column in transitions
]
display(transitions[transition_columns].head(18))
"""),
        md("## 4. Panel B - survivor-conditioned current state"),
        code(r"""
panel_b_summary = panel_b.groupby("model_name", observed=True).agg(
    points=("age_s", "size"),
    min_age_s=("age_s", "min"),
    max_age_s=("age_s", "max"),
).reset_index()
display(panel_b_summary)
display(panel_b.head(12))
"""),
        md("## 5. Panel C - survivor state exposure"),
        code(r"""
panel_c_summary = panel_c.groupby("model_name", observed=True).agg(
    points=("threshold_s", "size"),
    min_threshold_s=("threshold_s", "min"),
    max_threshold_s=("threshold_s", "max"),
).reset_index()
display(panel_c_summary)
display(panel_c.head(12))
"""),
        md("## 6. Held-out validation"),
        code(r"""
display(validation)
"""),
        md("## 7. Construct the publication figure"),
        code(r"""
fig = None
if BUILD_FIGURE:
    figure45.configure_style()
    created = []
    fig = figure45.plot_figure5(
        panel_a, panel_b, panel_c, created,
        close_figure=False,
        save_outputs=SAVE_FIGURE_OUTPUTS,
    )
display_mode = show_figure(fig, FINAL_FIGURES / "figure5_state_structured_survival.png")
print("figure_display_mode", display_mode)
"""),
    ]


def supplement_notebook() -> list[nbf.NotebookNode]:
    return [
        md("""
        # 06 - Supplementary figures and tables

        **Question.** Do the main duration and segmentation results remain
        stable under alternative model and segmentation choices?

        | Figure | Analysis |
        |---|---|
        | S1 | Duration-model comparison and leave-one-fixture-out contrasts |
        | S2 | Temporal-sampling and turning-threshold robustness |

        Both figures are reconstructed directly from tracked source-data CSVs;
        no AWS or private processed cache is required.
        """),
        code(COMMON_SETUP),
        md("## 1. Source-data and formal-table inventory"),
        code(r"""
source_files = sorted((SUPPLEMENT / "source_data").glob("*.csv"))
table_files = sorted((SUPPLEMENT / "tables").glob("*.csv"))
display(show_csv_shapes(source_files))
display(show_csv_shapes(table_files))
"""),
        md("## 2. Figure S1 inputs - duration-model comparison"),
        code(r"""
s1_curves = pd.read_csv(SUPPLEMENT / "source_data" / "figureS1_panelA_model_curves.csv")
s1_contrasts = pd.read_csv(SUPPLEMENT / "source_data" / "figureS1_panelB_lofo_contrasts.csv")
duration_gof = pd.read_csv(SUPPLEMENT / "source_data" / "duration_model_gof_final.csv")
display(panel_inventory({
    "S1 model curves": s1_curves,
    "S1 LOFO contrasts": s1_contrasts,
    "duration goodness of fit": duration_gof,
}))
display(duration_gof)
display(s1_contrasts.head(12))
"""),
        md("## 3. Construct Figure S1"),
        code(r"""
from analysis.levy_paper.scripts import create_supplementary_figures_from_source as supplement_figures
from analysis.levy_paper.util.paper_utils import configure_paper_plotting

configure_paper_plotting(base=10)
fig_s1 = None
if BUILD_FIGURE:
    fig_s1, paths_s1 = supplement_figures.figure_s1(close_figure=False)
display_mode_s1 = show_figure(fig_s1, SUPPLEMENT / "figures" / "figureS1_duration_model_comparison.png")
print("figureS1_display_mode", display_mode_s1)
"""),
        md("## 4. Figure S2 inputs - segmentation robustness"),
        code(r"""
s2_source = pd.read_csv(SUPPLEMENT / "source_data" / "figureS2_segmentation_robustness_source.csv")
s2_table = pd.read_csv(SUPPLEMENT / "source_data" / "tableS_segmentation_robustness_final.csv")
display(panel_inventory({"S2 plotted source": s2_source, "S2 numerical table": s2_table}))
display(s2_table)
"""),
        md("## 5. Construct Figure S2"),
        code(r"""
fig_s2 = None
if BUILD_FIGURE:
    fig_s2, paths_s2 = supplement_figures.figure_s2(close_figure=False)
display_mode_s2 = show_figure(fig_s2, SUPPLEMENT / "figures" / "figureS2_segmentation_robustness_final.png")
print("figureS2_display_mode", display_mode_s2)
"""),
    ]


def readme_text() -> str:
    return """# Publication Reproduction Notebooks

Run the notebooks in numerical order. They follow the same visible pattern: define the scientific question, declare the data pathway, inspect the input population, display intermediate numerical tables, and finally call the authoritative publication plotting function.

Notebook 00 selects seasons, tracked sources, and optional match dates and can rebuild the processed cache for authorised AWS users. Notebooks 01-03 reconstruct their analyses and figures from that processed cache. Notebooks 04-05 derive their panels from the processed cross-fitted model cache, with optional refitting from hazard intervals; without that cache they use tracked publication source tables. Notebook 06 reconstructs the supplement from tracked source tables.

Every figure notebook ends by displaying the Matplotlib figure it constructed. When a required private processed cache is absent, the notebook reports that fact and displays the tracked frozen publication figure instead.

The three supported paths are:

1. Public/reviewer: set `DATA_MODE = "reviewer"`; tracked source tables and frozen figures require no AWS.
2. Local cache: install the external cache under `analysis/levy_paper/data/processed/all_team_2020_2021_sticky_active/` and set `DATA_MODE = "cache"`.
3. AWS collaborator: supply the local NFF workbooks documented in `../metadata/schedules/README.md`, configure credentials, edit the selection in notebook 00, set `REBUILD_CACHE_FROM_AWS = True`, build a named cache, and then use that cache in the figure notebooks.

`DATA_MODE = "auto"` is the default. It uses the processed cache only when every required file exists and otherwise selects the reviewer pathway. `REFIT_FIGURE4_FIGURE5_MODELS = False` reuses the audited cross-fitted model cache; changing it to `True` performs the expensive local refit from hazard intervals without contacting AWS.

The schedule workbooks are not needed in reviewer or local-cache mode and are not redistributed in the public repository.
"""


def main() -> None:
    notebooks = {
        "00_data_access_and_cache.ipynb": data_cache_notebook(),
        "01_figure1_transport_mechanism.ipynb": figure1_notebook(),
        "02_figure2_transport_phenotype.ipynb": figure2_notebook(),
        "03_figure3_order_transport.ipynb": figure3_notebook(),
        "04_figure4_killed_transport.ipynb": figure4_notebook(),
        "05_figure5_state_structured_survival.ipynb": figure5_notebook(),
        "06_supplementary_material.ipynb": supplement_notebook(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in notebooks.items():
        write_notebook(OUT_DIR / name, cells)
    (OUT_DIR / "README.md").write_text(readme_text(), encoding="utf-8")
    print(f"publication_notebooks={OUT_DIR}")
    print(f"notebook_count={len(notebooks)}")


if __name__ == "__main__":
    main()
