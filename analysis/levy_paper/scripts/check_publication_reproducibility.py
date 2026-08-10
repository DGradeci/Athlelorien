from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
LEVY_DIR = ROOT / "analysis" / "levy_paper"
DATA_DIR = LEVY_DIR / "data"
METADATA_DIR = LEVY_DIR / "metadata"
NOTEBOOK_DIR = LEVY_DIR / "notebooks_publication"
FINAL_FIGURES = LEVY_DIR / "figures" / "main_figures"
SUPPLEMENT = LEVY_DIR / "figures" / "supplementary_material"
REPORT_DIR = LEVY_DIR / "reports"


EXPECTED_NOTEBOOKS = [
    "00_data_access_and_cache.ipynb",
    "01_figure1_transport_mechanism.ipynb",
    "02_figure2_transport_phenotype.ipynb",
    "03_figure3_order_transport.ipynb",
    "04_figure4_killed_transport.ipynb",
    "05_figure5_state_structured_survival.ipynb",
    "06_supplementary_material.ipynb",
]

EXPECTED_REPRO_DOCS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CITATION.cff",
    ROOT / "LICENSE",
    ROOT / "DATA_AND_ASSET_LICENSING.md",
    ROOT / ".env.example",
    LEVY_DIR / "README.md",
    LEVY_DIR / "PUBLICATION_GIT_SURFACE.md",
    LEVY_DIR / "PUBLICATION_REPRODUCIBILITY.md",
    LEVY_DIR / "README_CLEAN_STRUCTURE.md",
    NOTEBOOK_DIR / "README.md",
    DATA_DIR / "README.md",
    DATA_DIR / "cache_bundle_manifest.csv",
    LEVY_DIR / "figures" / "README.md",
    LEVY_DIR / "figures" / "source_data" / "README.md",
    METADATA_DIR / "schedules" / "README.md",
    METADATA_DIR / "pitches" / "README.md",
]

EXPECTED_AWS_CODE_INPUTS = [
    METADATA_DIR / "pitches" / "toppserien_pitches.json",
]

LOCAL_SCHEDULE_INPUTS = [
    METADATA_DIR / "schedules" / "kamper_2020.xlsx",
    METADATA_DIR / "schedules" / "kamper_2021.xlsx",
]

EXPECTED_FINAL_ASSETS = [
    FINAL_FIGURES / "figure1_transport_mechanism_schematic.png",
    FINAL_FIGURES / "figure1_transport_mechanism_schematic.pdf",
    FINAL_FIGURES / "figure2_transport_phenotype.png",
    FINAL_FIGURES / "figure2_transport_phenotype.pdf",
    FINAL_FIGURES / "figure3_order_transport.png",
    FINAL_FIGURES / "figure3_order_transport.pdf",
    FINAL_FIGURES / "figure4_killed_transport.png",
    FINAL_FIGURES / "figure4_killed_transport.pdf",
    FINAL_FIGURES / "figure5_state_structured_survival.png",
    FINAL_FIGURES / "figure5_state_structured_survival.pdf",
    FINAL_FIGURES / "final_figure_manifest.csv",
]

EXPECTED_SUPPLEMENT_ASSETS = [
    SUPPLEMENT / "figures" / "figureS1_duration_model_comparison.png",
    SUPPLEMENT / "figures" / "figureS1_duration_model_comparison.pdf",
    SUPPLEMENT / "figures" / "figureS2_segmentation_robustness_final.png",
    SUPPLEMENT / "figures" / "figureS2_segmentation_robustness_final.pdf",
    SUPPLEMENT / "tables" / "tableS1_dataset_processing_final.csv",
    SUPPLEMENT / "tables" / "tableS2_models_and_robustness_final.csv",
    SUPPLEMENT / "tables" / "tableS3_transition_matrices_final.csv",
    SUPPLEMENT / "tables" / "tableS4_active_player_reconstruction_parameters.csv",
    SUPPLEMENT / "supplementary_material_manifest.csv",
]

EXPECTED_SOURCE_DATA_ASSETS = [
    LEVY_DIR / "figures" / "source_data" / "figure1_transport_mechanism" / "real_match_20s_player_centroid_trajectories_audit.csv",
    LEVY_DIR / "figures" / "source_data" / "figure1_transport_mechanism" / "real_match_transport_mechanism_4panel_audit.json",
    LEVY_DIR / "figures" / "source_data" / "figure2_transport_phenotype" / "figure2_panelA_duration_survivor_audit.csv",
    LEVY_DIR / "figures" / "source_data" / "figure2_transport_phenotype" / "figure2_panelB_length_survivor_audit.csv",
    LEVY_DIR / "figures" / "source_data" / "figure2_transport_phenotype" / "figure2_panelC_msd_aggregate_audit.csv",
    LEVY_DIR / "figures" / "source_data" / "figure2_transport_phenotype" / "figure2_panelD_msd_decomposition_components.csv",
    LEVY_DIR / "figures" / "source_data" / "figure3_order_transport" / "figure3_order_transport_ccdf_audit.csv",
    LEVY_DIR / "figures" / "source_data" / "figure3_order_transport" / "figure3_order_transport_manifest.json",
    LEVY_DIR / "figures" / "source_data" / "figure4A_source_data.csv",
    LEVY_DIR / "figures" / "source_data" / "figure4B_source_data.csv",
    LEVY_DIR / "figures" / "source_data" / "figure4C_source_data.csv",
    LEVY_DIR / "figures" / "source_data" / "figure5A_source_data.csv",
    LEVY_DIR / "figures" / "source_data" / "figure5B_source_data.csv",
    LEVY_DIR / "figures" / "source_data" / "figure5C_source_data.csv",
]

PRIMARY_CACHE_DIR = LEVY_DIR / "data" / "processed" / "all_team_2020_2021_sticky_active"
PRIMARY_CACHE_FILES = [
    PRIMARY_CACHE_DIR / name
    for name in [
        "trajectory_long_2020_2021_all_teams_pitchfix_sticky_active.parquet",
        "runs_long_2020_2021_all_teams_pitchfix_sticky_active.parquet",
        "df_pmv_2020_2021_all_teams_pitchfix_sticky_active.parquet",
        "msd_long_2020_2021_all_teams_pitchfix_sticky_active.parquet",
        "centroid_order_runs_2020_2021_all_teams_pitchfix_sticky_active.parquet",
        "hazard_intervals_2020_2021_all_teams_pitchfix_sticky_active.parquet",
    ]
]

FIGURE_MODEL_CACHE_FILES = [
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "figure4_panelA_hazard_audit.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "figure5_panelA_selected_transitions_crossfit_package.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "figure5_crossfitted_panelB_current_high_fraction.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "figure5_crossfitted_panelC_survival.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "figure5_crossfitted_panelD_high_exposure_counterfactuals.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "fold_summary.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "state_thresholds.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "pi0.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "out_of_fold_interval_predictions.parquet",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "fold_curves.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "transition_matrices.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "hazard_parameters.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "crossfitted_validation_metrics.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "crossfitted_fixture_level_curve_metrics.csv",
    LEVY_DIR / "data" / "processed" / "figure4_fig5_crossfitted" / "robustness_fold_curves.csv",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def path_row(path: Path, group: str) -> dict:
    return {
        "group": group,
        "path": rel(path),
        "exists": bool(path.exists()),
        "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
    }


def validate_notebook(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        return True, f"valid_cells={len(nb.cells)}"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def manifest_paths(path: Path, *, exclude_archive_only: bool = False) -> list[Path]:
    if not path.exists():
        return []
    try:
        manifest = pd.read_csv(path)
    except Exception:
        return []
    if exclude_archive_only and "role" in manifest.columns:
        manifest = manifest[~manifest["role"].astype(str).eq("archive_only")]
    if "path" not in manifest.columns:
        return []
    paths: list[Path] = []
    for value in manifest["path"].dropna().astype(str).unique():
        candidate = Path(value)
        paths.append(candidate if candidate.is_absolute() else ROOT / candidate)
    return paths


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in EXPECTED_REPRO_DOCS:
        row = path_row(path, "reproducibility_doc")
        row["valid_notebook"] = None
        row["notes_code_only"] = "required for public reviewer/collaborator orientation"
        rows.append(row)
    for path in EXPECTED_AWS_CODE_INPUTS:
        row = path_row(path, "aws_code_input")
        row["valid_notebook"] = None
        row["notes_code_only"] = "tracked input required only for AWS raw-data rebuild path"
        rows.append(row)
    for path in LOCAL_SCHEDULE_INPUTS:
        row = path_row(path, "local_schedule_input")
        row["valid_notebook"] = None
        row["notes_code_only"] = "local-only NFF input; not required for reviewer or cache mode"
        rows.append(row)
    for name in EXPECTED_NOTEBOOKS:
        path = NOTEBOOK_DIR / name
        valid, note = validate_notebook(path)
        rows.append({
            "group": "publication_notebook",
            "path": rel(path),
            "exists": bool(path.exists()),
            "bytes": int(path.stat().st_size) if path.exists() else None,
            "valid_notebook": valid,
            "notes_code_only": note,
        })
    for path in EXPECTED_FINAL_ASSETS:
        row = path_row(path, "final_figure_asset")
        row["valid_notebook"] = None
        row["notes_code_only"] = ""
        rows.append(row)
    for path in EXPECTED_SUPPLEMENT_ASSETS:
        row = path_row(path, "supplement_asset")
        row["valid_notebook"] = None
        row["notes_code_only"] = ""
        rows.append(row)
    for path in manifest_paths(SUPPLEMENT / "supplementary_material_manifest.csv", exclude_archive_only=True):
        row = path_row(path, "supplement_manifest_asset")
        row["valid_notebook"] = None
        row["notes_code_only"] = "listed in public supplement manifest"
        rows.append(row)
    for path in EXPECTED_SOURCE_DATA_ASSETS:
        row = path_row(path, "tracked_source_data")
        row["valid_notebook"] = None
        row["notes_code_only"] = "required for GitHub-only figure inspection/source-data audit"
        rows.append(row)
    for path in manifest_paths(LEVY_DIR / "figures" / "source_data" / "source_data_manifest.csv"):
        row = path_row(path, "tracked_source_data_manifest_asset")
        row["valid_notebook"] = None
        row["notes_code_only"] = "listed in source-data manifest"
        rows.append(row)
    for path in PRIMARY_CACHE_FILES:
        row = path_row(path, "optional_local_cache")
        row["valid_notebook"] = None
        row["notes_code_only"] = "optional for regeneration; not required for frozen inspection"
        rows.append(row)
    for path in FIGURE_MODEL_CACHE_FILES:
        row = path_row(path, "optional_model_cache")
        row["valid_notebook"] = None
        row["notes_code_only"] = "optional for Figure 4/5 regeneration; not required for frozen inspection"
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = REPORT_DIR / "publication_reproducibility_check.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "publication_notebook_dir": rel(NOTEBOOK_DIR),
        "all_reproducibility_docs_exist": bool(df[df["group"].eq("reproducibility_doc")]["exists"].all()),
        "all_aws_code_inputs_exist": bool(df[df["group"].eq("aws_code_input")]["exists"].all()),
        "local_schedule_inputs_present": bool(df[df["group"].eq("local_schedule_input")]["exists"].all()),
        "all_expected_notebooks_exist": bool(df[df["group"].eq("publication_notebook")]["exists"].all()),
        "all_expected_notebooks_valid": bool(df[df["group"].eq("publication_notebook")]["valid_notebook"].all()),
        "all_final_figure_assets_exist": bool(df[df["group"].eq("final_figure_asset")]["exists"].all()),
        "all_supplement_assets_exist": bool(df[df["group"].eq("supplement_asset")]["exists"].all()),
        "all_public_supplement_manifest_assets_exist": bool(df[df["group"].eq("supplement_manifest_asset")]["exists"].all()),
        "all_tracked_source_data_exist": bool(df[df["group"].eq("tracked_source_data")]["exists"].all()),
        "all_source_data_manifest_assets_exist": bool(df[df["group"].eq("tracked_source_data_manifest_asset")]["exists"].all()),
        "optional_local_cache_complete": bool(df[df["group"].eq("optional_local_cache")]["exists"].all()),
        "optional_model_cache_complete": bool(df[df["group"].eq("optional_model_cache")]["exists"].all()),
        "csv_report": rel(csv_path),
    }
    summary["github_only_reviewer_ready"] = bool(
        summary["all_reproducibility_docs_exist"]
        and summary["all_expected_notebooks_exist"]
        and summary["all_expected_notebooks_valid"]
        and summary["all_final_figure_assets_exist"]
        and summary["all_supplement_assets_exist"]
        and summary["all_public_supplement_manifest_assets_exist"]
        and summary["all_tracked_source_data_exist"]
        and summary["all_source_data_manifest_assets_exist"]
    )
    summary["model_cache_can_be_rebuilt_from_primary"] = bool(summary["optional_local_cache_complete"])
    summary["local_cache_regeneration_ready"] = bool(
        summary["github_only_reviewer_ready"] and summary["optional_local_cache_complete"]
    )
    summary["aws_code_path_ready"] = bool(
        summary["all_reproducibility_docs_exist"]
        and summary["all_aws_code_inputs_exist"]
    )
    summary["aws_rebuild_ready_except_credentials"] = bool(
        summary["aws_code_path_ready"]
        and summary["local_schedule_inputs_present"]
    )
    json_path = REPORT_DIR / "publication_reproducibility_check.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
