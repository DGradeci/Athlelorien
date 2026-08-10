from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LEVY_DIR = ROOT / "analysis" / "levy_paper"
SOURCE_DIR = LEVY_DIR / "final_minimal_supplement_2026-07-30"
DEST_DIR = LEVY_DIR / "figures" / "supplementary_material"


@dataclass(frozen=True)
class SupplementAsset:
    asset_id: str
    role: str
    file_type: str
    source_rel: str
    notes_code_only: str = ""

    @property
    def source_path(self) -> Path:
        return SOURCE_DIR / self.source_rel

    @property
    def destination_path(self) -> Path:
        return DEST_DIR / self.source_rel


ASSETS = [
    SupplementAsset("figureS1_duration_model_comparison_png", "publication", "figure_png", "figures/figureS1_duration_model_comparison.png"),
    SupplementAsset("figureS1_duration_model_comparison_pdf", "publication", "figure_pdf", "figures/figureS1_duration_model_comparison.pdf"),
    SupplementAsset("figureS1_duration_model_comparison_svg", "publication", "figure_svg", "figures/figureS1_duration_model_comparison.svg"),
    SupplementAsset("tableS1_dataset_processing_csv", "publication", "table_csv", "tables/tableS1_dataset_processing_final.csv"),
    SupplementAsset("tableS1_dataset_processing_tex", "publication", "table_tex", "tables/tableS1_dataset_processing_final.tex"),
    SupplementAsset("tableS2_models_and_robustness_csv", "publication", "table_csv", "tables/tableS2_models_and_robustness_final.csv"),
    SupplementAsset("tableS2_models_and_robustness_tex", "publication", "table_tex", "tables/tableS2_models_and_robustness_final.tex"),
    SupplementAsset("tableS3_transition_matrices_csv", "publication", "table_csv", "tables/tableS3_transition_matrices_final.csv"),
    SupplementAsset("tableS3_transition_matrices_tex", "publication", "table_tex", "tables/tableS3_transition_matrices_final.tex"),
    SupplementAsset("figureS1_panelA_model_curves", "source_data", "csv", "source_data/figureS1_panelA_model_curves.csv"),
    SupplementAsset("figureS1_panelB_lofo_contrasts", "source_data", "csv", "source_data/figureS1_panelB_lofo_contrasts.csv"),
    SupplementAsset("duration_model_parameters", "source_data", "csv", "source_data/duration_model_parameters_final.csv"),
    SupplementAsset("duration_model_gof", "source_data", "csv", "source_data/duration_model_gof_final.csv"),
    SupplementAsset("tableS1_dataset_processing_source", "source_data", "csv", "source_data/tableS1_dataset_processing_source.csv"),
    SupplementAsset("tableS2_models_and_robustness_source", "source_data", "csv", "source_data/tableS2_models_and_robustness_source.csv"),
    SupplementAsset("tableS3_direct_transition_counts", "source_data", "csv", "source_data/tableS3_direct_transition_counts.csv"),
    SupplementAsset("tableS3_unrounded_probabilities", "source_data", "csv", "source_data/tableS3_unrounded_probabilities.csv"),
    SupplementAsset("final_supplement_publication_manifest", "report", "csv", "reports/final_supplement_publication_manifest.csv"),
    SupplementAsset("final_supplement_archive_manifest", "report", "csv", "reports/final_supplement_archive_manifest.csv"),
    SupplementAsset("figureS2_archive_manifest", "archive_only", "csv", "reports/figureS2_archive_manifest.csv", "Figure S2 is archive-only, not publication-facing."),
    SupplementAsset("final_minimal_supplement_checksums", "report", "csv", "reports/final_minimal_supplement_checksums.csv"),
    SupplementAsset("tableS3_row_sum_audit", "report", "csv", "reports/tableS3_row_sum_audit.csv"),
    SupplementAsset("manuscript_supplement_reference_inventory", "report", "csv", "reports/manuscript_supplement_reference_inventory.csv"),
    SupplementAsset("final_minimal_supplement_data_report", "report", "md", "reports/FINAL_MINIMAL_SUPPLEMENT_DATA_REPORT.md"),
]


def rel_to_root(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def copy_asset(asset: SupplementAsset) -> dict[str, str]:
    status = "missing_source"
    if asset.source_path.exists():
        asset.destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset.source_path, asset.destination_path)
        status = "copied"
    return {
        "asset_id": asset.asset_id,
        "role": asset.role,
        "file_type": asset.file_type,
        "path": rel_to_root(asset.destination_path),
        "source_path": rel_to_root(asset.source_path),
        "status": status,
        "notes_code_only": asset.notes_code_only,
    }


def copy_archive_only_tree() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source_archive = SOURCE_DIR / "archive_only"
    if not source_archive.exists():
        return rows
    for source_file in sorted(source_archive.rglob("*")):
        if not source_file.is_file():
            continue
        destination = DEST_DIR / "archive_only" / source_file.relative_to(source_archive)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
        rows.append(
            {
                "asset_id": f"archive_only_{source_file.stem}",
                "role": "archive_only",
                "file_type": source_file.suffix.lstrip(".").lower(),
                "path": rel_to_root(destination),
                "source_path": rel_to_root(source_file),
                "status": "copied",
                "notes_code_only": "Archive-only reviewer-defence material; not publication-facing.",
            }
        )
    return rows


def write_readme(rows: list[dict[str, str]]) -> None:
    publication = [row for row in rows if row["role"] == "publication" and row["status"] == "copied"]
    archive = [row for row in rows if row["role"] == "archive_only" and row["status"] == "copied"]
    text = "\n".join(
        [
            "# Supplementary Material",
            "",
            "Formal frozen supplementary package copied from `analysis/levy_paper/final_minimal_supplement_2026-07-30`.",
            "",
            "Publication-facing supplement assets are in `figures/` and `tables/`.",
            "Source data and checks are in `source_data/` and `reports/`.",
            "Reviewer-defence material not intended as publication-facing supplement is in `archive_only/`.",
            "",
            f"Publication-facing files: {len(publication)}",
            f"Archive-only files: {len(archive)}",
            "",
            "See `supplementary_material_manifest.csv` for exact paths and roles.",
            "",
        ]
    )
    (DEST_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    if not SOURCE_DIR.exists():
        manifest = DEST_DIR / "supplementary_material_manifest.csv"
        if manifest.exists():
            with manifest.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            source_rel = "source_data/figureS2_segmentation_robustness_source.csv"
            if not any(row.get("path", "").endswith(source_rel) for row in rows):
                rows.append(
                    {
                        "asset_id": "figureS2_segmentation_robustness_source",
                        "role": "source_data",
                        "file_type": "csv",
                        "path": rel_to_root(DEST_DIR / source_rel),
                        "source_path": "",
                        "status": "present" if (DEST_DIR / source_rel).exists() else "missing",
                        "notes_code_only": "Tracked source for create_supplementary_figures_from_source.py",
                    }
                )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["asset_id", "role", "file_type", "path", "source_path", "status", "notes_code_only"])
                writer.writeheader()
                writer.writerows(rows)
            write_readme(rows)
            print(f"Refreshed existing formal supplement manifest: {manifest}")
            return
        raise FileNotFoundError(f"Missing source supplement package: {SOURCE_DIR}")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    rows = [copy_asset(asset) for asset in ASSETS]
    rows.extend(copy_archive_only_tree())
    manifest_path = DEST_DIR / "supplementary_material_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["asset_id", "role", "file_type", "path", "source_path", "status", "notes_code_only"],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_readme(rows)
    copied = sum(row["status"] == "copied" for row in rows)
    missing = sum(row["status"] != "copied" for row in rows)
    print(f"supplementary_material_dir={DEST_DIR}")
    print(f"manifest={manifest_path}")
    print(f"copied={copied}")
    print(f"missing={missing}")


if __name__ == "__main__":
    main()
