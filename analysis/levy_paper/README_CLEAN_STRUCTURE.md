# Levy Paper Clean Structure

Use these folders for the publication/reviewer-facing paper repo:

- `notebooks_publication/`: canonical publication notebooks in run order.
- `notebooks/`: ignored local/exploratory notebook history; not part of the public run path.
- `scripts/`: reusable build/sync scripts.
- `data/processed/all_team_2020_2021_sticky_active/`: current all-team processed caches.
- `data/processed/h2h_2020_2021_sticky_active/`: current two-team/H2H processed caches.
- `data/processed/figure4_fig5_crossfitted/`: derived Figure 4/5 model cache; rebuildable from the primary hazard intervals.
- `figures/main_figures/`: current paper figures, PNG and PDF.
- `figures/supplementary_figures/`: current supplementary figures, PNG and PDF.
- `figures/source_data/`: source CSVs, manifests, captions, and figure audits.
- `figures/supplementary_material/`: formal supplement figures, tables and source data.
- `outputs/`: ignored local regenerated outputs, not required for GitHub-only inspection.
- `reports/`: ignored local cleanup/reproducibility reports.

Deletion candidates after the copy/reorg pass are listed in:

- `reports/repo_deletion_candidates_after_logical_reorg_2026-07-26.csv`

The deletion candidates are non-empty folders. They were not removed automatically because they include large historical archives and superseded output trees.

For a GitHub clone without AWS, the required paper-inspection surface is:

- `notebooks_publication/`
- `figures/main_figures/`
- `figures/supplementary_material/`
- `figures/source_data/`

The `data/processed/` folders are optional local cache bundles for full
computational regeneration without AWS.

## Cleanup Safety Audit, 2026-07-31

The publication reproducibility check passed after the clean-structure pass:

- all publication notebooks exist and are valid
- all final main figure assets exist
- all formal supplement assets exist
- all tracked figure source-data assets exist
- optional local processed cache is complete
- optional Figure 4/5 model cache is complete

Required keep set:

- `analysis/levy_paper/notebooks_publication/`
- selected reusable scripts under `analysis/levy_paper/scripts/`
- `analysis/levy_paper/util/`
- `analysis/levy_paper/figures/main_figures/`
- `analysis/levy_paper/figures/supplementary_material/`
- `analysis/levy_paper/figures/source_data/`
- `analysis/levy_paper/data/processed/`
- `analysis/levy_paper/data/README.md`

The following folders were checked as historical/generated cleanup candidates.
They are not part of the minimal publication inspection path:

- `analysis/levy_paper/archive/`
- `analysis/levy_paper/legacy/`
- `analysis/levy_paper/analysis/`
- `analysis/levy_paper/outputs/`
- `analysis/levy_paper/final_freeze_2026-07-30/`
- `analysis/levy_paper/final_paper_completion_2026-07-27/`
- `analysis/levy_paper/final_minimal_supplement_2026-07-30/`
- `analysis/levy_paper/minimal_final_supplement_2026-07-28/`
- `analysis/levy_paper/minimal_supplement_2026-07-28/`
- `analysis/levy_paper/supplement_minimal_final_2026-07-29/`
- `analysis/levy_paper/supplement_repair_2026-07-28/`
- `analysis/levy_paper/chatgpt_share*/`
- `analysis/levy_paper/final_fig2_update_2026-07-28/`
- root-level `outputs/`, `chatgpt_share/`, `tail_audit/`, `final_code_audit/`,
  `final_order_speed_audit/`, and `final_transition_sync/`

Notes:

- `analysis/levy_paper/scripts/create_figure2_transport_phenotype.py` now uses
  the canonical processed cache path instead of the old flat data files.
- `analysis/levy_paper/scripts/create_final_figure2_transport_phenotype.py`
  falls back to formal supplement source data if root-level `tail_audit/` is
  absent.
- `analysis/levy_paper/scripts/sync_formal_supplementary_material.py` keeps the
  existing formal supplement if the old frozen source package is absent.
- `analysis/levy_paper/scripts/sync_final_figures.py` copies from generated
  `outputs/final/` folders after regeneration; current publication-ready copies
  already live under `figures/main_figures/`.

Cleanup execution completed on 2026-07-31:

- historical/generated cleanup folders listed above were removed locally
- duplicate flat files in `analysis/levy_paper/data/` were removed locally
- `analysis/levy_paper/data/processed/` and `analysis/levy_paper/data/README.md`
  were preserved
- post-cleanup reproducibility check passed
- post-cleanup `analysis/levy_paper/` size was approximately 0.52 GB

Cleanup reports are under `analysis/levy_paper/reports/`.
