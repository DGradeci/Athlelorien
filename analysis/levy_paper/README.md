# Levy Paper Analysis

This folder contains the cleaned analysis pipeline for the football collective
transport paper. It is organised so a new collaborator can:

1. reproduce and inspect the frozen paper figures from a GitHub clone,
2. regenerate figures from optional local processed caches, or
3. rebuild processed caches from raw Soccermon/AWS data when access is available.

## Current Canonical Structure

```text
analysis/levy_paper/
  notebooks_publication/     collaborator-facing notebooks for Figures 1-5 and supplement
  notebooks/                 ignored local/development notebooks not used as the public run path
  scripts/                   reusable cache and figure builders
  util/                      paper-specific plotting/loading helpers
  data/                      local processed caches; large files are not tracked
  figures/
    main_figures/            current manuscript/candidate figures, PNG and PDF
    supplementary_material/  formal supplement package: Figures S1-S2, Tables S1-S4 and source data
    supplementary_figures/   supplementary figures, PNG and PDF
    source_data/             small CSV/JSON/MD source data and audits
  outputs/final/             full generated figure-output trees
  reports/                   cleanup, methods and reproducibility reports
  archive/                   old exploratory outputs retained locally
```

## Run Order

Run publication notebooks from the repository root or from their own folder after
installing dependencies from `requirements.txt`.
Use `requirements-lock.txt` when an exact match to the final checked Python 3.11
environment is required.

1. `notebooks_publication/00_data_access_and_cache.ipynb`
   - Selects seasons, tracked sources and optional exact match dates.
   - Previews schedules, pitch metadata and local processed caches.
   - Provides an explicit opt-in cell for rebuilding caches from AWS/S3.
   - Defaults to an AWS-free GitHub mode.

2. `notebooks_publication/01_figure1_transport_mechanism.ipynb`
   - Rebuilds Figure 1 from the trajectory/run/order cache, with frozen fallback.

3. `notebooks_publication/02_figure2_transport_phenotype.ipynb`
   - Computes the survivor, MSD and decomposition panels and builds Figure 2.

4. `notebooks_publication/03_figure3_order_transport.ipynb`
   - Computes order terciles, bootstrap survivors and builds Figure 3.

5. `notebooks_publication/04_figure4_killed_transport.ipynb`
   - Builds Figure 4 from exact cross-fitted panel tables; refitting is optional.

6. `notebooks_publication/05_figure5_state_structured_survival.ipynb`
   - Builds Figure 5 from cross-fitted transition and survivor tables.

7. `notebooks_publication/06_supplementary_material.ipynb`
   - Rebuilds the formal supplementary figures from tracked source tables.

The public figure run path is `notebooks_publication/`. Lower-level working
notebooks under `notebooks/` are not required for a new collaborator to inspect
or reproduce the paper figures. The scripted AWS rebuild entry point is
`scripts/build_multiseason_data_cache.py`.

## Main Scripts

These scripts are the stable programmatic entry points behind the notebooks:

- `scripts/build_multiseason_data_cache.py`
- `scripts/build_figure4_figure5_model_cache.py`
- `scripts/reproduce_publication.py`
- `scripts/create_real_match_transport_mechanism_4panel.py`
- `scripts/create_figure2_transport_phenotype.py`
- `scripts/create_final_figure2_transport_phenotype.py`
- `scripts/create_final_figure3_order_transport.py`
- `scripts/create_final_fig4_fig5_polished.py`
- `scripts/create_supplementary_figures_from_source.py`
- `scripts/run_pre_submission_robustness_audit.py`
- `scripts/sync_final_figures.py`
- `scripts/sync_formal_supplementary_material.py`
- `scripts/sanitize_publication_metadata.py`
- `scripts/check_publication_reproducibility.py`
- `scripts/create_publication_reproduction_notebooks.py`

## Data And Cache Policy

Raw tracking files and large processed caches are not tracked in Git.

The intended public/reviewer clone contains enough material to inspect the
paper outputs and source-data audits, but not enough to rerun the expensive
raw-data processing. Exact computational regeneration without AWS requires the
optional processed-cache bundle listed in:

```text
data/cache_bundle_manifest.csv
```

Tracked or intended-to-track small reproducibility inputs:

- `metadata/pitches/toppserien_pitches.json`
- `metadata/pitches/README.md`
- `metadata/schedules/README.md`
- `data/cache_bundle_manifest.csv`
- figure manifests and source-data CSVs under `figures/source_data/`
- frozen publication figures under `figures/main_figures/`
- formal supplement assets under `figures/supplementary_material/`

Ignored local/heavy files:

- `.env`
- raw AWS/S3 data
- large parquet caches under `data/processed/`
- generated video files
- exploratory archived outputs

To rebuild the full analysis from raw data, a collaborator needs:

- Python dependencies from `requirements.txt`
- access to the Soccermon/AWS data source
- a local `.env` file with credentials
- local NFF schedule exports at the paths documented in
  `metadata/schedules/README.md`
- the tracked pitch registry above

To inspect/reproduce the submitted figure files without AWS, a collaborator
only needs the Git-tracked assets under:

```text
figures/main_figures/
figures/supplementary_material/
figures/source_data/
notebooks_publication/
```

This GitHub-only route displays and audits the frozen outputs. It does not
recompute player filtering, run segmentation, model fitting, or cross-fitted
validation from raw tracking data.

To recompute figure outputs from processed run/interval tables without AWS, a
collaborator needs the primary processed cache bundle corresponding to:

```text
data/processed/all_team_2020_2021_sticky_active/
```

The Figure 4/5 model cache is derived from the primary hazard-interval parquet
by `scripts/build_figure4_figure5_model_cache.py`; it no longer has to be
supplied as an independent external input.

Full cache-mode reproduction:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\reproduce_publication.py --mode cache
```

The current final all-team cache contains:

- 66 source/team match records
- 62 competitive fixtures across 47 match dates
- 69,802 centroid runs
- 355,330 one-second hazard intervals

Run the reproducibility check after cloning or after moving cache bundles:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\check_publication_reproducibility.py
```

## Final Figure Locations

Use these folders for sharing:

```text
figures/main_figures/
figures/supplementary_material/
figures/source_data/
```

The figure manifest is:

```text
figures/figure_manifest.csv
```

The current manuscript core is Figures 1-5. State-structured survival outputs
are retained in the final figure folder and formal supplementary material.

## Notes For New Users

- Start by opening `notebooks_publication/00_data_access_and_cache.ipynb`.
- If processed caches are already present, downstream figure notebooks can be
  run without accessing AWS.
- If a cache is missing, rebuild through `notebooks_publication/00_data_access_and_cache.ipynb`
  or `scripts/build_multiseason_data_cache.py`.
- Do not commit `.env` or raw tracking data.
- Do not commit the local NFF schedule exports.
- Keep new exploratory outputs in `outputs/` or `archive/`, not in the final
  figure folders, until they are promoted.
- See the root `DATA_AND_ASSET_LICENSING.md` for code, figure, Soccermon, NFF,
  and OpenStreetMap licensing boundaries.
