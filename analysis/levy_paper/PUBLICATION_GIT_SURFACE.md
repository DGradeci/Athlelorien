# Publication Git Surface

This note defines the intended public repository surface for the football
collective-transport paper.

## Intended user modes

1. GitHub-only inspection
   - No AWS credentials.
   - No processed parquet caches.
   - Rebuilds Figures 4, 5, S1 and S2 from tracked source-data CSVs.
   - Opens frozen Figures 1-3 alongside their tables and manifests.

2. Local-cache regeneration
   - No AWS credentials.
   - Requires the optional processed-cache bundle listed in
     `analysis/levy_paper/data/cache_bundle_manifest.csv`.
   - Rebuilds figures from processed run, order, hazard and model tables.

3. Raw AWS rebuild
   - Requires AWS credentials and S3 permissions.
   - Rebuilds processed caches from Soccermon/AWS using the schedule
     spreadsheets and pitch registry, then reruns downstream figure builders.

## Track in Git

Repository-level onboarding and release metadata:

```text
README.md
CONTRIBUTING.md
CITATION.cff
LICENSE
DATA_AND_ASSET_LICENSING.md
.env.example
requirements.txt
requirements-lock.txt
```

Track these files/directories:

- `README.md`
- `requirements.txt`
- `src/`
- `analysis/levy_paper/README.md`
- `analysis/levy_paper/PUBLICATION_REPRODUCIBILITY.md`
- `analysis/levy_paper/README_CLEAN_STRUCTURE.md`
- `analysis/levy_paper/notebooks_publication/`
- `analysis/levy_paper/scripts/build_multiseason_data_cache.py`
- `analysis/levy_paper/scripts/check_publication_reproducibility.py`
- `analysis/levy_paper/scripts/run_cleanroom_reviewer_test.py`
- `analysis/levy_paper/scripts/create_publication_reproduction_notebooks.py`
- `analysis/levy_paper/scripts/create_real_match_transport_mechanism_4panel.py`
- `analysis/levy_paper/scripts/create_figure2_transport_phenotype.py`
- `analysis/levy_paper/scripts/create_final_figure2_transport_phenotype.py`
- `analysis/levy_paper/scripts/create_final_figure3_order_transport.py`
- `analysis/levy_paper/scripts/create_final_fig4_fig5_polished.py`
- `analysis/levy_paper/scripts/run_pre_submission_robustness_audit.py`
- `analysis/levy_paper/scripts/sync_final_figures.py`
- `analysis/levy_paper/scripts/sync_formal_supplementary_material.py`
- `analysis/levy_paper/figures/main_figures/`
- `analysis/levy_paper/figures/supplementary_material/`
- `analysis/levy_paper/figures/source_data/`
- `analysis/levy_paper/data/README.md`
- `analysis/levy_paper/data/cache_bundle_manifest.csv`
- `analysis/levy_paper/metadata/schedules/README.md`
- `analysis/levy_paper/metadata/pitches/README.md`
- `analysis/levy_paper/metadata/pitches/toppserien_pitches.json`

The formal supplement folder should contain publication-facing Supplementary
Figures S1-S2, Tables S1-S4, and source-data/evidence files for the active-player
reconstruction. Temporary `chatgpt_share*` folders should not be tracked.

Do not track exploratory/development figure notebooks unless they are promoted
to the publication notebook set. The public run path should remain
`analysis/levy_paper/notebooks_publication/`.

## Do not track

- `.env`
- `.venv/`
- raw AWS/S3 tracking files
- original NFF schedule spreadsheet exports
- parquet caches under `analysis/levy_paper/data/processed/`
- video files and animations
- exploratory output trees under `outputs/`, `archive/`, `legacy/` or
  `analysis/levy_paper/outputs/`
- temporary ChatGPT share folders

## Raw AWS rebuild entry point

From `analysis/levy_paper/notebooks_publication/00_data_access_and_cache.ipynb`,
set:

```python
REBUILD_CACHE_FROM_AWS = True
```

The notebook calls:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\build_multiseason_data_cache.py
```

Required local files:

- `.env` containing AWS credentials;
- `analysis/levy_paper/metadata/schedules/kamper_2020.xlsx`;
- `analysis/levy_paper/metadata/schedules/kamper_2021.xlsx`;
- `analysis/levy_paper/metadata/pitches/toppserien_pitches.json`.

The `.env` file and NFF schedule workbooks must remain local and must not be
committed. Schedule acquisition, expected filenames, and required columns are
documented in `analysis/levy_paper/metadata/schedules/README.md`.
