# Athlelorien

Football tracking analysis for the collective transport / team-run paper.

This repository contains the reusable utilities, notebooks and figure-building
scripts used to process Soccermon tracking data, reconstruct active on-pitch
players, build team-centroid transport statistics, and reproduce the current
paper figures.

## Quick Start

Use Python 3.11 for the checked publication environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For the exact Python 3.11 environment used in the final reproducibility check,
install `requirements-lock.txt` instead.

AWS is optional. Internal collaborators may use their existing AWS environment,
shared credentials file, named profile, or a local `.env` based on
`.env.example`. The `.env` file is ignored by Git.

## Paper Analysis

The main analysis lives in:

```text
analysis/levy_paper/
```

For new collaborators, start with the publication-facing notebooks:

```text
analysis/levy_paper/notebooks_publication/
```

These notebooks construct live Matplotlib figures whenever their inputs are
available. Figures 4, 5 and the formal supplement rebuild from tracked source
tables without AWS; Figures 1-3 rebuild from the optional processed cache and
fall back to the tracked publication images when that cache is absent. Notebook
00 provides the explicit AWS/S3 cache-build path for authorised collaborators.

Publication run order:

1. `00_data_access_and_cache.ipynb` checks local assets/caches and optional AWS configuration.
2. `01_figure1_transport_mechanism.ipynb` inspects or rebuilds Figure 1.
3. `02_figure2_transport_phenotype.ipynb` inspects or rebuilds Figure 2.
4. `03_figure3_order_transport.ipynb` inspects or rebuilds Figure 3.
5. `04_figure4_killed_transport.ipynb` inspects or rebuilds Figure 4.
6. `05_figure5_state_structured_survival.ipynb` inspects or rebuilds Figure 5.
7. `06_supplementary_material.ipynb` inspects or rebuilds the formal supplementary figures from tracked source tables.

The lower-level implementation lives in `analysis/levy_paper/scripts/` and
`src/utils/`. Older exploratory notebooks are not part of the public run path.

## Three Ways To Use This Repo

The repository is intended to work through three deliberately separate paths:

| Path | Intended user | AWS needed | Local processed caches needed | What works |
| --- | --- | --- | --- | --- |
| Browse / paper mode | Casual readers, colleagues, reviewers | No | No | Open the final figures, supplement assets, source-data tables and audit manifests. |
| Cache mode | Colleagues/reviewers checking computations | No | Yes | Recompute figures from processed run/interval/model caches. |
| AWS rebuild mode | Athlelorien collaborators with data access | Yes | No | Supply local NFF schedule exports, rebuild processed caches from Soccermon/AWS using the tracked pitch registry, then regenerate figures. |

The GitHub clone includes the publication notebooks, frozen figure files,
aggregate source-data CSV/JSON/MD files, and attributed pitch registry. Original
NFF schedule exports, raw tracking data, and processed parquet caches are
intentionally excluded from Git. The expected optional cache bundle is listed in:

```text
analysis/levy_paper/data/cache_bundle_manifest.csv
```

The processed cache bundle is not hosted in this repository. Internal
collaborators should obtain the frozen bundle from the project maintainers and
extract it at the paths recorded in the manifest. Public users do not need the
bundle to inspect figures, source tables, or audit outputs.

Current exported paper figures are synced to:

```text
analysis/levy_paper/figures/main_figures/
analysis/levy_paper/figures/supplementary_material/
analysis/levy_paper/figures/source_data/
```

See `analysis/levy_paper/README.md` for the full cache and reproducibility
policy.

To validate a clone, run:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\check_publication_reproducibility.py
```

For a stricter pre-submission smoke test that builds a temporary Git-visible
copy with no `.env`, no `.venv`, and no processed caches, run:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\run_cleanroom_reviewer_test.py
```

To test figure regeneration from local processed/model caches without AWS, run:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\run_cleanroom_reviewer_test.py --mode local-cache --timeout 600
```

To rebuild every publication figure from the primary processed cache, including
refitting the 47 leave-one-fixture-out Figure 4/5 models, run:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\reproduce_publication.py --mode cache
```

Collaborators with AWS access can rebuild the primary cache first with:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\reproduce_publication.py --mode aws
```

Before using AWS mode, place the local NFF exports at the paths documented in
`analysis/levy_paper/metadata/schedules/README.md`. These third-party
spreadsheets are not redistributed by this repository.

The macOS/Linux equivalents use `.venv/bin/python` in place of
`.\.venv\Scripts\python.exe`.

## Citation And Licence

Citation metadata are provided in `CITATION.cff`. The associated paper is
"The life and death of football team runs: survival of collective modes shapes
Levy-like transport."

Original software is released under the MIT License in `LICENSE`. The authors'
aggregate figure assets and figure-source tables are released under CC BY 4.0.
Raw Soccermon data are excluded, original NFF schedule exports remain
local-only, and the pitch registry contains OpenStreetMap data under ODbL 1.0.
See `DATA_AND_ASSET_LICENSING.md` for the exact scope and attribution.

Contribution and validation conventions are documented in `CONTRIBUTING.md`.
