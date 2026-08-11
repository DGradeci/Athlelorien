# Publication Reproducibility Guide

This guide separates three usage paths for the football collective-transport paper.

## Path 1: Browse / Paper Mode

No AWS credentials and no parquet caches are required. A fresh GitHub clone
contains the frozen figure assets and the small source-data/audit files needed
to inspect the plotted quantities.

Open:

```text
analysis/levy_paper/notebooks_publication/
```

Run:

```text
00_data_access_and_cache.ipynb
01_figure1_transport_mechanism.ipynb
02_figure2_transport_phenotype.ipynb
03_figure3_order_transport.ipynb
04_figure4_killed_transport.ipynb
05_figure5_state_structured_survival.ipynb
06_supplementary_material.ipynb
```

The notebooks default to constructing a live figure from available inputs.
Figures 4, 5, S1 and S2 use tracked source-data tables. Figures 1-3 use the
optional processed cache and otherwise display frozen local assets from:

```text
analysis/levy_paper/figures/main_figures/
analysis/levy_paper/figures/supplementary_material/
analysis/levy_paper/figures/source_data/
```

This path reproduces the submitted figure files and exposes the source tables
used to audit the panels. Public mode does not recompute run segmentation or
refit hazard models from raw tracking data.

Expected Git-tracked ingredients:

```text
analysis/levy_paper/notebooks_publication/
analysis/levy_paper/figures/main_figures/
analysis/levy_paper/figures/supplementary_material/
analysis/levy_paper/figures/source_data/
analysis/levy_paper/data/cache_bundle_manifest.csv
analysis/levy_paper/metadata/schedules/README.md
analysis/levy_paper/metadata/pitches/README.md
analysis/levy_paper/metadata/pitches/toppserien_pitches.json
```

## Path 2: Cache Mode

AWS is not required, but the processed parquet/model caches must be present.
These caches are intentionally ignored by Git because they are large.

The expected external cache bundle is listed in:

```text
analysis/levy_paper/data/cache_bundle_manifest.csv
```

The bundle is distributed separately because the processed parquets are too
large for ordinary Git. Obtain the frozen bundle from the project maintainers,
extract it into the repository root, and preserve the relative paths in the
manifest. Rebuilt caches from AWS may also be used; byte-for-byte identity is
not required for exploratory use.

Primary cache folder:

```text
analysis/levy_paper/data/processed/all_team_2020_2021_sticky_active/
```

The Figure 4/5 model cache is generated from the primary hazard intervals:

```text
analysis/levy_paper/data/processed/figure4_fig5_crossfitted/
```

Regenerate it with:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\build_figure4_figure5_model_cache.py
```

In the relevant publication notebook, set:

```python
DATA_MODE = "cache"
```

Figures 4/5 use an existing derived model cache when present. To refit all
leave-one-fixture-out models from the primary hazard intervals, also set:

```python
REFIT_FIGURE4_FIGURE5_MODELS = True
```

Leave:

```python
REBUILD_CACHE_FROM_AWS = False
```

The one-command equivalent is:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\reproduce_publication.py --mode cache
```

## Path 3: AWS Rebuild Mode

This route is intended for Athlelorien collaborators with authorised access.
Use it only when credentials and budget are available. The loader accepts the
standard AWS credential chain, a named `AWS_PROFILE`, or explicit environment
keys. A `.env` file is optional.

Required local files:

```text
analysis/levy_paper/metadata/schedules/kamper_2020.xlsx
analysis/levy_paper/metadata/schedules/kamper_2021.xlsx
analysis/levy_paper/metadata/pitches/toppserien_pitches.json
```

The two NFF workbooks are local-only third-party inputs and are not included in
the public Git clone. See `analysis/levy_paper/metadata/schedules/README.md` for
their source, filenames, and required columns. Their absence does not affect
the browse/reviewer or cache pathways.

If used, create `.env` from the tracked `.env.example`. It must stay local and
must not be committed. At minimum, configure either a valid AWS profile or an
access-key/secret-key pair. Temporary credentials may also provide
`AWS_SESSION_TOKEN`. The default region is `eu-west-2` and the default bucket
is `ucl-ai-soccormon-dataset`.

The collaborator identity needs permission to list the bucket and read objects
below the configured 2020/2021 `objective_TEAM_A`, `objective_team_B`,
`objective_Team_A`, and `objective_Team_B` prefixes. Account policies or budget
controls may still deny access even when credentials are valid.

In:

```text
analysis/levy_paper/notebooks_publication/00_data_access_and_cache.ipynb
```

set:

```python
REBUILD_CACHE_FROM_AWS = True
```

Or run the full primary-cache and figure chain directly:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\reproduce_publication.py --mode aws
```

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with
`.venv/bin/python` in all commands in this guide.

## Validation

Run:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\check_publication_reproducibility.py
```

Outputs:

```text
analysis/levy_paper/reports/publication_reproducibility_check.csv
analysis/levy_paper/reports/publication_reproducibility_check.json
```

Important summary flags:

- `github_only_reviewer_ready`: frozen figures, source data, supplement assets
  and publication notebooks are present.
- `local_cache_regeneration_ready`: `github_only_reviewer_ready` plus optional
  processed/model caches are present.
- `aws_code_path_ready`: tracked code, documentation, and pitch registry needed
  by the AWS pathway are present.
- `local_schedule_inputs_present`: both local NFF workbooks are available.
- `aws_rebuild_ready_except_credentials`: the AWS code path and local schedule
  inputs are ready; AWS credentials are still local-only.

For the strict reviewer-mode smoke test, run:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\run_cleanroom_reviewer_test.py
```

This creates a temporary copy from Git-visible files only, checks that no
`.env`, `.venv`, processed cache, or large data artifact is present, and
executes all publication notebooks with no processed cache. Cache-dependent
figures use their explicit frozen fallback; source-table figures are rebuilt.

To test local-cache regeneration without AWS:

```powershell
.\.venv\Scripts\python.exe analysis\levy_paper\scripts\run_cleanroom_reviewer_test.py --mode local-cache --timeout 600
```

## Folder Policy

- `figures/main_figures/`: publication figures.
- `figures/supplementary_material/`: formal supplement package with Figures S1-S2 and Tables S1-S4.
- `figures/source_data/`: small source-data tables and audits for final figure assets.
- `data/processed/`: local processed caches, not tracked in Git.
- `outputs/`, `archive/`, `legacy/`: exploratory or historical generated outputs.
