# Publication Reproduction Notebooks

Run the notebooks in numerical order. They follow the same visible pattern: define the scientific question, declare the data pathway, inspect the input population, display intermediate numerical tables, and finally call the authoritative publication plotting function.

Notebook 00 selects seasons, tracked sources, and optional match dates and can rebuild the processed cache for authorised AWS users. Notebooks 01-03 reconstruct their analyses and figures from that processed cache. Notebooks 04-05 derive their panels from the processed cross-fitted model cache, with optional refitting from hazard intervals; without that cache they use tracked publication source tables. Notebook 06 reconstructs the supplement from tracked source tables.

Every figure notebook ends by displaying the Matplotlib figure it constructed. When a required private processed cache is absent, the notebook reports that fact and displays the tracked frozen publication figure instead.

The three supported paths are:

1. Public/reviewer: set `DATA_MODE = "reviewer"`; tracked source tables and frozen figures require no AWS.
2. Local cache: install the external cache under `analysis/levy_paper/data/processed/all_team_2020_2021_sticky_active/` and set `DATA_MODE = "cache"`.
3. AWS collaborator: supply the local NFF workbooks documented in `../metadata/schedules/README.md`, configure credentials, edit the selection in notebook 00, set `REBUILD_CACHE_FROM_AWS = True`, build a named cache, and then use that cache in the figure notebooks.

`DATA_MODE = "auto"` is the default. It uses the processed cache only when every required file exists and otherwise selects the reviewer pathway. `REFIT_FIGURE4_FIGURE5_MODELS = False` reuses the audited cross-fitted model cache; changing it to `True` performs the expensive local refit from hazard intervals without contacting AWS.

The schedule workbooks are not needed in reviewer or local-cache mode and are not redistributed in the public repository.
