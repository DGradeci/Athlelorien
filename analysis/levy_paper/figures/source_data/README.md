# Figure Source Data

This folder contains small CSV/JSON/MD files supporting the frozen paper
figures. These files are intended to be tracked in Git.

The GitHub-only workflow is:

1. Open `analysis/levy_paper/notebooks_publication/`.
2. Set `DATA_MODE = "reviewer"` or leave the default `DATA_MODE = "auto"`.
3. Inspect the frozen figure assets and source-data tables.

Full computational regeneration from one-second trajectory/run caches uses the
ignored local cache folders under `analysis/levy_paper/data/processed/`.
Rebuilding those caches from raw Soccermon data requires AWS access.

These files are aggregate, author-created analytical outputs and are available
under CC BY 4.0. They do not include raw Soccermon records, player names,
device identifiers, raw GPS coordinates, or individual trajectories. See the
repository root `DATA_AND_ASSET_LICENSING.md` for the complete scope.

Tracked source-data folders:

- `figure1_transport_mechanism/`
- `figure2_transport_phenotype/`
- `figure3_order_transport/`
- Figure 4/5 source tables at the top level.
