# Contributing

## Set up

Create a Python 3.11 environment and install `requirements.txt`. Use
`requirements-lock.txt` only when reproducing the exact checked environment.

## Work on the paper analysis

- Use `analysis/levy_paper/notebooks_publication/` as the supported notebook path.
- Put reusable data processing in `src/` and paper-specific modeling or plotting
  in `analysis/levy_paper/`.
- Keep exploratory outputs out of the final figure and source-data folders.
- Never commit `.env`, AWS credentials, raw tracking data, or processed parquet caches.
- Never commit original NFF schedule exports; keep them at the documented local paths.
- Preserve OpenStreetMap attribution when changing the pitch registry.

## Validate changes

Run:

```text
python -m pytest -q
python analysis/levy_paper/scripts/check_publication_reproducibility.py
python analysis/levy_paper/scripts/run_cleanroom_reviewer_test.py
```

For changes to cache-derived calculations, also run the local-cache clean-room
test described in `analysis/levy_paper/PUBLICATION_REPRODUCIBILITY.md`.

## Figures and metadata

Regenerate figures through the publication notebooks or
`analysis/levy_paper/scripts/reproduce_publication.py`. Do not manually edit
generated numerical source tables. Run `sanitize_publication_metadata.py`
before proposing changes to public manifests.

Review `DATA_AND_ASSET_LICENSING.md` before adding data, figures, or other
third-party material to the public Git surface.
