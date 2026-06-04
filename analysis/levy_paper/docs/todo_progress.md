# TODO Progress — Lévy Paper Analysis

Tracking how each item from the original notebook TODO list has been addressed
in this refactor, and what still needs doing.

---

## Original TODO List

> Copied verbatim from `notebooks/Levy_Paper_Analysis.ipynb` cell 0.

```
1. Add position labelling for players (probs easiest via distance to own goal at starting time of match)
2. Add 2021 data to AWS. Will need to find match schedule for 2021 as current kamper.xlsx is only for 2020 matches
3. Compile statistics from multiple matches (should have 4 matches across 2 years where teams go head to head)
4. Adding length CCDF to Markov model and stochastic models via centroid speed + duration of runs
5. Add variable to model truncation, distance/duration to field limits, opponent pressure, something like that...
6. Team coupling analysis: team polarisations are highly correlated, mess around with some 'modes'.
7. General refactoring
8. Feel free to improve animation at the end
```

---

## Status per item

---

### 1. Position labelling
**Status**: ✅ Code complete — needs data to validate

**What was done**:
`01_data_loading.ipynb` §3 now contains a `assign_position_labels()` function that:
- Takes the first `n_kickoff_frames` (default 5) of the first half for each player.
- Computes each player's median `x_m` (depth relative to pitch centre).
- Assigns one of four coarse labels — `GK/DEF`, `DEF/MID`, `MID/FWD`, `FWD` — by
  quartile of depth within the team.
- Adds a `position_label` column to every match's active-player dataframe.

**What it enables**:
- Condition analyses in notebooks 03 / 06 by position (e.g., does order matter
  more for midfielders or forwards?).
- Check whether centroid-dominated transport is stronger for certain lines.

**Outstanding**:
- [ ] Validate labels visually against known player positions for at least one match.
- [ ] Consider finer labelling if role data (formation sheet) becomes available.
- [ ] Propagate `position_label` through `runs_long` so notebook 03 can condition on it.

---

### 2. Add 2021 data to AWS
**Status**: ⚠️ External dependency — cannot be done in code alone

**What was done**:
- `01_data_loading.ipynb` §1 documents the `SOURCE_A` / `SOURCE_B` source keys
  and the call to `index.matches_for_source()`.
- The pipeline is **generic** — it will automatically pick up 2021 matches once
  they are loaded into S3 with the same parquet structure as 2020.

**What still needs to happen (manual steps)**:
1. Locate the 2021 match schedule (where is `kamper.xlsx` for 2021?  Check with
   whoever runs the data pipeline).
2. Run the existing ETL / upload scripts against the 2021 GPS files.
3. Verify the new matches appear in `index.matches_for_source("rosenborg")` etc.
4. Re-run `01_data_loading.ipynb` — everything else updates automatically.

**Notes**:
- Target: 4 head-to-head fixtures (2 × 2020, 2 × 2021) — see item 3.
- No code changes needed once data is in S3.

---

### 3. Compile statistics from multiple matches
**Status**: ✅ Architecture complete — blocked on item 2 for 2021 data

**What was done**:
`01_data_loading.ipynb` is now fully loop-based:
- Iterates over `single_matches` (all matches for Team A from `index`).
- Iterates over `h2h_fixtures` (head-to-head fixtures for both teams).
- Concatenates all matches into unified `runs_long`, `df_pmv`, `hazard_intervals`, etc.
  with a `match_id` column so per-match breakdown is always possible.
- Head-to-head caches (`df_pmv_teamA`, `df_pmv_teamB`, `hazard_intervals_h2h`)
  are saved separately for notebook 07.

**Pipeline run results** (2020 season):
- 90 matches found in schedule.
- 17 matches successfully loaded from GPS data (remaining have missing parquet files in S3).
- 707,556 base runs; 18,021 centroid runs.
- 89,947 PMV rows (polarisation/milling/velocity per frame).
- 2 head-to-head fixtures loaded.

**For notebook 07**: once 4 head-to-head fixtures are loaded, simply re-run
`01_data_loading.ipynb` and notebook 07 will use all 4.

**Outstanding**:
- [ ] Add 2021 data (item 2).
- [ ] Add a match-level summary table (notebook 01 §7 diagnostic plots).
- [ ] Report per-match and pooled statistics side-by-side to check consistency.

---

### 4. Length CCDF in Markov and stochastic models
**Status**: ⚠️ Partially addressed — full implementation needs a centroid-speed model

**What was done**:
- **Figures 2B and 3C** already show length CCDFs empirically (notebooks 02 and 03).
- **Figure 4D** (Markov simulation) currently only simulates *durations* $T$.
- **Figure 5E** (OU model) also simulates durations only.

**What remains** (`04_hazard_mechanism.ipynb` and `05_stochastic_order_model.ipynb`):

To simulate run lengths $L = T \cdot \bar{v}$ you need a model for the **mean centroid
speed** $\bar{v}$ conditional on the order state.  The cleanest approach:

1. Fit a linear (or log-linear) model: $\bar{v} \mid p_{\text{state}} \sim \text{Normal}(\mu_v(s), \sigma_v^2)$.
2. At each simulated run, draw a speed from the appropriate state's distribution.
3. Set $L = T \cdot \bar{v}$.
4. Plot the simulated length CCDF alongside empirical.

**Suggested code location**: Add to the end of `04_hazard_mechanism.ipynb` §D and
`05_stochastic_order_model.ipynb` §E.

**Priority**: Medium — durations are the primary quantity; lengths follow if the
speed model is reasonable.

---

### 5. Truncation / field limits / opponent pressure variable
**Status**: ⚠️ Design only — no code yet

**What this is asking**: Runs near the pitch boundary are *truncated* (the team
can't keep going in the same direction past the touchline), and opponent pressure
may force direction changes independently of collective order.  These are
confounders that could make order look more protective than it is.

**Suggested implementation** (add to `hazard_intervals` in notebook 01):

| Variable | How to compute | Where to add |
|----------|---------------|-------------|
| `dist_to_boundary_m` | Min distance from centroid to any pitch edge at the start of each run | `01_data_loading.ipynb`, run-building step |
| `heading_to_boundary_s` | Estimated seconds until centroid would exit at current speed/heading | Same |
| `opponent_dist_m` | Distance between team centroids (H2H only) | `07_head_to_head_coupling.ipynb` |
| `opponent_speed_mps` | Opponent centroid speed (H2H only) | Same |

Then in `04_hazard_mechanism.ipynb` add a third Cox model:
```python
h(a, z_p, z_v, z_boundary, z_opp) = h0(a) * exp(...)
```
and compare AIC to the simpler model.

**Priority**: High for the paper — referees will ask whether the inverse-age hazard
is just a boundary effect.

**Outstanding**:
- [ ] Add `dist_to_boundary_m` and `heading_to_boundary_s` to notebook 01.
- [ ] Add these as covariates in notebook 04 Cox model.
- [ ] For H2H: add opponent centroid distance and speed to hazard intervals in notebook 07.

---

### 6. Team coupling analysis / polarisation modes
**Status**: ✅ Code complete — needs H2H data

**What was done** (`07_head_to_head_coupling.ipynb`):
- **Panel A**: Hexbin scatter of $p_A$ vs $p_B$ with Pearson $r$.
- **Panel B**: Cross-correlation of $p_A$ and $p_B$ up to ±30 s lag — detects
  whether one team leads the other.
- **Panel C**: Inter-centroid distance vs each team's polarisation.
- **Panel D**: Coupled hazard model with focal-team order *and* opponent order as
  covariates — tests whether opponent pressure raises your termination risk.
- **Panel E**: PCA of the joint $(p_A, p_B)$ time series:
  - **PC1** = in-phase mode (both teams polarised together — mutual coordination).
  - **PC2** = anti-phase mode (one high, one low — one team pressing, one retreating).

**Outstanding**:
- [ ] H2H caches (`df_pmv_teamA`, `df_pmv_teamB`) need to be explicitly saved in
  notebook 01 (the H2H block saves `transport` but not the split PMV tables — add these).
- [ ] Increase H2H sample to 4 fixtures (item 2 / item 3).
- [ ] Consider time-frequency analysis (wavelet coherence) to detect synchronisation
  at specific time scales (e.g., are modes stronger in the final 10 minutes?).

---

### 7. General refactoring
**Status**: ✅ Complete

**What was done**:

The monolithic `Levy_Paper_Analysis.ipynb` (~36 cells, ~3,000+ lines of mixed code)
has been split into:

```
notebooks/
  01_data_loading.ipynb           (data pipeline, position labels, cache save)
  02_transport_phenotype.ipynb    (Figure 2)
  03_order_and_transport.ipynb    (Figure 3)
  04_hazard_mechanism.ipynb       (Figure 4 + AIC/BIC + Cox model)
  05_stochastic_order_model.ipynb (Figure 5 + OU fitting)
  06_robustness_checks.ipynb      (theta sensitivity, speed confounding, selection)
  07_head_to_head_coupling.ipynb  (coupling, modes, coupled hazard)

util/
  paper_utils.py    (single source of truth: constants, plotting, cache I/O)
  __init__.py

docs/
  paper_skeleton.md    (full paper draft with section outlines)
  figures_guide.md     (per-panel documentation)
  todo_progress.md     (this file)
```

**Design principles applied**:
- **One notebook = one figure** (or one coherent analysis).
- **Cache-first**: notebook 01 saves all heavy computation; all other notebooks
  load from parquet — no re-running the GPS pipeline.
- **Single source of truth**: all shared constants (`THETA_DEG`, `STATE_COLORS`,
  `A0`, `N_SIM`, etc.) live only in `paper_utils.py`.
- **Reusable plot functions**: `plot_ccdf`, `plot_msd`, `plot_hazard`,
  `plot_state_transition_matrix` — consistent style across all figures.
- **Publication quality**: `configure_paper_plotting()` sets serif fonts,
  300 dpi, solid white backgrounds, PDF/PNG export.

**Post-refactor fixes applied**:
- `06_robustness_checks.ipynb`: corrected column references `length_m` → `run_length_m`
  and `speed_mps` → `v_mean_mps` to match the parquet cache schema.
- `04_hazard_mechanism.ipynb`: added `age_bin` creation from `age_mid_s`; improved
  diagnostics; Cox model gracefully skips if `lifelines` is not installed.
- `05_stochastic_order_model.ipynb`: simplified panel A polarisation time series
  selection — now uses `run_id` lookup directly without a fallback path.
- `paper_utils.py` `plot_hazard()`: switched inverse-age fitting from MLE to SSE
  (Nelder-Mead on sum of squared errors against binned hazard) for a more direct
  and stable curve fit.

---

### 8. Improve animation
**Status**: ⚠️ Not yet refactored — flagged for separate task

**Original code**: `viz/animation.py` and cells 13–14 / 34–35 in the original notebook.

**Suggested improvements** (for a future iteration):

| Improvement | Description |
|------------|-------------|
| **Order overlay** | Display instantaneous polarisation as a colour ring or halo around the centroid marker |
| **Run boundaries** | Flash or mark when a centroid run starts/ends |
| **State colouring** | Colour player trails by their current order state (low/mid/high) |
| **Speed bar** | Horizontal bar at the bottom showing centroid speed relative to match maximum |
| **Side-by-side H2H** | For head-to-head fixtures, show both teams simultaneously with inter-centroid distance indicator |
| **Export quality** | Increase DPI, add title card, match timestamp ticker |

**Location**: These improvements belong in `viz/animation.py` (existing file at
`src/viz/animation.py`).  A separate notebook `08_animation.ipynb` could wrap
them and produce the output MP4 without cluttering the analysis notebooks.

**Priority**: Low for the paper (figure panels matter more), but high for talks
and supplementary video.

---

## Part 2 — New Items (Blocked on More Data)

These items cannot be completed until the 2021 data is uploaded to S3.

| # | Item | Blocked on |
|---|------|------------|
| P2-1 | Cross-year stability: do the key statistics (exponents, hazard ratios) replicate in 2021? | 2021 data |
| P2-2 | Seasonal effects: does collective order differ between early-season and late-season matches? | 2021 data + match dates |
| P2-3 | Multi-match meta-analysis: pool all H2H fixtures, compute effect sizes with CI | 2021 data (for N=4) |
| P2-4 | Opponent-pressure covariate in hazard model (item 5 above) | H2H data for both years |
| P2-5 | Formation-aware position labels (GK / CB / WB / CM / AM / W / ST) | Formation sheets or manual label |
| P2-6 | Wavelet coherence of joint polarisation (H2H coupling at multiple timescales) | N=4 H2H fixtures |
| P2-7 | Supplementary animation (item 8) | Any data — but low priority |

---

## Summary

| Item | Status |
|------|--------|
| 1. Position labelling | ✅ Code complete |
| 2. 2021 data to AWS | ⚠️ External / manual |
| 3. Multi-match stats | ✅ Architecture complete |
| 4. Length CCDF in models | ⚠️ Needs speed model extension |
| 5. Truncation / boundary / pressure | ⚠️ Design only — no code |
| 6. Team coupling / modes | ✅ Code complete (needs H2H caches) |
| 7. General refactoring | ✅ Complete |
| 8. Animation improvements | ⚠️ Planned — separate task |
