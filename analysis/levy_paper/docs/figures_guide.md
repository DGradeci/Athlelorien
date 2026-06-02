# Figures Guide

This document describes every panel in every figure, its scientific purpose,
the notebook that produces it, and the outstanding questions that each panel
is designed to answer.

---

## Figure 1 — Schematic Workflow
**Notebook**: *to be created* (static diagram / Inkscape / matplotlib)
**Status**: ✗ Not yet created

### Purpose
Provide a visual map of the multi-scale analysis pipeline so readers understand
the logical flow from raw GPS to statistical inference.

### Design
```
Raw GPS (lat/lon, 1 Hz)
    │
    ▼ pitch calibration
Pitch coordinates (x_m, y_m)
    │
    ▼ phase labelling + active filter
Active players only (1H + 2H)
    │
    ├──────────────────────────────┐
    ▼ centroid                    ▼ individual tracks
Team centroid C(t)           Player paths rᵢ(t)
    │                              │
    ▼ run segmentation             ▼ run segmentation
Centroid runs (T, L, v̄)      Player runs
    │
    ├─────────────────┬────────────────────────────┐
    ▼                 ▼                            ▼
Polarisation p(t)  MSD analysis              Hazard intervals
    │                  │                        │
    ▼ order states  Figure 2               Figure 4
Figure 3                                   Figure 5
```

### Insets
- 3 example frames showing low/mid/high polarisation states:
  - Low: players scattered, arrows pointing in many directions.
  - Mid: partial alignment.
  - High: all arrows pointing similarly.

### Key message
Football movement is a multi-scale process.  Individual tracks partially
cancel; centroid captures the collective component.

---

## Figure 2 — Transport Phenotype
**Notebook**: `02_transport_phenotype.ipynb`
**Status**: ✓ Code complete

### Panel A — CCDF of centroid run durations
- **x-axis**: Run duration $T$ (s), log scale.
- **y-axis**: $P(T \geq t)$, log scale.
- **Lines**: centroid (solid green), player (dashed purple), exponential reference (grey dotted).
- **What to look for**: Both curves lie above the exponential reference at long $T$,
  indicating heavy tails.  Centroid tail is heavier than individual-player tails,
  consistent with emergent collective persistence.

### Panel B — CCDF of centroid run lengths
- Same as A but for arc-length $L$ (m).
- **What to look for**: Heavy tail in length — long-range transport events are disproportionately common.

### Panel C — MSD decomposition
- **x-axis**: Lag $\tau$ (s), log scale.
- **y-axis**: MSD (m²), log scale.
- **Lines**: player (purple), centroid (green), relative (red), each with power-law fit.
- **What to look for**: Centroid MSD exponent $\alpha_c > 1$ (superdiffusive); relative MSD
  exponent $\alpha_r \approx 1$ (normal diffusion of players within the team).

### Panel D — Centroid fraction of player MSD
- **x-axis**: Representative lags (1, 5, 10, 30, 60, 120 s).
- **y-axis**: MSD_centroid / MSD_player.
- **What to look for**: Fraction rises with lag, reaching > 0.5 at long lags — at long
  timescales, most player displacement is centroid-driven.

### Outstanding questions for Figure 2
- [ ] Quantify and report power-law exponents with bootstrap CIs.
- [ ] Check whether individual-player tails vary by position (GK vs FWD).
- [ ] Add second-team or multi-match breakdown as supplementary.

---

## Figure 3 — Collective Order and Transport Coupling
**Notebook**: `03_order_and_transport.ipynb`
**Status**: ✓ Code complete

### Panel A — Polarisation vs centroid speed
- **x-axis**: Polarisation $p$ (0 → 1).
- **y-axis**: Centroid speed $v_c$ (m/s).
- **Content**: Binned means ± SE with Pearson $r$ annotation.
- **What to look for**: Monotone increase — high-order frames move faster.

### Panel B — Duration CCDF by order state
- **x-axis**: Duration (s), log.
- **y-axis**: $P(T \geq t)$, log.
- **Lines**: low (blue dashed), mid (orange dotted), high (red solid).
- **What to look for**: Stochastic dominance — high-order CCDF lies above low-order
  across the full range.

### Panel C — Length CCDF by order state
- Same as B but for length.
- **What to look for**: Same dominance pattern.

### Panel D — Speed–duration decomposition
- **x-axis**: Order state (low / mid / high).
- **y-axis**: Relative to 'low' state (median).
- **Bars**: duration (blue), speed (red), length (green).
- **What to look for**: Both duration and speed increase with order; neither is the
  sole driver of longer runs.

### Outstanding questions for Figure 3
- [ ] Use $p_{\text{start}}$ (not $p_{\text{mean}}$) for panels B–D to make causal claim.
- [ ] Separate by team (is pattern consistent across teams?).
- [ ] Test statistical significance of CCDF shifts (KS test or log-rank test).

---

## Figure 4 — Hazard Mechanism and State Dynamics
**Notebook**: `04_hazard_mechanism.ipynb`
**Status**: ✓ Code complete (Cox model requires `lifelines` package)

### Panel A — Empirical hazard vs run age
- **x-axis**: Run age $a$ (s).
- **y-axis**: Hazard $h(a)$.
- **Content**: Points with Poisson CIs, inverse-age fit overlaid.
- **What to look for**: Decreasing hazard with age — fragile runs die early,
  survivors stabilise.  Inverse-age model fits well.

### Panel B — Hazard by order state
- Same axes as A.
- **Lines**: low (blue), mid (orange), high (red).
- **What to look for**: Uniform downward shift for high order — order protects
  at every age, not just by surviving longer to see lower baseline hazard.

### Panel C — Markov transition matrix
- **Content**: 3×3 heatmap of $P(s_{t+1} = j \mid s_t = i)$.
- **What to look for**: Strong diagonal — states are persistent (not random-switching).
  Off-diagonal elements quantify transition rates.

### Panel D — Empirical vs simulated run-duration CCDF
- **Lines**: empirical (black), simulated (red dashed).
- **What to look for**: Agreement across the full range (especially the tail) validates
  the Markov + killing mechanism.

### Panel E — Hazard ratios (Cox model) / Model comparison table
- **Content**: Forest plot of HR for polarisation and speed, or AIC/BIC table.
- **What to look for**: HR_p < 1 (protective); HR speed may be slightly > 1 if fast
  runs are more likely to overshoot and turn.  AIC favours order model.

### Outstanding questions for Figure 4
- [ ] Bootstrap CIs on hazard ratios (currently point estimates only).
- [ ] Formal AIC/BIC table comparing baseline / order / order+speed models.
- [ ] Use $p_{\text{start}}$ in hazard model to avoid retrospective bias.
- [ ] Investigate whether selection fully explains inverse-age hazard
  (see notebook 06, §5).

---

## Figure 5 — Continuous Stochastic Order Model
**Notebook**: `05_stochastic_order_model.ipynb`
**Status**: ✓ Code complete

### Panel A — Sample polarisation time series
- **x-axis**: Time within run (s).
- **y-axis**: $p(t)$ (offset between examples).
- **Content**: 3 runs (low / mid / high mean polarisation) with mean line.
- **What to look for**: Diversity of polarisation phenotypes; high runs show
  sustained near-1 values; low runs fluctuate widely.

### Panel B — Drift and diffusion
- **x-axis**: Polarisation $p$.
- **y-axis**: Drift $b(p)$ (left) or $\sigma^2(p)$ (right).
- **What to look for**: Drift is negative for $p > \mu_p$ and positive for $p < \mu_p$
  (mean-reverting), consistent with OU.  Diffusion roughly constant.

### Panel C — State-dependent killing rate $\phi(p)$
- **x-axis**: Polarisation $p$.
- **y-axis**: $\phi(p)$ at fixed age $a = 10$ s.
- **Content**: Empirical (points), exponential fit (dashed).
- **What to look for**: Monotone decrease — high-$p$ states are significantly less
  likely to end at any given age.

### Panel D — Polarisation distribution
- **x-axis**: Polarisation $p$.
- **y-axis**: Density.
- **Bars**: empirical (green), OU simulation (red).
- **What to look for**: Close match — OU process captures the empirical marginal
  distribution.

### Panel E — Run-duration CCDF (continuous model)
- **Lines**: empirical (black), OU + killing (red dashed).
- **What to look for**: OU model reproduces empirical tail.  This is the key
  quantitative test of the continuous model.

### Outstanding questions for Figure 5
- [ ] Fit $\beta$ via MLE rather than moment matching.
- [ ] Add 95% simulation envelopes (N=100 repetitions of 20k runs).
- [ ] Compare OU with a two-state Markov (which fits better?).
- [ ] Add length CCDF comparison (needs centroid speed model, see TODO §4 in notebook).

---

## Supplementary Figures

### S1 — Directional Anisotropy
**Notebook**: From original `Levy_Paper_Analysis.ipynb` (cells 21–22)
**Status**: ✗ Not yet refactored

- Polar histogram of run heading angles per player.
- Weighted by run length.
- **What to look for**: Preferred movement directions along the pitch axis
  (forwards/backwards bias), anisotropic if tactical constraints dominate.

### S2 — Robustness Tables
**Notebook**: `06_robustness_checks.ipynb`
**Status**: ✓ Code complete (depends on per-θ caches)

### S3 — Head-to-Head Coupling
**Notebook**: `07_head_to_head_coupling.ipynb`
**Status**: ✓ Code complete (depends on h2h caches)
