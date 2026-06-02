# Levy-Type Transport in Football: Collective Order and Run Survival

## Executive Summary

This analysis tests whether football team-centroid movement can be understood as a **collective active-matter transport process** in which coherent team order increases speed, suppresses run termination, and generates broad run-duration and run-length statistics. We treat the team as a single collective object and ask: *do coherent, polarised team states create longer-lived and longer-ranged centroid transport?*

## Central Hypothesis

$$\text{Collective Order} \Rightarrow \text{Lower Termination Hazard + Higher Speed} \Rightarrow \text{Broad Centroid-Transport Tails}$$

---

## Paper Structure

### 1. Introduction
- Football teams as active collectives
- Why team-level movement matters (not just individual players)
- Connection to movement ecology and collective motion physics
- Research question: does alignment/order affect team transport persistence?

### 2. Data and Methods

#### 2.1 GPS Tracking Data
- Player-level tracking during matches
- Both individual and head-to-head fixtures
- Temporal resolution: 1 Hz (downsampled from raw)
- Spatial reference: pitch-aligned Cartesian coordinates

#### 2.2 Preprocessing Pipeline
1. **Raw GPS loading** → `df_raw`
2. **Pitch calibration** → `df_xy` (metres, relative to centre)
3. **Match phase labelling** → identify 1H and 2H
4. **Active-player filtering** → `df_active` (exclude bench players)
5. **Path segmentation** → continuous trajectories per player

#### 2.3 Team Centroid
$$\mathbf{C}(t) = \frac{1}{N(t)} \sum_{i=1}^{N(t)} \mathbf{r}_i(t)$$

The centroid is the team-level analogue of an individual trajectory.

#### 2.4 Collective Order Metrics

**Polarisation** (main metric):
$$p(t) = \left| \frac{1}{N(t)} \sum_{i=1}^{N(t)} \hat{\mathbf{v}}_i(t) \right|$$
- $p \approx 1$: coordinated movement
- $p \approx 0$: disorganised movement

**Milling** (secondary):
$$m(t) = \left| \frac{1}{N(t)} \sum_i (\hat{\boldsymbol{\rho}}_i \times \hat{\mathbf{v}}_i)_z \right|$$
- Rotational/tangential motion

#### 2.5 Run Segmentation
Partition trajectories into **runs** using a turning-angle threshold:
- Continue while turning angle $\theta_k < \theta_{\max}$ (default 30°)
- Restart on sharp turns
- Compute per-run duration $T$, arc length $L$, mean speed $\bar v = L/T$

#### 2.6 Mean Squared Displacement (MSD)
$$\text{MSD}(\tau) = \left\langle |\mathbf{r}(t+\tau) - \mathbf{r}(t)|^2 \right\rangle_t \sim \tau^\alpha$$

Decomposition:
$$\mathbf{r}_i(t) = \mathbf{C}(t) + \boldsymbol{\rho}_i(t) \Rightarrow \text{MSD} = \text{MSD}_{\text{centroid}} + \text{MSD}_{\text{relative}} + \text{cross}$$

#### 2.7 Survival and Hazard Analysis

**Hazard function**:
$$h(a) = \lim_{\Delta t \to 0} \frac{P(a \leq T < a + \Delta t \mid T \geq a)}{\Delta t}$$

**Inverse-age hazard model** (baseline):
$$h_0(a) = \lambda_\infty + \frac{\mu}{a_0 + a}$$

**Order-dependent hazard** (with covariates):
$$h(a, p) = h_0(a) \exp(\beta z_p)$$

where $z_p$ is z-scored polarisation.

---

## Figures and Results

### Figure 1: Schematic Workflow
*Status: To Be Created*

- Boxes: Raw GPS → Pitch coordinates → Active players → Centroid → Run segmentation → Collective order → Hazard model
- Arrows: Information flow and data transformations
- Inset: Example frames showing low/mid/high polarisation states

**Key Message**: Football movement is a multi-scale process from individual players to team centroid.

---

### Figure 2: Transport Phenotype
*Notebook: `02_transport_phenotype.ipynb`*

**Panels**:
- **A**: CCDF of centroid run durations $P(T \geq t)$
  - Log-log plot shows broad, heavy-tailed distribution
  - Exponential fit as reference
- **B**: CCDF of centroid run lengths $P(L \geq \ell)$
  - Similar heavy-tail signature
- **C**: MSD of individual players, centroid, and relative motion
  - Log-log plot with power-law fits
  - Extract exponents $\alpha$ for each component
- **D**: Decomposition bar chart
  - Fraction of player MSD explained by centroid component
  - Supports idea that team-level transport is substantial

**Key Result**: Player runs are broad-tailed and substantially driven by centroid motion.

---

### Figure 3: Order and Transport Coupling
*Notebook: `03_order_and_transport.ipynb`*

**Panels**:
- **A**: High-polarisation frames move faster
  - Scatter + regression: $p(t)$ vs. $v_c(t)$
  - Or binned means with error bars
- **B**: CCDF of centroid run durations split by realised order (low/mid/high terciles)
  - Low order (dashed): shorter, steeper tail
  - High order (solid): longer, heavier tail
- **C**: CCDF of centroid run lengths split by order
  - Similar pattern: high order → longer runs
- **D**: Decomposition of length difference
  - Is it due to longer duration, higher speed, or both?
  - Show $L = T \cdot \bar v$ panel

**Key Result**: High collective order is associated with longer, more persistent centroid transport episodes.

---

### Figure 4: Hazard Mechanism and State Dynamics
*Notebook: `04_hazard_mechanism.ipynb`*

**Panels**:
- **A**: Empirical hazard as a function of run age
  - Points with Poisson confidence intervals
  - Inverse-age fit overlaid
  - Shows decreasing hazard with age (fragile runs die early)
- **B**: Hazard split by collective order states
  - Low order: higher hazard (runs terminate faster)
  - High order: lower hazard (runs persist longer)
  - Inverse-age baseline for each state
- **C**: Transition probability matrix among order states
  - Heatmap or network diagram
  - Shows persistence: diagonal blocks are large
- **D**: Empirical vs. simulated run-duration distributions
  - Simulate state process + age-dependent killing
  - CCDF comparison
  - Measures whether mechanism can explain observations
- **E** (optional): Hazard model coefficients
  - Hazard ratios for order and speed
  - $HR_p < 1$ if order is protective
  - Speed controls to show order is not just a proxy

**Key Result**: Run termination depends on both age and collective order; state-dynamics model can reproduce empirical run statistics.

---

### Figure 5: Continuous Stochastic Order Model
*Notebook: `05_stochastic_order_model.ipynb`*

**Panels**:
- **A**: Empirical polarisation time series
  - Show 1-2 sample runs with low/mid/high phenotypes
  - Highlight transitions and persistence
- **B**: Drift and diffusion of polarisation
  - Estimated from data (e.g., Fokker-Planck style)
  - Or fitted SDE: $dp = b(p)dt + \sigma(p)dW_t$
- **C**: State-dependent killing rate $\phi(p)$
  - How does hazard vary with instantaneous $p$?
  - Expected: $\phi(p)$ decreases as $p$ increases
- **D**: Empirical vs. simulated polarisation distributions
  - Histogram of instantaneous $p$ values
  - Simulated trajectories should match
- **E**: Empirical vs. simulated run-duration distribution
  - Final check: CCDF of $T$ from continuous model
  - Should match Figure 4D

**Key Result**: A simple stochastic model of continuous polarisation coupled to age-and-order-dependent killing reproduces observed team movement statistics.

---

## Analysis Notebooks

All notebooks live in `notebooks/`.  Run `01_data_loading.ipynb` once to populate
`data/` caches; all other notebooks load from those caches.

| Notebook | Purpose | Key outputs |
|----------|---------|-------------|
| [`01_data_loading.ipynb`](notebooks/01_data_loading.ipynb) | Full pipeline: GPS → active players → transport/order/hazard tables. Adds position labelling. | `runs_long`, `trajectory_long`, `msd_long`, `df_pmv`, `centroid_order_runs`, `hazard_intervals` |
| [`02_transport_phenotype.ipynb`](notebooks/02_transport_phenotype.ipynb) | **Figure 2**: CCDF of run durations/lengths, MSD decomposition | `figures/figure2_*` |
| [`03_order_and_transport.ipynb`](notebooks/03_order_and_transport.ipynb) | **Figure 3**: order vs speed, CCDFs by state, decomposition | `figures/figure3_*` |
| [`04_hazard_mechanism.ipynb`](notebooks/04_hazard_mechanism.ipynb) | **Figure 4**: inverse-age hazard, Markov states, Cox model, AIC/BIC | `figures/figure4_*` |
| [`05_stochastic_order_model.ipynb`](notebooks/05_stochastic_order_model.ipynb) | **Figure 5**: OU model, drift/diffusion, killing rate, simulation | `figures/figure5_*` |
| [`06_robustness_checks.ipynb`](notebooks/06_robustness_checks.ipynb) | θ sensitivity, bench inclusion, cross-team, speed confounding, selection | `figures/supp_robustness_*` |
| [`07_head_to_head_coupling.ipynb`](notebooks/07_head_to_head_coupling.ipynb) | H2H: joint polarisation, cross-correlation, distance, coupled hazard, PCA modes | `figures/supp_h2h_*` |

---

## Key Outstanding Questions

### 1. **Figure 1 Schematic**
- [ ] Create visual workflow diagram (Inkscape or matplotlib)
- [ ] Include example frames (low/mid/high order)
- [ ] Explain the multi-scale nature of analysis

### 2. **Robustness Checks** → `06_robustness_checks.ipynb`
- [ ] Test sensitivity to turning-angle threshold ($\theta_{\max} = 20°, 30°, 40°$)
  — needs per-θ caches built from notebook 01
- [ ] Compare results with/without bench players
- [ ] Cross-team consistency (do patterns hold across teams?)
- [ ] Speed confounding (partial correlations) — **code complete**
- [ ] Selection effect vs. true age-dependence — **code complete**

### 3. **Head-to-Head Coupling** → `07_head_to_head_coupling.ipynb`
- [ ] Joint polarisation correlation and cross-correlation — **code complete** (needs h2h caches)
- [ ] Inter-team distance vs. order — **code complete**
- [ ] Coupled hazard model (focal + opponent order) — **code complete**
- [ ] Polarisation modes via PCA — **code complete**

### 4. **Mechanistic Depth** → `04_hazard_mechanism.ipynb`
- [ ] What drives the inverse-age hazard? Selection test — **code in 06**
- [ ] Order state persistence times (from transition matrix diagonal) — **code complete**
- [ ] Is speed a mediator or confounder? Cox model with controls — **code complete**

### 5. **Statistical Inference** → `04_hazard_mechanism.ipynb`
- [ ] Bootstrap CIs on hazard ratios — **TODO**
- [ ] Formal AIC/BIC model comparison — **code complete**
- [ ] Log-rank / KS test for CCDF separation by state — **TODO**

### 6. **Data Extension** → `01_data_loading.ipynb`
- [ ] Add 2021 season data to S3 (need match schedule)
- [ ] Compile 4 head-to-head fixtures across 2 years
- [ ] Add length CCDF to stochastic model comparison (notebook 05 §TODO)

### 7. **Position Labelling** → `01_data_loading.ipynb`
- [x] Add GK/DEF/MID/FWD labels from kickoff depth — **code complete**
- [ ] Condition order–transport analyses on position group (notebook 03)

---

## Interpretation and Theory

### Core Claim
Football team-centroid transport is **not a random walk**. It is a **collective, state-dependent survival process** where coherent team motion (high polarisation) both:
1. Increases transport speed
2. Suppresses run termination risk

### Mechanisms
1. **Collective order → higher speed**: Players moving together achieve more directed transport
2. **Collective order → lower hazard**: High-order states are more stable/resistant to disruption
3. **Selection effect**: Fragile, disorganised runs die early; survivors become biased toward stable contexts

### Analogy to Animal Movement
In animals, **Lévy flights** and **state-switching** movement emerge from environmental heterogeneity and internal state. Football teams, like biological collectives, exhibit state-dependent transport with broad tails—but are additionally constrained by **tactics and team structure**.

---

## Limitations and Caveats

1. **$p_{\text{mean}}$ is retrospective**: Run-average polarisation describes observed phenotype but should not be used as a predictive variable. Use $p_{\text{start}}$, $p_{\text{early}}$, or strictly-past rolling averages instead.

2. **Speed and order are confounded**: High-order teams tend to move faster. Hazard models should include speed controls.

3. **Run segmentation is threshold-dependent**: Results depend on the turning-angle threshold. Robustness checks are essential.

4. **Head-to-head analysis is exploratory**: Only a subset of matches have both teams tracked. Opponent effects should be reported separately.

5. **Causality is inferred, not proven**: The model shows that order correlates with survival, not that order *causes* survival. Alternative explanations (e.g., external pressure, tactical setup) cannot be ruled out.

---

## Code and Data Organization

```
analysis/levy_paper/
├── README.md               (this file — overview and outstanding TODOs)
├── docs/
│   ├── paper_skeleton.md   (full paper draft skeleton with section outlines)
│   └── figures_guide.md    (per-panel documentation: purpose, design, questions)
├── notebooks/
│   ├── 01_data_loading.ipynb           (pipeline → cache)
│   ├── 02_transport_phenotype.ipynb    (Figure 2)
│   ├── 03_order_and_transport.ipynb    (Figure 3)
│   ├── 04_hazard_mechanism.ipynb       (Figure 4)
│   ├── 05_stochastic_order_model.ipynb (Figure 5)
│   ├── 06_robustness_checks.ipynb      (Supplementary robustness)
│   └── 07_head_to_head_coupling.ipynb  (Supplementary H2H)
├── util/
│   ├── paper_utils.py      (shared constants, plotting helpers, cache I/O)
│   └── __init__.py
├── figures/                (output: PNG + PDF — gitignored)
└── data/                   (parquet caches — gitignored)
```

### Shared utilities (`util/paper_utils.py`)

| Symbol | Type | Description |
|--------|------|-------------|
| `THETA_DEG` | int | Turning-angle threshold (30°) |
| `STATE_LABELS` | list | `["low", "mid", "high"]` |
| `STATE_COLORS` | dict | Colour-blind friendly palette per state |
| `DATA_DIR` | Path | `../data/` |
| `FIGURES_DIR` | Path | `../figures/` |
| `configure_paper_plotting()` | fn | Set Matplotlib rcParams for publication |
| `plot_ccdf(ax, values, ...)` | fn | Log-log CCDF with optional exp. reference |
| `plot_ccdf_by_state(ax, df, ...)` | fn | CCDF split by order state |
| `plot_msd(ax, tau, msd, ...)` | fn | MSD with power-law fit |
| `plot_hazard(ax, age, h, ...)` | fn | Hazard plot with inverse-age fit |
| `plot_state_transition_matrix(ax, P)` | fn | Heatmap of Markov matrix |
| `save_cache(df, name)` | fn | Save parquet to `data/` |
| `load_cache(name)` | fn | Load parquet from `data/` |
| `save_figure(fig, name)` | fn | Save PDF + PNG to `figures/` |
| `assign_order_state(p)` | fn | Bin polarisation into state labels |

---

## Contact and Citation

**Authors**: [Your names]  
**Journal**: [Target venue]  
**Date**: June 2026  

If using this analysis or data, please cite:  
> [Full citation TBD]

---

## Revision History

- **2026-06-02**: Initial structure and figure skeleton created
- **2026-06-02**: Branch `feature/dg/levy_paper_analysis_dg` — full refactor:
  - All notebooks created under `notebooks/` (01–07)
  - Shared `util/paper_utils.py` with constants, plotting, cache helpers
  - Paper skeleton and figures guide in `docs/`
  - Outstanding TODOs tracked in README §Key Outstanding Questions
  - New analyses: position labelling (01), robustness suite (06), H2H coupling (07)
