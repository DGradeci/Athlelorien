# Lévy-Type Transport in Football: Collective Order and Run Survival
## Paper Skeleton — Draft Structure

> **Status**: Skeleton / outline.  Sections marked ✗ are missing data or need writing.
> Figures reference the notebooks in `../notebooks/`.

---

## Abstract

Football teams move as collective entities whose centroid traces broad, heavy-tailed
displacement statistics reminiscent of Lévy-type transport observed in biological
foraging.  We show that this arises from a state-dependent survival process: when
players are collectively polarised (moving in the same direction), the centroid
moves faster **and** runs of directed transport last longer.  We model this as a
continuous-time process in which a stochastic polarisation variable drives both
speed and a decreasing termination hazard.  A parsimonious two-parameter
stochastic differential equation, coupled to an age-and-order-dependent killing
rate, reproduces the empirical run-duration and run-length distributions.

**Keywords**: collective motion, active matter, Lévy flight, survival analysis,
hazard function, football, GPS tracking, polarisation.

---

## 1  Introduction

### 1.1  Collective motion and transport statistics

- Animal groups — flocks, schools, herds — exhibit superdiffusive transport
  [Vicsek et al. 1995; Katz et al. 2011].
- Lévy flights emerge from state-switching and environmental heterogeneity
  [Viswanathan et al. 1999; Reynolds 2015].
- Football teams are human active-matter collectives: bounded domain,
  tactical constraints, intermittent coordination.

### 1.2  Why team-level movement matters

- Individual GPS tracks conflate personal movement with collective displacement.
- Team centroid is the natural *order parameter* for group-level transport.
- Centroid motion couples to spatial coverage, pressing, transition speed.

### 1.3  Research question

> Does collective alignment (polarisation) affect the **persistence** and **range**
> of team-centroid transport, and can a minimal stochastic model explain it?

### 1.4  Outline

Section 2 describes data and methods.  Sections 3–5 present results.
Section 6 discusses implications and limitations.

---

## 2  Data and Methods

### 2.1  GPS Tracking Data

- League: Toppserien (Norwegian women's football), season 2020.
- Resolution: 1 Hz (downsampled from raw).
- Spatial reference: pitch-aligned Cartesian coordinates (metres, relative to centre).
- N matches: `TBD` (single-team); `TBD` head-to-head fixtures.

> **TODO**: Add 2021 season data (4 additional head-to-head fixtures).

### 2.2  Preprocessing

1. Raw GPS loading → `df_raw` (`notebooks/01_data_loading.ipynb` §1–2).
2. Pitch calibration → `df_xy` in metres.
3. Match phase labelling → filter to 1H and 2H.
4. Active-player filtering → `df_active` (ACTIVE_DEPTH = 6 m from goal line).
5. Position labelling → GK/DEF/MID/FWD based on kickoff depth.

### 2.3  Team Centroid

$$\mathbf{C}(t) = \frac{1}{N(t)} \sum_{i=1}^{N(t)} \mathbf{r}_i(t)$$

$N(t)$ is the number of active players at frame $t$.

### 2.4  Collective Order Metrics

**Polarisation** (primary):
$$p(t) = \left| \frac{1}{N(t)} \sum_{i=1}^{N(t)} \hat{\mathbf{v}}_i(t) \right|$$

- $p \approx 1$: all players moving in the same direction.
- $p \approx 0$: incoherent movement.

**Milling** (secondary, not used in main figures):
$$m(t) = \left| \frac{1}{N(t)} \sum_i (\hat{\boldsymbol{\rho}}_i \times \hat{\mathbf{v}}_i)_z \right|$$

### 2.5  Run Segmentation

A *run* is a contiguous stretch of centroid movement in which consecutive turning
angles $\theta_k < \theta_{\max}$.

- Default: $\theta_{\max} = 30°$.
- Robustness checks: $20°, 40°$ (Notebook 06).
- Per-run statistics: duration $T$, arc-length $L = \int |\dot{\mathbf{C}}|\,dt$,
  mean speed $\bar v = L/T$.

### 2.6  Mean Squared Displacement

$$\text{MSD}(\tau) = \langle |\mathbf{r}(t+\tau) - \mathbf{r}(t)|^2 \rangle_t \sim \tau^\alpha$$

Decomposition into centroid and relative-motion components (Figure 2C–D).

### 2.7  Collective Order Covariates

Three order measures per centroid run:
- $p_{\text{mean}}$: retrospective mean over the run (for **descriptive** comparison only).
- $p_{\text{start}}$: polarisation at the first frame (causal covariate).
- $p_{\text{early}}$: mean over the first 3 seconds (causal covariate).

### 2.8  Survival and Hazard Analysis

**Hazard function**:
$$h(a) = \lim_{\Delta t \to 0} \frac{P(a \leq T < a + \Delta t \mid T \geq a)}{\Delta t}$$

**Baseline model** (inverse-age):
$$h_0(a) = \lambda_\infty + \frac{\mu}{a_0 + a}$$

Fit by maximum likelihood (Poisson approximation on age-binned intervals).

**Order-dependent model** (with covariate):
$$h(a, p) = h_0(a) \exp(\beta\, z_p)$$

$z_p$ = z-scored polarisation.  HR < 1 ↔ order is protective.

**Cox proportional hazards model** (time-varying):
$$h(a, z_p, z_v) = h_0(a) \exp(\beta_p z_p + \beta_v z_v)$$

Model comparison via AIC/BIC (Notebook 04).

### 2.9  Continuous Stochastic Model

Ornstein–Uhlenbeck polarisation process:
$$dp = \theta(\mu_p - p)\,dt + \sigma_p\,dW_t, \quad p \in [0,1]$$

Killing rate:
$$\phi(a, p) = \left[\lambda_\infty + \frac{\mu}{a_0 + a}\right] \exp(\beta\, z_p)$$

Simulate $N = 20{,}000$ runs and compare CCDF to empirical (Figure 5E).

---

## 3  Results

### 3.1  Transport Phenotype (Figure 2)

> **Notebooks**: `02_transport_phenotype.ipynb`

- 3A: CCDF of centroid run durations deviates strongly from exponential →
  broad, heavy-tailed distribution.
- 3B: CCDF of run lengths shows similar tail behaviour.
- 3C: MSD analysis reveals superdiffusive centroid motion ($\alpha_c \approx \text{TBD}$),
  distinct from relative motion ($\alpha_r \approx 1$).
- 3D: At lag $\tau = 10\text{s}$, centroid accounts for $\approx \text{TBD}\%$ of player MSD.

**Interpretation**: Team centroid transport is a dominant contributor to individual
player displacement and exhibits anomalous diffusion.

### 3.2  Order–Transport Coupling (Figure 3)

> **Notebooks**: `03_order_and_transport.ipynb`

- 3A: Strong positive correlation between instantaneous polarisation and centroid speed
  ($r \approx \text{TBD}$).
- 3B–C: CCDFs of duration and length shift systematically with order state:
  high-order runs stochastically dominate low-order runs.
- 3D: Both duration **and** speed increase with order, with the duration effect
  larger in relative terms.

**Interpretation**: Higher collective order produces more persistent *and* faster
centroid transport — both mechanisms contribute to longer runs.

### 3.3  Hazard Mechanism (Figure 4)

> **Notebooks**: `04_hazard_mechanism.ipynb`

- 4A: Empirical hazard $h(a)$ decreases with run age — inverse-age model provides
  good fit ($\lambda_\infty \approx \text{TBD}$, $\mu \approx \text{TBD}$).
- 4B: Stratifying by order state, high-order runs have uniformly lower hazard at
  every age — order suppresses termination beyond a maturation effect.
- 4C: Markov transition matrix shows strong diagonal structure
  (persistence times $\approx \text{TBD}$ s per state).
- 4D: Simulated CCDF from the state-dynamics model matches empirical distribution.
- 4E: Cox model: $\text{HR}_p = \text{TBD}$ (95% CI: TBD–TBD),
  $\text{HR}_v = \text{TBD}$.  AIC favours order model over baseline.

**Interpretation**: Run termination depends on both age (maturation) and collective
order (state protection).  A Markov state-switching model with age-dependent killing
is sufficient to reproduce the observed tail.

### 3.4  Continuous Stochastic Model (Figure 5)

> **Notebooks**: `05_stochastic_order_model.ipynb`

- 5A: Polarisation time series shows diverse phenotypes from low-persistence to
  near-constant high-order runs.
- 5B: Estimated drift $b(p)$ is mean-reverting; diffusion $\sigma^2(p)$ is
  approximately constant (consistent with OU).
- 5C: State-dependent killing rate $\phi(p)$ decreases monotonically with $p$
  ($\beta \approx \text{TBD}$).
- 5D–E: OU-simulated polarisation distribution and run-duration CCDF both match
  empirical data.

**Interpretation**: A parsimonious OU + killing model captures the full transport
phenotype without free parameters beyond three ($\theta$, $\sigma$, $\beta$).

---

## 4  Robustness

> **Notebooks**: `06_robustness_checks.ipynb`

| Check | Finding |
|-------|---------|
| $\theta_{\max} = 20°$ | TBD |
| $\theta_{\max} = 40°$ | TBD |
| Bench inclusion | TBD |
| Cross-team | TBD |
| Speed confounding | Partial $r$ (order ~ duration | speed) = TBD |

---

## 5  Head-to-Head Coupling (Exploratory)

> **Notebooks**: `07_head_to_head_coupling.ipynb`

- Contemporaneous correlation $r(p_A, p_B) \approx \text{TBD}$.
- Cross-correlation peak at lag TBD s (Team TBD leads).
- Inter-centroid distance correlates with Team A order ($r \approx \text{TBD}$),
  not with Team B order.
- Coupled hazard: $\text{HR}_{\text{opponent}} \approx \text{TBD}$.
- PCA: PC1 captures $\approx \text{TBD}\%$ of joint variance
  (in-phase polarisation mode).

---

## 6  Discussion

### 6.1  Football as a collective active-matter system

- Teams exhibit hallmarks of active-matter transport: superdiffusion, state-switching,
  collective alignment effects.
- Constraints (bounded pitch, tactical roles) create qualitatively different
  dynamics from unconstrained biological flocks.

### 6.2  Why does order protect runs?

Three non-exclusive mechanisms:
1. **Coordinated momentum**: aligned players sustain directed centroid motion against
   individual perturbations.
2. **Tactical stability**: high-order configurations correspond to organised formations
   that are harder to disrupt.
3. **Selection**: only intrinsically stable (high-order) runs survive to old age
   (partially controlled by conditioning on age; see §Robustness).

### 6.3  Implications for analysis

- Centroid MSD and run statistics depend strongly on collective state — ignoring
  polarisation confounds transport analyses.
- $p_{\text{mean}}$ is *retrospective*; use $p_{\text{start}}$ or $p_{\text{early}}$
  for causal inference.

### 6.4  Limitations

1. **Causal inference**: order correlates with survival, but causality is not proven.
2. **Data scope**: single season (2020); 2021 data pending upload.
3. **Position labels**: coarse (quartile-based); finer role labelling may improve
   position-conditional analyses.
4. **Speed confounding**: controlled in Cox model but may persist at within-state level.
5. **Head-to-head**: small fixture sample — treat as exploratory.

---

## 7  Conclusion

Football team-centroid transport is not random: it is a collective, state-dependent
survival process.  Coherent polarisation simultaneously increases transport speed
and suppresses run termination, producing the observed broad-tailed displacement
statistics.  A minimal stochastic model (OU polarisation + age-and-order killing)
is sufficient to reproduce these statistics without invoking external environmental
heterogeneity.

---

## Supplementary Materials

- **S1**: Directional anisotropy of player runs.
- **S2**: Robustness checks (extended tables).
- **S3**: Model fitting procedures and parameter uncertainties.
- **S4**: Head-to-head coupling full results.

---

## References

> TBD — add key citations:
> - Vicsek et al. (1995) — original flocking model
> - Viswanathan et al. (1999) — Lévy flights in foraging
> - Katz et al. (2011) — fish schools
> - Reynolds (2015) — movement ecology Lévy review
> - Méndez et al. (2016) — stochastic foundations of Lévy walks
> - Duarte et al. — football collective motion
