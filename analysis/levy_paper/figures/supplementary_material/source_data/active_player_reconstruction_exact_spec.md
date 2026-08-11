# Exact Specification: Final Active-Player Reconstruction

This specification documents the implementation currently used by the final all-team cache. It does not redesign the classifier.

## Scope and inputs

Production calls `label_active_players(...)` from `analysis/levy_paper/scripts/build_multiseason_data_cache.py` lines 346-354 after:

- loading one source/day at 1 Hz;
- calibrating pitch-frame coordinates;
- assigning match phases;
- filtering to rows whose `match_phase` is `1H` or `2H`.

The active-player function expects at least `player_name`, `x_m`, `y_m`, and either `timestamp` or `time`. It labels rows; it does not filter rows internally. Production filters to `player_status == "active"` immediately afterwards.

## Pitch geometry and coordinate handling

Pitch calibration returns a rotated pitch-corner list in metres, centred at the pitch centre (`src/utils/pitch_calibration.py` lines 135-154). Coordinates are attached by converting latitude/longitude to local metres, applying the pitch rotation matrix, and dropping invalid latitude/longitude rows (`src/utils/pitch_calibration.py` lines 262-321).

The active-player code does not perform a general polygon point-in-polygon test. It computes an axis-aligned bounding rectangle from `pitch_xy`:

\[
x_{\min}, y_{\min}=\min(\mathrm{pitch\_xy}), \qquad
x_{\max}, y_{\max}=\max(\mathrm{pitch\_xy}).
\]

For each row:

\[
d_s = \min(x-x_{\min}, x_{\max}-x, y-y_{\min}, y_{\max}-y).
\]

This is `signed_depth_from_edge` (`src/utils/player_status.py` lines 58-80). Positive values mean the sample is inside the calibrated rectangle. Negative values mean outside the rectangle. At corners, the magnitude is not Euclidean distance to the rectangle corner; it is the largest axis violation as induced by the `min(...)` expression.

The non-negative pitch depth is:

\[
d = \max(d_s, 0).
\]

This is `depth_from_edge` (`src/utils/player_status.py` lines 29-55).

Derived flags:

- `on_pitch = signed_depth_from_edge >= on_pitch_eps_m`, with `on_pitch_eps_m = 0.1`;
- `in_active_zone = signed_depth_from_edge >= active_depth_m`, with production `active_depth_m = 6.0`.

Thus the production `6.0` m parameter is an interior active-zone threshold: a row is in the active zone only if it is at least 6 m inside every pitch boundary. It is not the soft outside-pitch candidate tolerance in the sticky selector.

Rows with missing or non-finite `x_m`/`y_m` fail `hard_position_valid` in the sticky selector. Rows with invalid latitude/longitude are normally dropped earlier by `attach_xy_from_pitch(...)`.

## Preliminary geometry state

`label_active_players(...)` always computes a per-player geometry state before dispatching to the sticky selector (`src/utils/player_status.py` lines 540-588).

State variables for each player:

- `current`, initial value `"bench"`;
- `time_in_active`, initial value `0.0`;
- `time_off_pitch`, initial value `0.0`.

Rows are sorted by `[player_col, time_col]`. Per-player time increment:

\[
\Delta t = \max(t_k - t_{k-1}, 0),
\]

with the first row filled as `0.0`. There is no upper cap in this preliminary geometry state.

Bench-state update:

- if `in_active_zone` is true, `time_in_active += dt`;
- otherwise `time_in_active = 0.0`;
- if `time_in_active >= activate_s`, set `current = "active"` and record `active_filter_event = "bench_to_active"`.

Active-state update:

- if `on_pitch` is false, `time_off_pitch += dt`;
- otherwise `time_off_pitch = 0.0`;
- if `time_off_pitch >= bench_off_s`, set `current = "bench"` and record `active_filter_event = "active_to_bench"`.

Production values passed from `paper_utils.py` are `activate_s = 70` and `bench_off_s = 110`.

Important: in the production sticky method, this preliminary geometry state is not the final active set. It contributes the binary score term `geometry_player_status == "active"` and its events can differ from final `player_status`.

## Recent motion features

The sticky selector calls `_add_recent_motion_features(...)` (`src/utils/player_status.py` lines 83-145).

Rows are sorted by player and time. For each player:

- `dt` is the timestamp difference, filled with `0.0`, clipped to `[0, 5]` s;
- `dx`, `dy` are coordinate differences, filled with `0.0`;
- `step_raw = sqrt(dx^2 + dy^2)`, set to `0.0` when `dt <= 0`;
- an impossible-jump clamp uses `max_step = 8.0 * max(dt, 1.0)` metres;
- displacement components are scaled by `min(1, max_step / step_raw)` when moving;
- `speed_mps = step / dt`, with zero when `dt <= 0`;
- `vx_mps = dx / dt`, `vy_mps = dy / dt`, with zero when `dt <= 0`.

With production `likelihood_window_s = 45.0`, `window_n = 45` samples. Rolling features are:

- `recent_path_m`: rolling sum of step distance over the last 45 samples;
- `recent_vx_mps`: rolling mean of `vx_mps`;
- `recent_vy_mps`: rolling mean of `vy_mps`;
- `recent_speed_mps`: rolling mean of `speed_mps`.

The rolling window is implemented as a sample count, not a timestamp-indexed rolling duration.

## Initial-XI seed

The seed is computed by `_seed_playing_xi(...)` (`src/utils/player_status.py` lines 187-228).

The window is:

\[
t_0 \leq t \leq t_0 + 120~\mathrm{s}
\]

using `inclusive="both"` (`line 199`). Rows must have finite coordinates and `signed_depth_from_edge >= -coordinate_sanity_margin_m`, with default `coordinate_sanity_margin_m = 30.0`.

For each player, the seed summary includes:

- maximum `recent_path_m`;
- summed `in_active_zone` rows, named `in_zone_seconds`;
- summed `on_pitch` rows, named `on_pitch_seconds`;
- median `depth_from_edge`;
- count of rows where `geometry_player_status == "active"`, named `geometry_active_seconds`.

The seed score is:

\[
S_{\mathrm{seed}} =
0.15\,\max(\mathrm{recent\_path\_m})
+ 0.02\,\sum I(\mathrm{in\_active\_zone})
+ 0.01\,\sum I(\mathrm{on\_pitch})
+ 0.03\,\mathrm{median}(\mathrm{depth\_from\_edge})
+ 0.02\,\sum I(\mathrm{geometry\_player\_status}=\mathrm{active}).
\]

The seed set is the top `seed_top_n = 11` player names after sorting `seed_score` descending. No explicit secondary tie-breaker is coded.

The sticky selector initializes `selected_prev = set(seed_xi)`.

## Sticky candidate gate

The production sticky selector is `_apply_sticky_hierarchical_active_xi(...)` (`src/utils/player_status.py` lines 231-458).

Frame iteration is over `df.groupby(time_col, sort=True).groups.items()` (`line 297`). The frame-step increment used for sticky counters is:

\[
\Delta t_{\mathrm{frame}} =
\begin{cases}
1.0, & \mathrm{first\ frame},\\
\min(5.0, \max(0, t_k-t_{k-1})), & \mathrm{otherwise},
\end{cases}
\]

with non-positive values reset to 1.0 (`lines 297-302`). This is a global frame-to-frame increment, not a per-player increment.

The hard coordinate gate is:

\[
\mathrm{hard\_position\_valid}
= I(\mathrm{finite}\ x,y) \land I(d_s \geq -30.0).
\]

The soft candidate gate at a frame is:

\[
\mathrm{position\_gate}
=
\mathrm{hard\_position\_valid}
\land
\left[
I(d_s \geq -5.0) \lor I(\mathrm{name}\in\mathrm{selected\_prev})
\right].
\]

Thus a previously active player can override the soft 5 m outside-rectangle margin, but cannot override the hard 30 m sanity gate or missing/non-finite coordinates.

Players outside the gate remain labelled `"bench"` unless later selected at another frame. Gated but non-selected players are labelled `"rejected_hoverer"`.

## Velocity coherence

Velocity coherence is computed by `_velocity_coherence_score(...)` (`src/utils/player_status.py` lines 148-184).

For each candidate:

- the velocity vector is `(recent_vx_mps, recent_vy_mps)`;
- speed is `sqrt(recent_vx_mps^2 + recent_vy_mps^2)`;
- a valid recent velocity requires speed `>= min_velocity_speed_mps`, with production value `0.25` m/s;
- valid velocity vectors are converted to unit vectors;
- candidates without valid recent velocity have their row/column of the cosine matrix set to NaN.

For candidate \(i\), finite cosine similarities to other candidates are sorted, the largest \(k\) are retained, and:

\[
C_i = \mathrm{clip}\left(\frac{\mathrm{mean}(\mathrm{top}_k(\cos_{ij}))+1}{2},0,1\right),
\]

with `k = min(velocity_top_k, n-1)` and production `velocity_top_k = 7`. If there are no finite similarities, \(C_i = 0\). If there is only one candidate, \(C_i = 1\).

Velocity coherence is not part of the candidate gate. It affects ranking and replacement through the score.

## Framewise scoring

For each gated candidate, the position score is:

\[
P_i =
\mathrm{clip}
\left(
\frac{d_{s,i}+5.0}{5.0+12.0},
0,1
\right).
\]

The base likelihood score stored as `playing_likelihood_score` is:

\[
\begin{aligned}
L_i ={}&
0.10\,\mathrm{recent\_path\_m}_i
+ 0.025\,\mathrm{depth\_from\_edge}_i \\
&+ 0.25\,I(\mathrm{geometry\_player\_status}_i=\mathrm{active})
+ 0.20\,I(\mathrm{in\_active\_zone}_i) \\
&+ 0.35\,I(i\in \mathrm{seed\_xi})
+ 1.50\,C_i
+ 0.35\,P_i .
\end{aligned}
\]

Missing numeric values are filled as zero for the movement, depth, coherence, and position terms. Boolean terms are cast to 0/1.

The sticky score stored as `hierarchical_active_score` is:

\[
H_i = L_i + 2.50\,I(i\in \mathrm{selected\_prev}).
\]

The continuity term affects ranking but is not stored in `playing_likelihood_score`.

## Selection, replacement and labels

Persistent sticky state variables:

- `selected_prev`: selected active set from the previous processed frame;
- `weak_seconds`: cumulative weak evidence per selected player;
- `challenger_seconds`: cumulative replacement evidence per non-selected candidate;
- `last_t`: previous processed frame timestamp.

At each frame:

1. Start with `selected_now = selected_prev`.
2. If `selected_now` is empty, choose the top `max_active_players` candidates by descending `hierarchical_active_score`.
3. Remove selected players not present among current candidates.
4. While `len(selected_now) < 11`, add the highest `hierarchical_active_score` non-selected candidate immediately. This vacancy-filling step has no 70 s or 35 s delay.
5. For each selected player, update `weak_seconds[name]`:
   - add `step_s` if the base score `L_i < weak_score_threshold`;
   - otherwise reset to `0.0`.
   Production `weak_score_threshold = 1.6`.
6. Let `selected_score_min` be the minimum base score among selected players.
7. For each non-selected challenger, update `challenger_seconds[name]`:
   - add `step_s` if `L_i >= selected_score_min + switch_margin`;
   - otherwise reset to `0.0`.
   Production `switch_margin = 0.75`.
8. A selected player is weak if `weak_seconds >= weak_confirm_s`, production `weak_confirm_s = 35.0`.
9. A challenger is ready if `challenger_seconds >= replacement_confirm_s`, production `replacement_confirm_s = 35.0`.
10. While there is at least one weak selected player and one ready challenger:
    - choose the ready challenger with largest `(challenger_seconds, base_score)` in descending order;
    - remove the weak selected player with the smallest `(base_score, -weak_seconds)` in ascending order;
    - insert the challenger and reset its counters.
11. Finally, cap the selected set by re-sorting selected candidates by descending `hierarchical_active_score` and taking the first 11.

The code uses `pandas.DataFrame.sort_values(..., ascending=False)` without explicit secondary tie-break columns. Therefore no code-level deterministic secondary or tertiary tie-breaker is specified beyond pandas/input ordering behaviour.

Labels:

- selected rows: `player_status = "active"`;
- gated but non-selected rows: `player_status = "rejected_hoverer"`;
- rows outside the candidate gate: remain `"bench"`.

Events:

- newly selected rows get `playing_selection_event = "hierarchical_selected"`;
- removed rows get `playing_selection_event = "hierarchical_removed"` if the removed player has a row in the current frame.

## Missing data, phase boundaries and gaps

- Invalid latitude/longitude rows are dropped during coordinate attachment.
- Rows with non-finite `x_m` or `y_m` fail the sticky hard-position gate.
- Timestamp parsing failures are dropped in `label_active_players(...)`; all-invalid time columns raise an error.
- Duplicate or non-positive per-frame time gaps in sticky updates are reset to a 1 s counter increment.
- Sticky counter increments are capped at 5 s per processed frame gap.
- The production call filters to `1H` and `2H` before active-player labelling, but does not call the selector separately by phase. State is therefore not explicitly reset at halftime. The long halftime gap contributes at most 5 s to sticky counters, while the preliminary per-player geometry state uses uncapped per-player `dt`.
- Continuous downstream active-player and centroid paths are later split at gaps greater than 2 s and paths shorter than 30 samples are removed.
- If fewer than 11 candidates exist, the selector can output fewer than 11 active players. Downstream centroid construction then requires at least seven active players at the frame.

## Substitutions and participation changes

Substitutions are not identified from an external substitutions feed. They are implicit in the sticky selected set:

- a new player becomes eligible by passing the candidate gate;
- if a vacancy exists, the player can be selected immediately by score;
- otherwise replacement requires both a weak selected player and a ready challenger under the 35 s weak/challenger rules;
- a departing or missing player is removed from `selected_now` when not present among candidates;
- overlapping plausible candidates can exist, but the selected active set is capped at 11.

No retrospective correction/backfill step is implemented after the forward loop. The exception is conceptual rather than a backfill pass: the initial seed set is computed from the first inclusive 120 s window before the frame loop begins and is applied from the first processed frame.

## Warm-up areas, benches and tunnels

There is no explicit semantic model of warm-up areas, benches, tunnels, substitution zones, or roles. These tracks are handled only through calibrated rectangle depth, recent motion, geometry state, seed membership, velocity coherence and continuity. No goalkeeper-specific role flag exists; goalkeeper protection is indirect through seed membership, depth, continuity, and the fact that velocity coherence is not an eligibility gate.

## Downstream centroid requirement

After production filters to active rows, `build_transport_tables_from_active_v2(...)` standardizes valid active rows, splits paths, and constructs the centroid as a framewise mean:

\[
\mathbf X_c(t) = \frac{1}{N(t)}\sum_{i=1}^{N(t)}\mathbf x_i(t).
\]

Centroid rows are retained only if `n_players >= min_players_centroid`, with production default `min_players_centroid = 7`.

## Velocity-coherence sensitivity

The retained final supplement scripts define the no-coherence sensitivity as a production active-player relabel on the pitch-calibrated candidate-player cache with `velocity_coherence_weight = 0.0`, retaining pitch geometry, the soft candidate margin and hard coordinate-sanity gate, entry/demotion persistence, continuity, active-player cap, centroid minimum, phase rules and gap rules. See `analysis/levy_paper/scripts/create_minimal_supplement_final.py` line 571 and `analysis/levy_paper/scripts/rerun_figureS2_high_rep_bootstrap.py` line 490.

The no-coherence parquet referenced by the bootstrap script is not present in this cleaned working tree, so the raw relabel run cannot be independently re-executed here without restoring that cache or raw AWS access.
