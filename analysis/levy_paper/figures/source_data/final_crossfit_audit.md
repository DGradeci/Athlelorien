# Final Crossfit Audit

Generated: 2026-08-05T19:33:44
Script: analysis\levy_paper\scripts\create_final_fig4_fig5_polished.py

## Checks

```csv
check_id,status,value,details

every_physical_fixture_held_out_once,PASS,47,fold_summary fixtures=47; out-of-fold prediction fixtures=47

heldout_fixture_excluded_from_training,PASS,0,trained_on_heldout_fixture_violation count in out-of-fold predictions

both_team_records_held_out_together,PASS,2,fold unit is physical_match_id; all team/half records sharing fixture_id are in the same held-out fold

state_thresholds_fold_specific,PASS,47,state thresholds keyed by heldout_physical_match_id

pi0_transition_hazard_parameters_fold_specific,PASS,47,hazard parameters keyed by heldout_physical_match_id; pi0/transitions copied separately

one_prediction_per_interval,PASS,0,duplicates over fixture/team-half/run/age interval key

figure5_no_heldout_future_state_sequence,PASS,0,"Figure 5B/C uses propagated training-estimated pi0, transitions and hazards, not observed held-out state histories"

no_heldout_outcomes_used_for_fitting,PASS,0,cached fold outputs mark no training on held-out fixture; fitting tables are keyed by held-out fixture and train-only folds

figure4c_observed_path_covariate_declared,PASS,0,Figure 4C order-dependent hazards use observed held-out order sequence only as the declared time-varying covariate

no_duplicate_source_team_training_test_overlap,PASS,1,each OOF row fold_model_identifier matches its physical_match_id held-out fold

pooled_empirical_union_equals_heldout_union,PASS,355330,Figure 4A empirical points are rebuilt from the union of out-of-fold held-out intervals; Figure 4B uses the same terminal-interval observation universe with production recent-order states
```

## Fold Manifest

- physical_fixture_folds: 47
- out_of_fold_interval_rows: 355330
- out_of_fold_fixtures: 47
- figure4C_observed_path_prediction: yes
- figure5B_C_uses_heldout_future_state_sequence: no
