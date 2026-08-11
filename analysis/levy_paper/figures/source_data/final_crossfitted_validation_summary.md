# Final Crossfitted Validation Summary

- hazard_log_loss_age_only: 0.471038
- hazard_log_loss_age_plus_order: 0.465662
- hazard_log_loss_full: 0.463754
- current_high_IAE_stationary: 0.141179
- current_high_IAE_age_banded: 0.0208137
- paired_current_high_improvement_mean: 0.0780694
- paired_current_high_improvement_median: 0.0857214
- paired_current_high_improvement_bootstrap_ci95: [0.0653096, 0.0887411]
- paired_current_high_fixtures_improved: 44/47
- survival_age_banded_improved_fixtures: 47/47
- high_order_exposure_full_IAE: 0.0225549
- high_order_exposure_fixture_distribution: full model fixture IAE mean=0.0595305, median=0.0470526, max=0.161468
- all_state_metric_rows: 3
- s0_s1_s2_robustness_rows: 2
- season_transfer: crossfit script unavailable

## Validation Table

```csv
metric_family,model_name,n_intervals,n_events,heldout_log_likelihood,heldout_log_loss,brier_score,calibration_slope,calibration_intercept,integrated_absolute_error,log_survival_rmse,n_points,metric_scope,ci_low,ci_high,median_integrated_absolute_error,status,details

hazard,age-only hazard,355330.0,69670.0,-167373.8919804576,0.4710378858538755,0.1481308106017607,0.8095684026296767,-0.2396739378481974,,,,leave-one-physical-fixture-out aggregate,,,,,

hazard,age+order hazard,355330.0,69670.0,-165463.68254903058,0.4656620115076987,0.1476214348934029,0.8996777216532547,-0.1229062045115476,,,,leave-one-physical-fixture-out aggregate,,,,,

hazard,full hazard,355330.0,69670.0,-164785.70075991645,0.4637539773166252,0.1473437358268892,0.989931955578399,-0.0316723505260177,,,,leave-one-physical-fixture-out aggregate,,,,,

survival,stationary generative model,,,,,,,,0.0351962469984624,0.217086608904971,36.0,leave-one-physical-fixture-out aggregate,,,,,

survival,age-banded generative model,,,,,,,,0.0325732647747961,0.1838312012850939,36.0,leave-one-physical-fixture-out aggregate,,,,,

survival,observed-path hazard benchmark,,,,,,,,0.0335612486573578,0.190367390665668,36.0,leave-one-physical-fixture-out aggregate,,,,,

current_high_composition,stationary generative model,,,,,,,,0.141178593412679,0.4807766061342486,36.0,leave-one-physical-fixture-out aggregate,,,,,

current_high_composition,age-banded generative model,,,,,,,,0.0208136651406018,0.1167029188467894,36.0,leave-one-physical-fixture-out aggregate,,,,,

high_order_exposure,full generative model,,,,,,,,0.0225549164218065,0.1168688535680989,35.0,leave-one-physical-fixture-out aggregate,,,,,

high_order_exposure,transitions only,,,,,,,,0.0584910050365272,0.1691735647340496,35.0,leave-one-physical-fixture-out aggregate,,,,,

high_order_exposure,differential termination only,,,,,,,,0.0889663013418969,0.3321289975618741,35.0,leave-one-physical-fixture-out aggregate,,,,,

survival,paired fixture improvement: stationary minus age-banded,,,,,,,,0.0022164147346099556,,47.0,paired physical-fixture bootstrap,0.002104691130029493,0.002325692660006883,,,

current_high_composition,paired fixture improvement: stationary minus age-banded,,,,,,,,0.07806941490009878,,47.0,paired physical-fixture bootstrap,0.06522358737858054,0.08982405478334057,,,

high_order_exposure_fixture_distribution,full generative model,,,,,,,,0.05953054545298767,,47.0,fixture-level distribution,,,0.047052601659352204,,

all_state_composition,stationary generative model,,,,,,,,0.09506434845337587,,36.0,leave-one-physical-fixture-out aggregate,,,,,

all_state_composition,age-banded generative model,,,,,,,,0.018522346492605406,,36.0,leave-one-physical-fixture-out aggregate,,,,,

all_state_exposure,full generative model,,,,,,,,0.019425651895548523,,35.0,leave-one-physical-fixture-out aggregate,,,,,

,,,,,,,,,,,,season transfer,,,,SKIPPED,crossfit script unavailable
```
