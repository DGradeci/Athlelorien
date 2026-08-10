# Final Figure 4/5 Polish Report

Generated: 2026-08-05T19:33:44
Script: analysis\levy_paper\scripts\create_final_fig4_fig5_polished.py

## Inputs

- Figure 4A pooled model source: analysis\levy_paper\figures\source_data\figure4_killed_transport\audits\figure4_panelA_smooth_late_hazard_audit.csv
- Figure 4A pooled empirical source: out-of-fold interval prediction union from analysis\levy_paper\data\processed\figure4_fig5_crossfitted\out_of_fold_interval_predictions.parquet
- Figure 4B empirical interval source: analysis\levy_paper\data\processed\figure4_terminal_excess_hazard\terminal_interval_analysis_table.csv
- Cross-fitted package source: analysis\levy_paper\data\processed\figure4_fig5_crossfitted
- Figure 4A/B bootstrap unit: physical_fixture_id; B=500

## Final Report Questions

1. Were any scientific parameters or model definitions changed? No.
2. What is the exact supported age range in Figure 4A? 0-35 s.
3. How are model curves shown beyond the supported range? Lighter dashed continuation beyond the supported range.
4. Is Figure 4B restricted or de-emphasised after 30 s? Restricted to 0-30 s.
5. Is Figure 4C on the same interval convention as Figure 4A? Yes; survival is reconstructed as S(t_right)=S(t_left)*(1-p_interval), so the [0,1] interval drops survival at t=1.
6. Does Figure 4C add any new model parameter? No; it reuses the Figure 4A empirical/model one-second interval probabilities.
7. Is Figure 5A explicitly conditional on survival? Yes; transition_conditioning=conditional_on_survival in source data.
8. Is Figure 5A pooled rather than cross-fitted? Yes.
9. Does Figure 5B avoid all held-out future-state information? Yes; it propagates training-estimated pi0, transitions and termination.
10. Does the age-banded model outperform the stationary model numerically? Yes; current-High IAE stationary=0.141179, age-banded=0.0208137.
11. How many physical fixtures improve? Current-High IAE improves in 44/47; survival IAE improves in 47/47.
12. Does Figure 5C reproduce held-out exposure within uncertainty? Full-model high-order exposure IAE=0.0225549; fixture error distribution: full model fixture IAE mean=0.0595305, median=0.0470526, max=0.161468.
13. Are the counterfactuals correctly labelled non-additive? Yes.
14. Do S0/S1/S2 support the same qualitative conclusion? Robustness rows available=2; see supplementary_s0_s1_s2_composition_robustness.
15. What do the season-transfer tests show? crossfit script unavailable.
16. Are Figures 4 and 5 ready for manuscript insertion? A. READY FOR MANUSCRIPT INSERTION.

A. READY FOR MANUSCRIPT INSERTION
