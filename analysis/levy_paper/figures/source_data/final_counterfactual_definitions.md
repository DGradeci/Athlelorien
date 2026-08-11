# Final Figure 5 Counterfactual Definitions

The full model propagates the training-estimated birth distribution, age-banded transition matrices and state-dependent termination probabilities.

The transitions-only curve uses the age-banded transition matrices but replaces state-dependent termination by a common mortality matched to the fold-level state mixture.

The differential-termination-only curve preserves state-dependent termination but freezes state switching with an identity transition operator.

These frozen counterfactuals are not additive contributions.
