# Uninterrupted held-out policy evaluation

Base: integrated revision `c791812c1f6fa69c07db43d42a154e08db9a9e8e`; tested source and runtime hashes are in `sha256.txt`. Apple M6; the independent CPU contact reference overlapped, so no timing or performance claim is made.

The default host scheduler resets environment 1 at zero-based step 220. For a 256-step selector evaluation, this previously split that environment's supposedly held-out trajectory. The added `--no-periodic-resets` option preserves the authored initial reset while suppressing periodic benchmark resets; policy selection now requests this option. Ordinary rollout defaults remain unchanged.

The synthetic native test passes all 26 cases with zero actions and again with a same-runtime compiled zero policy. With periodic resets disabled, both environments receive exactly two initial host resets in total, both reach the 256-step timeout, and clean rates are 1, 1 and 0 at steps 221, 256 and 257. The prior scheduled behavior, failure terminations and repeat boundaries remain covered.

A separate integrated one-update Franka training and selection smoke completed; both candidate and incumbent evidence records include the option, two initial host resets, two final-step timeouts, clean rate 1 and zero failed physics steps. Fixture and runtime input hashes remained unchanged. This is a synthetic workflow check, not evidence of learned manipulation skill, calibrated forces or general deployment.

The selector suite completed 46 of 47 tests; the remaining guided-task assertion is a pre-existing conflict about `--interaction-reset-only`, outside this non-G1 change. The complete failure is retained in `unittest.stderr`.
