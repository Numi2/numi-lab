# Correct the observation encoding in the verifier

The first analysis rejected the two failed-reset depth planes because it incorrectly expected 5.0, the configured far distance in metres, in the actor row. The actor masked-depth plane is normalized: far/no-return is 1.0. The runtime documentation describes normalization over the calibrated near/far range; HybridRenderer.metal explicitly writes 1.0 for reset depth planes. The original PLAN.md remains unchanged, as do the captured executions, first analysis and its nonzero exit. This is a verifier error, not an established engine failure.

Reanalyze the exact same raw captures with the numerical expectation corrected to 1.0. Also record whether accepted and rejected tracker candidates contain any depth pixel below 1.0; this post-observation diagnostic checks that the fixture includes visible masked geometry and is reported as exploratory. Do not rerun or alter the runtime or the physical-state comparisons.
