# Solver-timed contact velocity audit

The logging-only rerun matched the previous observer in all 3,208 recorded state/action/contact endpoint records and all 5,888 recorded contact substeps, with no recorded contact overflow or substep error. It captured all 3,584 required candidate velocities. Sampled native FK positions matched replay to 1.31e-8 m. These comparisons do not establish bitwise equivalence of hidden state.

Reconstructing contact-point velocity from substep-start position and solver candidate velocity, the better-sign normal-speed RMSE was 0.000269 m/s against a solver normal RMS of 0.04453 m/s. This passed the preregistered 0.005 m/s and 10% timing screens. The previous beginning-of-substep velocity observer failed that check. The solver uses the opposite sign convention for the normal component.

For the six prespecified correlated development lanes, 30,274 contact entries had normal impulse above 1e-7 Ns. Estimated pre-solve tangential speed was 0.0124 m/s at the median and 0.1745 m/s at the 90th percentile across entries. Across lanes, 39.0–41.3% of summed normal impulse occurred on contacts with estimated pre-solve tangential speed at least 0.02 m/s, and 6.6–9.0% occurred where that condition coincided with tangential impulse at least 95% of the configured dynamic-friction limit. These 0.02 m/s associations and per-lane breakdowns are posthoc descriptions, not independent samples or causal tests. Full per-step distributions are in analysis.json.

Pre-solve tangential velocity describes the velocity presented to the contact solver; it is not by itself measured post-solve slip or evidence that friction caused the failed takeoff. The dynamic coefficient 0.6 is a model setting, not a crow-specific measurement. This test does not qualify launch, flight or landing, and the used seed and correlated lanes are not held-out trials.

The initial supplementary breakdown failed with a KeyError due to quoting in the analysis program; its rejected source and log are retained. The corrected program completed. See PLAN.md, parity.json, analysis.json, full compressed run, source, binaries, and build and input hashes.
