# Run 019: dense follow-up of run018 unresolved derivatives

This is a focused, post-hoc CPU FP64 diagnostic over the 16 row-directions that did not meet run018's adjacent-epsilon criterion. It uses powers 20.00 through 32.00 in 0.25 increments, centered around and extending below the coarse-grid best values near 26. All 784 direction/grid records and both one-sided endpoints are retained.

The 1e-6 relative-L2 derivative gate and strict projection gates are unchanged. Every plus/minus endpoint passed the strict scaled-residual and dynamic-correction checks, and all retained the captured center branch. Fifteen of the 16 previously unresolved cases now have adjacent quarter-grid centered comparisons at or below 1e-6.

One case remains unresolved: F[2,0], paper row/tet 172. It passes at powers 30.75 and 31.75 only, with no adjacent passing power. Nearby errors become nonmonotone as realized perturbations shrink; this is consistent with cancellation/rounding influence but does not establish an absolute numerical noise floor. The isolated passes are not promoted.

Run018 is unchanged and remains the predeclared full nine-basis result. This targeted follow-up does not retroactively change its score or qualify the full audit. See adjudication.json and dense-grid-records.jsonl for exact values and bindings.
