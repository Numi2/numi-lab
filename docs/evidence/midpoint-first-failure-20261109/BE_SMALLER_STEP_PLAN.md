# One smaller backward-Euler accuracy attempt

The previous fixed-law n=2 backward-Euler ladder ended at 3.90625 us with no
qualifying step, so no matched-cost comparison is available. Test one further
dyadic step, 1.953125 us, for the same 10 ms fixture and the identical
reference-area law, material, physical scales, error gates and 0.25 us RK4
reference. The single attempt is capped at 5120 requested steps; retain a
rejection or error as a result. The prediction is that at least one original
force or impulse gate will still fail, but the observation decides it. A pass
would justify a fresh matched-binary midpoint and BE benchmark only after the
CPU reference jobs finish; this accuracy run includes snapshots and is not a
performance measurement. Record current executable and shader hashes: the
previous measured probe executable has since been overwritten, so this run
cannot inherit its binary identity or relabel old timings. No tolerances,
iteration budgets, friction or contact coefficients are changed.
