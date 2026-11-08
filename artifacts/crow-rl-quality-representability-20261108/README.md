# Adjacent FP32 quality refinement

The original retained crow quality problem admits a representable FP32 state
inside its unchanged 1e-10 stationarity gate. The native solver accepts that
state with zero Newton iterations. Rounding the FP64 solution alone fails.
This refutes an unavoidable FP32 representability floor for that problem.

The new generic SIMD32 refinement searches adjacent single-coordinate and
two-coordinate FP32 moves using the current generalized Hessian. It operates
only after an otherwise unsuccessful Newton budget, for at most eight moves,
at most 64 velocities, residual within 16 times the existing tolerance, existing
feasibility already satisfied, and a sufficient checked Hessian arena. Every
winner is fully reprojected. It must improve the native residual, preserve
feasibility and satisfy a tight objective roundoff bound. Ordinary final
acceptance uses the original gates. Existing Newton/PCG budgets stay unchanged.
Older dispatches with one-float Hessian scratch safely skip refinement.

Five numerical cases pass after correction versus one before correction.
Independent FP64 reprojection gives a worst accepted residual of
9.723268394131896e-11. The insufficient-scratch negative control reproduces
the original failure capture exactly. All 624 motor regression records are
physically identical, excluding GPU timing. The default 640-transition crow
replay has ten byte-identical physical files. Full Release and the owning
constraint IR/native contact probes pass.

In the coupled crow fixture, the first rejection advances from physics
substep 2 to 7 at control tick zero. No complete coupled control transition
is accepted. Chunk sizes one and eight preserve exactly the same protected
first failure capture. The new rejection has native residual 5.1471453e-8;
the independent FP64 reference reaches 7.5897306e-17 in one Newton step,
but rounding its velocity to FP32 produces 5.8327862e-8. This is the next
numerical problem to solve, not evidence that the environment is qualified.

Plans were committed and pushed before their native observations. The
explorations reuse exposed conditions and are engineering diagnosis. Four
free-velocity variants were declared before execution; they are numerical
controls, not ecological held-out behavior. No tolerance, physical setting,
training profile, or policy was promoted. Timings are retained but this slice
does not establish a general throughput gain.

Start with study-result.json, native-trial-plan.json, native-numerical-result.json,
warm-validation-result.json and substep7-fp64-analysis.json. The payload
manifest preserves exact inputs, runtimes, failures, command logs and outputs
through content-deduplicated Git objects. The companion Git-recovery directory
records the executed recovery, exact comparisons and scoped cleanup.
