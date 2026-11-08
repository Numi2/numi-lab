# Native split-precision representation experiment

On the Apple M4 Pro, a Metal high/low FP32 expansion solves both retained
crow contact problems in one Newton update. Independent FP64 evaluation gives
scaled stationarity residuals 2.5718118e-15 (substep seven) and 1.1882440e-17
(substep two), below the unchanged 1e-10 target. Maximum velocity difference
from the independent FP64 solve is 3.3306691e-15. No reference solution or
FP64 values enter the native solve; it starts from the captured failed FP32
velocity and consumes the original compiled FP32 problem.

The matched ablation discards each state's low part after the same update.
After four updates its residuals remain 5.8327862e-8 and 2.1846683e-10.
Rounding the successful split solutions gives those same residuals. Thus
improved arithmetic plus retained low parts resolves these two algebraic
problems, while publishing only their rounded velocities loses the strict
certificate. This does not prove that all other FP32 points are infeasible.

All eight retained-problem submissions repeat binary output exactly.
Exact-fraction arithmetic controls, a separate FP64 cone calibration, null
motion, a two-dimensional SPD interval problem and a saturated scalar control
pass. A cone-boundary negative control returns explicit unsupported code3
and preserves the starting state. Fourteen native submissions were made,
including two calibration submissions; there are no independent ecological
units or new physical world transitions.

This is an instrument, not a production solver. It supports scalar intervals
and zero-adhesion uncapped cones in interior/vertex regions. Boundary, cap and
adhesion handling, globalization, persistent high/low state through world
integration, rollback/checkpoint semantics and performance at environment
scale remain open. Its triangular solve is intentionally an instrument-only
serial section. The per-problem GPU times are diagnostic, not end-to-end RL
throughput or a speedup claim. The current world ABI remains single FP32.

Two preparation failures are preserved: a reserved Metal identifier caused
the initial compile failure; synthetic block packing omitted sixteen trailing
alignment bytes and was rejected by the parser before those native controls.
The original sources/plans are reachable in Git history. All earlier native
observations were retained; only the three unsubmitted controls ran after
the padding repair. Predictions and native arithmetic did not change.

Model revision: retain the split-representation mechanism for these exposed
problems. Reject a plan that checks the split residual then silently publishes
a rounded FP32 state as satisfying that certificate. Next native work needs
the low-part representation propagated through its consumers and a generic
boundary projection; qualify physical outcomes and throughput before promotion.

No production source, native setting, tolerance, protected policy or deployment
changed. The coupled physical crow fixture is still unqualified at substep
seven. The full lifelong embodied environment goal remains open.
