# Native split-precision product-cone solver experiment

This bounded research instrument extends the previous two-FP32 prototype to
uncapped boundaries, adhesion, capped boundaries and scalar endpoints. It uses
the corrected cap curvature, deterministic SIMD32 sums, two-FP32 LDL and
line search entirely in native Metal. No physical transition uses this kernel.

The two retained coupled-contact problems are inputs, not fresh trials. Sixteen
valid problems, one rounding ablation and one invalid-regularization control
are each submitted twice. Repeats test determinism only. Constructed controls
choose free velocity from a target, then pack all inputs into FP32 and solve
independently from an unrelated initial state. Native inputs contain no FP64
reference solution. All seven product-cone regions and exact scalar endpoints
are included. Predictions and budgets are sealed in plan.json before native
observations; runtime.json binds the actual toolchain and generated binaries.

The native state is a high/low FP32 expansion. Independent FP64 gradient,
projection, derivative, feasibility and Newton checks evaluate the actual
expanded result. They also measure the residual after FP32 output rounding.
An internal split certificate does not certify the rounded world ABI.

No tolerance relaxation, production/world integration, physical ecology
qualification or policy promotion is part of this experiment. Failure remains
evidence and is archived without unplanned retries or expanded budgets.

## Observed result

All predeclared predictions passed: 16 valid cases at the native and independent
1e-10 residual/feasibility gates, all seven regions, exact repeated status and
scratch outputs, typed invalid-r rejection with unchanged initial velocity, and
rejection of the rounding ablation after 20 Newton updates. Worst combined
residual is 3.71097e-12, independent feasibility 3.55e-15, reference velocity
error 4.44e-12, scaled impulse error 3.86e-15, scaled derivative error 4.23e-15
and scaled gradient error 4.78e-15. Independent finite-difference derivative
calibration error is at most 3.37e-11.

The retained substep-7 problem converges in one native update (independent
residual 2.46e-15), as does substep-2 (1.19e-17). Their rounded FP32 residuals
remain 5.83279e-8 and 2.18467e-10. Dropping low parts during optimization
leaves substep-7 rejected at 5.83279e-8. These observations bind this experiment;
they do not prove global FP32 infeasibility or a new physical trajectory.

Thirty-six native submissions consumed about 2.4 seconds of wall time, including
process/device setup and independent analysis. Retained successful solves report
about 5–13ms of GPU command time in this instrument. These are individual
observations, not production throughput benchmarks or performance qualification.
The solver has device scratch and a lane-zero triangular solve. It is not a
production replacement.

The first prebuild validation stopped before compilation because preparation
stdout was still open when its hash entered the plan. The original seal,
failure and amendment are retained. Excluding the log stream changed no input,
prediction or native budget. Compilation and the subsequent first native
execution succeeded.

No production source, runtime, numerical tolerance or deployed policy changed.
The strict coupled crow remains unqualified. Next work must reconcile canonical
state precision with the public FP32 ABI and qualify end-to-end native behavior
and performance before promotion.
