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
