# Selected capture360 MP90 projection and derivative audit

This is a bounded CPU-only audit of three tetrahedra from the exact source101 capture360 input (root 2, Newton iteration 23). It asks whether the selected FP64 projection/tangent discrepancies survive a high-precision local-root comparison, and whether a failed finite-difference comparison is branch-local.

The runner is [replay_selected.py](replay_selected.py). It replays the exact captured float32 high/low deformation-gradient words, accepted material history, parameter words, captured `grad(delta_v)` per second, and timestep. Each endpoint uses the same accepted history. Perturbations are `F± = F +/− epsilon * dt * captured_grad_delta_v`; the reported rate direction is recovered from the realized FP64 endpoints. MP90 endpoint stress subtraction and division are done in MP arithmetic before any float conversion. The MP90 center and endpoint roots require scaled residual and Newton correction at `1e-70`, with a bounded 48-iteration endpoint solve. The separate stock FP64 reference is reported as implemented, including its `2e-10` residual stop.

## Findings

All three independent FP64 center roots closely match the MP90 root, stress, and directional tangent:

- Medium tet 1635: state `1.18e-12`, stress `1.16e-12`, tangent `3.76e-13` relative L2. The captured native cached action is `9.79e-7` relative to MP90; its full 9×9 native matrix differs by `2.66e-8`.
- Medium tet 679: state `1.14e-13`, stress `1.14e-13`, tangent `1.15e-14`. The captured native directional action differs by `5.63e-6` while its full matrix differs by `4.91e-8`; preserve this as a separate cancellation-sensitive native action discrepancy.
- Liner tet 457: state `1.88e-13`, stress `1.87e-13`, tangent `5.10e-13`; captured native action and full matrix differ by `2.57e-8` and `2.85e-8` respectively.

At tet 1635, the center yield gap is only `+6.96e-10`. At target perturbations `1e-5` through `1e-6`, the two projected endpoints have negative yield gap; at `3e-7` the branch sign returns but both bounded MP projections fail to converge. The first converged, branch-stable pair is `1e-7`, where the MP90 central FD differs from the MP90 tangent applied to the realized direction by `3.67e-7`; at `3e-8` that error is `3.30e-8`. Thus the coarse-step miss is explained by crossing/unresolved endpoint projection for this selected row, not a center tangent mismatch.

At tet 679, endpoint projections converge and remain branch-stable through the sweep. The MP90 central FD error against the realized direction is `5.12e-10` at `1e-6`, falling to `4.61e-13` at `3e-7`. At liner tet 457, endpoints converge and stay on branch from `3e-8` downward; errors are `4.60e-9` at `3e-8` and `5.11e-10` at `1e-8`.

Every target, endpoint status, branch label, one-sided gap, and raw captured input words is retained in `output/report.json`. Branch-changed and nonconverged rows are not treated as derivative passes.

## Superseded audit

`source101-capture360-selected-mp90-001` is retained. Its center MP90 and center tangent comparisons remain useful, but its MP finite-difference values are superseded: that version converted endpoint MP stresses to float64 before subtracting them. Version 002 subtracts endpoint stresses in MP arithmetic first. No prior result file was overwritten.

## Limits

This addresses only tet1635, tet679, and liner tet457 from one rejected Newton candidate. It does not reclassify the other captured rows, close the 1e-6 per-element gate for the native directional action at tet679, establish an absolute FP32 noise floor, compare all nine directions per tetrahedron, or qualify the coupled solver, full coupon, or physical behavior. The native action error at tet679 remains an open result; the full-matrix norm is not a substitute for that directional-action gate.

## Reproduction

Run `python3 replay_selected.py` with NumPy and mpmath available. The runner refuses to overwrite its output directory and checks the capture SHA before execution. `python3 -m py_compile replay_selected.py` passed before the recorded run. Run time was 241.17 seconds on the recorded host.
