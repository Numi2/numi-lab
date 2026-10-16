# Native 497/498 and 482 CGLS comparison

This is a CPU-only audit of retained native runs. It does not execute Metal or alter source, tolerances, checkpoints, or solver state. The final comparison is `report-v4.json`; `report.json`, `report-v2.json`, and `report-v3.json` are preserved preliminary report attempts and are not the adjudicated result.

## Result

The diagnostic-on run 497 and diagnostic-off run 498 have byte-identical production outputs and production log records. All 35 physical/production output files match across 482, 497, and 498; the only differing output file is the diagnostic `coupon-resume-segment.json`. The retained 4,096 true-residual cycle records and 64 CGLS direction records match exactly. The two observation rows, output meshes, material state, checkpoint metadata, result and ordinary manifest all match.

Run 482 is source173, while 497/498 are source182. Their frozen-source maps differ in eight paths, and source182 adds the explicit diagnostic-resume option and glue-material binding. The effective solver environment matches after removing the six registered observer controls and accounting for the newly recorded incidence-sort setting being unset. The 482 comparison is therefore recorded as observed parity, not controlled attribution across a source revision.

All three resumed segments fail at control step 20 and accept zero controls in that segment. Bent environment 0 ends with status 10; the held-reference row does not establish a completed bent trajectory. Nothing here qualifies a full coupon cycle or physical cardboard behavior.

## Cached-operator residual scales

For the selected environment-0 FEM-only cached operator, `||M x0||=2.27e-6` and `||dt²K x0||=1.93e-5`, while `||b||=1.88e-5`, `||A x0||=1.89e-5`, and the offline FP64 cached-matrix residual is `2.10e-7`. At total candidate `X`, the mass and stiffness action norms are `2.27e-6` and `1.93e-5`, while `||b-A X||=2.65e-8`. This is clear subtraction cancellation in this reconstructed cache; it does not, by itself, prove ill-conditioning.

The captured recurrence residual vector differs from the offline FP64 reconstructed final residual by `4.90e-9`: 2.34% of the starting residual, 18.44% of the small final residual, and 0.026% of `||b||`. The native freshly recomputed residual scalar is within 0.194% of the FP64 reconstructed norm. That contrast is useful evidence, but it does not distinguish cancellation, conditioning, and recurrence drift by itself. No same-capture condition estimate or direct solve is included in this package.

The smallest next diagnostic is an offline multi-precision residual ladder on the exact retained FP32 cache and vectors, with the same-generation `b`, `x0`, total `X`, native `Q=A X`, fresh residual, and recurrence `R`. Compare FP32/native and FP64/compensated residual vectors and backward error against `||b||`, `||A||·||x||+||b||`, and the target residual, then pair that with a documented condition estimate and forward error to a direct solve of this cached matrix. This leaves the physical operator and acceptance gates unchanged.

## Reproduction and bindings

Run `python3 audit_native497498.py` to generate a fresh report under a new package name, and `python3 test_native497498.py` to check the final report. The audit refuses to overwrite `report-v4.json`. The report and manifest bind the three native receipts, predictions, environment contracts, logs, every output file, the 497 diagnostic payloads, both frozen-source maps, and the prior sealed 497 operator-replay report.

The offline operator reconstruction reuses the exact captured FP32 tangent cache and geometry and is not a constitutive reference or native FP64 execution. It excludes rigid/contact/cohesive unknowns. Preserve the failed/incomplete outcomes as such.
