# Source125 medium tet 1691 endpoint recovery and secant audit

## Verdict

The source125/native424 medium-material plus endpoint for tet 1691 is recoverable under the unchanged FP64 bilinear-15 equations and captured accepted history. The ordinary direct elastic-start reference solve failed; an extended safeguarded solve also stalled after entering a negative plastic-multiplier increment and negative regularized yield gap. Continuation reaches the same exact plus endpoint with four center-to-plus F subdivisions (and eight minus-to-plus subdivisions), satisfying the strict reference residual and dynamic-correction gates. This identifies an initialization/continuation-path failure in the independent reference solve, not absence of an endpoint root.

After inserting that recovered endpoint, all 1,920 tetrahedra (1,728 paper and 192 glue) are present in the independent center/plus/minus assembled secant. Native and independent full secants agree to relative 0.00284769. The independent secant remains relative 1.09494 from center A and 1.05090 from the center tangent on the realized direction. Tet 1691 contributes 3.74338e-4 Ns/epsilon to the residual secant; dropping it changes the incomplete-subset result materially. Thus native FP32 force quantization does not explain the finite-step discrepancy. One finite perturbation does not distinguish curvature from unobserved constitutive branch behavior and does not qualify the derivative.

## Strict reference gates and diagnostics

The equations and source parameters were not changed. The local residual gate is `max(abs(r) * [1e4 x 6, 1 x 6, 1e4 x 3]) <= 2e-10`. The local correction gate is `max(abs(J^-1 r) / max(abs(old), abs(z), p26)) <= 2e-8`. The extended direct attempt allowed up to 128 Newton updates and 64 backtracks as a diagnostic only; it failed with scaled residual 90.8129225 and dynamic correction 2.593487. Its final multiplier increment was negative (`-6.769758e-16`) and the yield gap was negative (`-0.00454103`). Center-to-plus continuation with one and two subdivisions did not reach the strict gates; four and eight did. Minus-to-plus with eight subdivisions passed.

At tet 1691, the recovered plus endpoint has scaled residual `1.25207e-13` and dynamic correction `2.67786e-16`. Its plus yield gap is `4.04685e-8`, versus `1.23849e-3` at center. Lambda remains active and the regularized yield gap positive at the three converged samples, but these two predicates are not a complete branch signature. The tet's finite endpoint force secant differs from its center tangent by 35.66%, consistent with strong finite-step response near yield; it is not itself proof of a discrete branch switch.

The independent reference emits NumPy divide/invalid warnings because its `np.where` expression evaluates an unselected zero-denominator rationalized Fischer-Burmeister branch. Warnings are retained; output finiteness and strict gates were checked separately. No warning was suppressed or converted to a noise floor.

## Binding and scope

All input and source bindings are in `result.json`, `completed-secants-v2.json`, and `branch-sweep.json`. They identify frozen source snapshot `source-status5-vtccd-diagnostics-125`, native capture 419, native residual sweep/audit 424, independent reference implementation, and the previous audit003 files by SHA-256. `manifest.json` hashes every package evidence/script file except itself. Failed intermediate attempts remain in this package and are enumerated in `attempt-log.json`; the successful outputs are `result.json`, `completed-secants-v2.json`, `branch-sweep.json`, `branch-signatures.jsonl.gz`, and `report.json`.

This is a CPU-only, source-bound diagnostic of one finite perturbation and one recovered local endpoint. It is not a native rerun, full solver convergence result, physical calibration, material qualification, or a universal derivative/noise-floor claim. The sealed package does not modify frozen sources or audit003.

## Reproduction scripts

The scripts use absolute paths to the already-bound source125, capture419, native424, reference, and prior audit inputs. In a disposable copy of this directory, run:

```sh
python3 diagnose.py
python3 center_continuation.py
python3 complete_secant_v6.py
python3 branch_sweep_v2.py
```

The earlier failed script attempts are retained for audit. Reruns should use a copy because scripts write their outputs alongside themselves.
