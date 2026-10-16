# Tet 172 high-precision derivative audit

This CPU-only audit isolates the remaining run019 finite-difference case: paper row 172 / tet 172, liner material index 0, row-major deformation-gradient component `F[2,0]` (tangent column 6). Inputs are the exact accepted-history row from native capture 469, the exact run016 center deformation and 28 FP32 parameter words widened to binary64, and run019's exact seven realized plus/minus endpoints at quarter powers 30.5 through 32. The center deformation is bit-identical between the 469 capture and run016 archive.

At the center, the independent mpmath return map converged from the accepted-history elastic seed at 70, 90, and 110 decimal digits. All runs took 12 Newton iterations and met the unchanged `2e-12` scaled residual and `2e-11` dynamic correction gates: residual `5.8850742457490186e-16`, correction `7.2216478842916452e-13`. Center state differences were `4.69e-73` (70 vs 110 digits) and `2.10e-92` (90 vs 110). Full 9x9 MP IFT tangents agreed to relative L2 `2.89e-69` and `9.38e-90`, respectively. The strict source173 v2 compensated-binary64 tangent for column 6 agrees with the MP110 IFT action to relative L2 `6.20e-13` (action norm `2.6035533256517586e7`). The center has positive multiplier increment and positive yield residual; all 14 MP endpoints at seven powers remain on the same branch and meet both strict gates.

The MP90 centered secant converges smoothly toward the MP IFT action: relative L2 error falls from `6.12e-7` at power 30.5 to `7.63e-8` at power 32. The source173 strict compensated-binary64 centered secants pass `1e-6` only at the isolated powers 30.75 (`8.89e-7`) and 31.75 (`8.08e-7`); the other five miss, with a maximum `1.23e-5`. This reproduces run019's nonadjacent-pass classification. It does not close the FP64 finite-difference gate.

For each endpoint, the audit also evaluates stress with ordinary FP64 and with MP at the *same exact FP64 projected state*. The same-state stress discrepancies are about `1.1e-7` to `2.3e-7` Pa. Their endpoint-difference contribution to the centered secant is 20–321 Pa per unit F, while the separately measured MP-root versus FP64-root projection shift contributes only about `0.0002–0.0355` Pa per unit F. The smooth MP secant signal is about `2.60e7` Pa per unit F. This isolates endpoint stress-evaluation rounding/cancellation as the dominant source of the FP64 secant variation for this selected row/direction, rather than a center tangent mismatch or branch transition. These are finite deterministic comparisons, not an independently measured repeatability or absolute-noise floor.

This result resolves the smooth derivative for this one row/direction against an independent 70/90/110-digit IFT reference. It does not qualify the full nine-basis derivative sweep, other rows, native Metal accuracy, assembled operator, coupon convergence, calibration, or physical box behavior. All original strict gates and failed run019 points remain unchanged.

## Reproduction

Run from any directory with the qualified Python and numerical dependencies:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 /Users/home/cardboard-implementation-20261014/analysis/native174-tet172-mp-iftfd-005/audit_tet172.py
```

The script verifies row/tet/material joins and all declared source/input hashes before running; it refuses to overwrite an existing report. It dispatches no GPU or native code. The full JSON report retains center/endpoint states, branch values, residuals, corrections, iterations, endpoint stress comparisons, one-sided and centered secants, and hashes.
