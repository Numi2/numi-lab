# Tet 172 high-precision derivative audit

This CPU-only audit isolates run019's remaining finite-difference case: paper row 172 / tet 172, liner material index 0, row-major deformation-gradient component `F[2,0]` (tangent column 6). Inputs are the exact accepted-history row from native capture 469, the exact run016 center deformation and 28 FP32 parameter words widened to binary64, and run019's exact seven realized plus/minus endpoints at quarter powers 30.5 through 32. The center deformation is bit-identical between the 469 capture and run016 archive. The script hard-binds the source, equation reader, MP solver, capture arrays, v2 package, run016 center, and run019 records by SHA-256 before reading inputs.

At the center, the independent mpmath return map converged from the accepted-history elastic seed at 70, 90, and 110 decimal digits. All runs took 12 Newton iterations and met the unchanged `2e-12` scaled-residual and `2e-11` dynamic-correction gates: residual `5.8850742457490186e-16`, correction `7.2216478842916452e-13`. Center state differences were `4.69e-73` (70 vs 110 digits) and `2.10e-92` (90 vs 110). Full 9x9 MP IFT tangents agreed to relative L2 `2.89e-69` and `9.38e-90`, respectively. The source173 v2 compensated-binary64 tangent for column 6 agrees with the MP110 IFT action to relative L2 `6.20e-13` (action norm `2.6035533256517586e7`). Both the multiplier increment and yield residual are positive at the center; all 14 MP endpoints remain on that same branch and pass both strict gates.

The MP90 centered secant converges toward its MP IFT action: relative L2 error decreases from `6.12e-7` at power 30.5 to `7.63e-8` at power 32. For the corresponding compensated-binary64 strict projections and ordinary source173 binary64 stress evaluations, the centered secant passes the `1e-6` gate only at the isolated powers 30.75 (`8.89e-7`) and 31.75 (`8.08e-7`). The other five miss, with a maximum `1.23e-5`, reproducing run019's nonadjacent-pass classification. This does not close the FP64 finite-difference gate.

At each exact FP64 projected endpoint state, ordinary binary64 stress and MP stress were evaluated with the same widened state. Their differences are about `1.1e-7` to `2.3e-7` Pa. Differencing those errors gives 20–321 Pa per unit F; the MP-root versus FP64-root projection shift contributes only about `0.0002–0.0355` Pa per unit F. The smooth MP secant action is about `2.60e7` Pa per unit F. For this selected row and direction, endpoint stress-evaluation rounding/cancellation dominates the projection-root shift and explains the FP64 secant variation; the center IFT derivative agrees. These are finite deterministic pointwise comparisons, not an independently measured repeatability or absolute-noise floor.

This resolves the smooth derivative for this single row/direction against an independent 70/90/110-digit IFT reference. It does not qualify the full nine-basis derivative sweep, other rows, native Metal accuracy, assembled operator, coupon convergence, calibration, or physical box behavior. All original strict gates and run019 failures remain unchanged.

## Reproduction

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 /Users/home/cardboard-implementation-20261014/analysis/native174-tet172-mp-iftfd-006/audit_tet172.py
```

The script refuses to overwrite an existing report, dispatches no GPU/native code, and retains center/endpoint states, branch values, residuals, corrections, iterations, stress comparisons, one-sided and centered secants, and hashes.
