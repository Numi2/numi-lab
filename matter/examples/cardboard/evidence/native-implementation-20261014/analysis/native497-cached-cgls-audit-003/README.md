# Native 497 cached-operator direct solve and conditioning audit

This CPU-only extension solves the exact environment-0 FEM block reconstructed in the earlier 497 audit. It validates the retained 497 prediction, receipt, source inputs, CGLS/state captures, and operator-FD capture through the sealed audit-001 reader path before constructing the matrix. It does not launch Metal or alter the solver.

The selected block is a 2,538-row free-xyz system with 71,046 nonzeros: 846 free nodes, 36 fixed nodes, and 108 fixed xyz DOFs eliminated. The captured CGLS selection words are `[1, 0]`; environment 1 is unselected scratch. Lumped mass and the full, unsymmetrized `dt² K` are kept separate. Field backing, rigid rows, contact, and cohesive terms are outside this matrix.

SuperLU sparse LU gives a binary64 solution with `||b-Ax||₂ = 1.12e-17` and normwise 1-backward error `3.50e-18`. The source182 cached matrix has a repeated Hager–Higham one-norm condition estimate of about `3.20e6`; `onenormest` supplies an estimate of the inverse norm, not a certified upper bound. LU diagnostics retain the row/column permutation hashes, U-diagonal range, and U/A maximum-entry ratio.

Against that same direct solution, the captured Newton `x0` has residual `2.10e-7` and relative forward error `1.4760e-3`. The captured total warm CGLS candidate `X` has residual `2.65e-8` but nearly unchanged relative forward error `1.4761e-3`. Casting the CPU direct solution to FP32 yields a replay residual of `2.73e-9`; this is a comparison point, not a native-reproducible candidate. The native candidate’s normwise 1-backward error is `1.90e-8`, while its componentwise backward error is much larger (`4.01e-4`), reflecting small component denominators.

Re-summing the same binary64 element-entry contributions with `math.fsum` changes the assembled stiffness by only `9.76e-17` relative Frobenius norm and the direct solution by `1.24e-12` relative norm. This tests only global duplicate-entry coalescing; local tensor/geometry arithmetic is unchanged. A separate float32 sparse-CSR action is labeled as a proxy, not as native Metal element/gather order.

These results show that the cached free FEM block is sensitive enough for conditioning to matter, but the estimate does not uniquely explain the native recurrence discrepancy or establish a causal fault. The matrix is reconstructed from the captured FP32 production tangent cache, not a constitutive FP64 reference. The retained native coupon segment still exits with its expected failure and accepted zero controls; no whole-coupon convergence or physical qualification follows.

## Reproduction

Run:

```sh
python3 audit_condition_497.py
python3 test_condition_497.py
```

The final adjudicated result is `report-v2.json`. `report.json` is preserved as the first successful exploratory result before adding the complete native run binding to the final schema. The script refuses to overwrite an existing report path.

## Preserved analyzer attempts

The first two execution attempts are recorded in `attempts.jsonl`: one rejected an incorrect prior-manifest-versus-report hash binding, and one used a node-count assertion where the CGLS xyz mask correctly counts DOFs. Both occurred before a report was emitted; neither changed native input or source data. The corrected run validates all prediction inputs and prior audit bindings.
