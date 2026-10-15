# Native433 base-Cauchy kernel adjudication

This package independently adjudicates the completed source133 standalone Metal kernel checker against the exact 419 captured FEM block and its offline FP64 CSR operator. It reads the retained run and source artifacts without changing them. Re-run with:

```sh
python3 -W error /Users/home/cardboard-implementation-20261014/analysis/native433-cauchy-kernel-adjudication-001/adjudicate.py
```

The run is bound to `native-base-cauchy-kernel-result-433/manifest.json`, prediction, receipt, native metallib/build fingerprints, checker source/binary, the 419 replay payload, and the 419 CPU package plus 424 residual-sweep report. The script validates every native result-manifest file and each protected-input hash before reading values; its report contains the complete absolute path, byte count, and SHA-256 inventory.

## Result

The selected environment’s preserved free-node base RHS matches the 419 FP64 reference vector exactly after float32 promotion. The native transpose action has relative L2 error `7.01e-8` against independent FP64 `A^T b`; the native forward action has relative L2 errors `8.02e-8` against `A` times the native gradient and `9.12e-8` against `A` times the FP64 gradient. The float32 Cauchy scalar differs from the FP64 value by `1.02e-7`; the published candidate is bitwise the float32 product `tau * native_gradient` in selected free rows.

The selected fixed rows and inactive field rows are exactly zero. The unselected environment’s operator outputs are zero, and its solution, metrics, state, line-search row, and alpha are unchanged. The selected invalid-cache, nonzero field RHS, and nonfinite free-base-RHS controls fail closed in environment 0 only. The completed-FGMRES-latch control suppresses operator output, while the positive control preserves the pre-clear latch evidence in the trace. The native runner completed normally, left protected inputs unchanged, and reported zero physical transactions.

## Limits

This is evidence for an isolated native Metal kernel path, not runtime fallback dispatch or a completed nonlinear transaction. The 419 state is a rejected Newton candidate. Its compiled world contains no rigid, contact, cohesive, MPM, or field contribution, so only the free FEM block `A = M + dt^2 K` is covered. No Armijo acceptance, full coupled residual/certificate, cycle admission, performance claim, or physical qualification follows. The source-matched 424 sweep reproduced retained endpoints bitwise but did not qualify a finite-difference derivative or a numerical noise floor.
