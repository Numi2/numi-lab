# Source101 diagnostic-run physical-output parity: 360 vs 352

This report compares two **failed physical attempts** from the frozen source101 build. It tests whether the selected diagnostics changed the ordinary produced files; it does not qualify the solver or turn either failed run into a successful simulation.

## Result

All 37 files under each run's `output/` directory are byte-identical. The compared tree includes both accepted-step meshes, the rejected-step diagnostic mesh, material state, observations CSV, final result, and the rejected-conditioning P/AP, candidate, RAP, pivots, restart state, and factor-valid files. Source/build/input archives and physical argv match; argv differs only in the declared output directory. The two receipts report exit code 2, no timeout, two accepted steps, then rejection at step 2 with `physical_validation=false`.

The Armijo traces are retained but cannot be treated as on/off telemetry parity: run360 selects control root 2, while run352 selects root 0. Both have 64 iterations, but they describe different physical Newton states. After removing only the root field, no record matches; the comparator makes no telemetry-parity claim.

Run360's operator-FD capture is also non-qualifying. Environment 0 has a nonzero direction and operator status 0, but both tested epsilons have plus/minus stage status `[5, 5]` and null residual norms, so the centered derivative comparison was not evaluated. Environment 1 has `[0, 0]` stages and finite residual values, but its direction is exactly zero. That is an inconclusive zero-direction control, not a derivative pass.

The parity result therefore means only that these diagnostics did not alter the ordinary output files for these two failed attempts. It does not establish that every diagnostic is safe in every execution or that the physical run converges.

## Inputs and evidence

- Frozen source: `/Users/home/cardboard-implementation-20261014/snapshots/source-boundary-reset-order-101`
- Source build provenance SHA-256: `b40d30669b286449a3c5d50da2c865a052cd69c1e9f556132f0fc4bc66cd9fb0`
- Diagnostic attempt: `/Users/home/cardboard-implementation-20261014/solver/reset-augmentation-first-failure-capture-360`
- Control attempt: `/Users/home/cardboard-implementation-20261014/solver/reset-order-augmentation-complete-coupon-352`
- Operator-FD payload: 38,105,896 bytes, SHA-256 `548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33`
- Machine-readable result: `report.json`; the run comparator is `compare.py`.

The report records hashes for predictions, receipts, logs, the FD payload when present, observations, result files, and every ordinary output file. Run `python3 compare.py` to revalidate against the retained runs. The comparator rejects changed archived inputs, source/build identity, physical argv, undeclared environment differences, or differing output bytes.
