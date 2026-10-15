# Source104 run 362 vs source105 default-off run 369

## Verdict

**`FAIL_UNEXPECTED_METADATA`**. The production payloads are byte-identical after masking only the source-bound `device_program_fingerprint` word, but the center-capture manifest has one unexpected metadata difference: `rhs_preserved_before_fgmres` is `true` in run 362 and `false` in run 369. The field was not normalized. Both native attempts exited 1, so this audit does not establish a successful physical cycle.

## Inputs and protocol

- Baseline: `/Users/home/cardboard-implementation-20261014/native/health-and-vt-marker-firstfatal-362`, source104 snapshot.
- Candidate: `/Users/home/cardboard-implementation-20261014/native/true-residual-default-parity-369`, source105 snapshot.
- Both runs executed, did not time out, reported no termination errors, and preserved their inputs.
- The normalized physical argv and prediction environment match. Both use `--sparse-free-body-parity --diagnostic-state-output --extended-budget --compensated-material-arithmetic` and capture root/iteration `1201:7`. The candidate environment contains no `NM_FEM_TRUE_RESIDUAL_*` setting; this is a default-off comparison.
- Source inventories are 778 vs 780 files. The exact reviewed source delta is three changed files (`matter/CMakeLists.txt`, `matter/src/metal/fgmres.metalinc`, `matter/src/runtime.mm`) and two added true-residual checker/contract files. Both retained source-build receipts and their runtime artifacts match the run prediction hashes.

## Comparison

- The center capture has 145 arrays and an 11,670,948-byte payload. The raw payload and every array descriptor/hash/byte range are identical. `full_rhs_before_fgmres` has SHA-256 `965d38af46af2bbc311cb1534193b8892581cb4d4548936dfc8d7ba9db0cc2e4` in both runs.
- `source_physics_fingerprint` is the same. The `device_program_fingerprint` differs with the separately built source105 program, as expected. After normalizing only that provenance field and the capture output path, the one remaining capture-manifest delta is the RHS-preserved boolean above.
- All 32 Armijo JSON records are exact matches. `stdout.log` is exact; `stderr.log` and the dense control logs match after replacing only each run’s output directory.
- All three transaction JSON documents are exact. All six 904,359-byte before/after transaction payloads match at every byte except offsets 12–19, which serialize the expected per-build device-program fingerprint. Those 8 bytes are verified against each capture manifest’s fingerprint.

The physical payload parity result is separate from the overall verdict. A metadata bit contradicted by identical preserved-RHS bytes is an instrumentation inconsistency and blocks strict parity. The false bit in run369 remains evidence; it is not repaired in-place or treated as a harmless default value. The separate callback-lifetime candidate is at `/Users/home/cardboard-implementation-20261014/analysis/source105-capture-callback-lifetime-overlay-001`; its fresh native verification is pending.

## Source-bound files

- `audit.py` validates source inventories, receipts/build bindings, normalized invocation, capture payload/arrays, diagnostic transaction state, stderr/stdout, and all Armijo records.
- `report.json` contains the full machine-readable comparison and hashes.
- `test_report_contract.py` asserts the failed metadata verdict and ensures the RHS bit remains visible.

No native run is launched by this audit.
