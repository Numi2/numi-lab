# Native 468 source157 dt/2 command004

Preparation only: this package corrects command003's executable path. It has not launched a native process or GPU. Command003 and its preflight refusal remain unchanged.

## Exact correction

The only argument-vector difference from sealed command003 is the executable immediately after `--`: command003 points to the source154 probe, while command004 points to the source157 probe. The `--runtime-dir`, `--build-provenance`, and `--metallib` already point to source157. The corrected source157 executable SHA-256 is `19492c5cff7dd94148335f7ed53a358bbcebcaa163241cfd8012abbac9f765e0` and matches the live source157 build-provenance artifact entry.

The preserved refusal record is `/Users/home/cardboard-implementation-20261014/analysis/native468-executable-preflight-refusal-001/report.json` (SHA-256 `2759ffeaf645a0a2a3cb4a1723b44d369d76790740db3a2f91da42ca1f4f7ef1`). It says the runner rejected the source154 executable before case creation, launch-guard creation, or native execution.

## Scope and protocol

All 103 external inputs are byte-hash bound under `external-input-bindings.json` and remain in immutable package003. The command changes exactly one argv token. The source157 snapshot/build and compile-only evidence are bound by `source-build-bindings.json`; this package performs no build.

The command retains dt `0.00005`, phases 24/6/12/6/6/6, Newton64, CGLS64, restart16, and the conditional residual certificate bound declared by command003. The 64-vs-128 Newton budget remains a diagnostic difference, not pure time-refinement qualification. The certificate corresponds numerically to 0.01 N only while referenceScale remains 1; postrun free-force imbalance still needs separate review.

The capacity file is only a preparation-time observation. Recheck storage, source/build artifacts, case absence, and admission immediately before any dispatch.
