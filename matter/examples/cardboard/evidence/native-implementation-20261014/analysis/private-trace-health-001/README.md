# Private Armijo trace health audit

This package applies a strict CPU adjudicator to the retained native logs for runs 346, 347, 348, 333, and 334. `report.json` preserves all 32 raw Armijo rows per run, the raw source logs and predictions/receipts, source-snapshot/build-manifest hashes, capture-manifest hashes, and decoded candidate status records. Capture binaries remain at the original native paths recorded in the report; their byte counts and SHA-256 values are bound by the copied manifests.

## Adjudication

The solver status written by the Armijo decision is the authority for Newton-sequence health. A trial status can be stale or can describe a candidate rejected while the center solve remains healthy, so it is retained but never used to mark the whole sequence fatal. The row's `base_l2` is measured before that row's decision. The first-fatal row can therefore have a valid predecision center, but its failed search is not a residual pair. Rows after that fatal decision are failure-latched/incomplete and cannot serve as a center residual.

For a selected private capture, the candidate phase status is checked separately. A healthy center does not make an infeasible candidate suitable for derivative comparison.

## Results

- Runs 346, 347, and 348: first solver-fatal decision is Newton 7. Newton 7 has a finite predecision base residual `3.77258448e-6`; its line search rejects with solver status 10 and private trial status 6 at index 283. The reported `0.0100020869` base values at Newton 8 onward are not valid center residuals. The private capture in 346 is selected at Newton 12, after the fatal decision, even though its captured selected phase reports status 0. The center capture in 348 is also selected after the fatal decision. Both comparisons are rejected.
- Runs 333 and 334: Newton 12 is still before the first solver-fatal decision at Newton 13. The selected Newton-12 base is valid, and the trace shows its smaller accepted alpha `2.38418579e-7`. The separately captured private candidates 20 and 21 fail during private contact rebuild with status 6, failing index 331, diagnostics `[88, 127, 0, 0]`. These candidate comparisons are rejected even though the Newton-12 center is valid.
- All five runs have native receipt exit code 1. They remain failed physical attempts; this package does not convert a diagnostic trace into an accepted simulation result.

The raw rows preserve the `trial_status` value in every iteration. For example, several accepted rows have a trial-status code of 10. Treating that field as the sequence fatal indicator would misclassify the run. The adjudicator instead uses `solver_status_code` and independently evaluates the selected captured candidate status.

## Source interpretation

In the source bound to runs 346–348 (`source-boundary-reset-order-101`), `nm_fem_armijo_decide` writes solver status 10 after exhausting the authored Armijo search; the sequence continues to emit diagnostic rows. The mechanical residual builder returns early when the environment is unhealthy. Consequently, later `base_l2` records are partial failure-latched metrics, not recomputed physical center residuals. The rigid inertia residual path can still write its rows while external-wrench work is skipped under unhealthy status, which is another reason not to interpret the post-fatal value as a physical state change.

The source bound to 333–334 (`source-center-action-twoenv-restart-099`) has the same solver/candidate distinction. Newton 12 was accepted by the full solver, but the separately selected larger private candidate fails at contact rebuild. Its later status cannot invalidate the accepted Newton-12 center; it does invalidate that private trial as a derivative comparison sample.

## Reproduce the CPU adjudication

```sh
python3 -m unittest -v test_trace_health.py
python3 build_report.py
```

The synthetic tests cover healthy runs with stale trial status, private-candidate rejection while the solver remains healthy, the first fatal row, post-fatal rows, malformed or reordered iteration sequences, capture root/iteration mismatches, and bounded/aligned status payload reads.

This is diagnostic adjudication only. It does not alter solver code, relax acceptance criteria, re-run a native transaction, or qualify a successful physical step.
