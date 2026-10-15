# Source101 capture360 material-direction audit (CPU-only)

This is a bounded independent-equation audit of one rejected native Newton linearization, not a rerun of the solver. It uses source101 capture360 (control root 2, Newton iteration 23, environment 0) and keeps the captured accepted material history fixed. It includes 1,152 liner and 576 medium tetrahedra; glue/interface elements are excluded.

Run:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/source101-capture360-directional-fd-003/audit_rows_and_tolerance_subset.py
```

The script checks the capture hash before reading it. `all_1728_row_metrics.csv` gives each element's independent-reference action denominator, central finite-difference absolute and relative error, native cached-action error, branch labels, and projection residuals. `output/report.json` includes global and worst-row comparisons, exact zero-denominator count, the twenty smallest positive reference-action denominators, per-material summaries, and a preselected ten-row projection-tolerance sensitivity subset. The independent projection threshold sweep uses only CPU reference solves; it does not alter the native constitutive law or acceptance settings.

## Results

At a per-element perturbation norm `||delta F||_F = 1e-6`, `F± = F0 ± epsilon_i * dt * captured_grad(delta_v)`, and `delta(Fdot) = ±epsilon_i * captured_grad(delta_v)`, all 1,728 plus/minus projections completed with zero branch-label changes. The central-difference directional action differs from the independent implicit tangent by global relative L2 `3.1027e-7`, but the worst element is medium tet 1635: relative `3.6721e-5`, absolute `2.4650e8 Pa/s`, denominator `6.7128e12 Pa/s`. Thus 34/1,728 rows exceed a `1e-6` per-row criterion (0/1,728 exceed `1e-3`). The global number does not erase the worst-row failure.

The captured native cached tangent action against the same independent tangent has global relative L2 `1.3194e-7`, but worst element medium tet 679 is `5.6304e-6` relative and `2.4172e7 Pa/s` absolute over a `4.2932e12 Pa/s` reference denominator. 54/1,728 rows exceed `1e-6` (0/1,728 exceed `1e-3`). These are one captured direction per element, not all nine independent matrix directions.

There are no exactly zero reference action denominators. The smallest is medium tet 856 at `8.5816e10 Pa/s`; the median is `5.7527e12 Pa/s`. The report lists the bottom twenty positive denominators without excluding rows or imposing a made-up near-zero cutoff. The maximum FD error is not caused by the smallest denominator: tet 1635's denominator is `6.7128e12 Pa/s`.

In the bounded reference-tolerance subset, tighter stopping is not uniformly a monotone fix: some rows change the multiplier-positive branch at `2e-12`, and several selected solves fail to reach `2e-14` within the reference's 80 iterations. Those outcomes are retained per row. They mean branch-local finite-difference and stopping sensitivity still need resolution; no repeated-run/noise floor is inferred.

Separately, the parent-reported native368 assembled-operator replay found cached-vs-fresh action relative difference `0.329792%`, with fresh action exactly matching capture, while both cached and fresh `b-Ax` norms (`2.161252895e-4 Ns` and `2.157817650e-4 Ns`) remain far above the `1.1993e-5 Ns` target. This is a distinct assembled Krylov residual finding, not evidence that the material tangent passes or fails by itself.

## Provenance and limits

- Capture: `/Users/home/cardboard-implementation-20261014/solver/reset-augmentation-first-failure-capture-360/operator-capture`, SHA-256 `548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33`.
- Source snapshot: `source-boundary-reset-order-101`; `frozen-source.json` SHA-256 is recorded in `manifest.json`.
- Independent equations: `bilinear15_reference.py`; its exact SHA-256 is in the report and manifest.
- Raw JSON integer token `-0` is decoded as binary32 negative zero before widening. Capture input and result row IDs are retained.
- The finite differences are FP64 independent-equation endpoints, not native FP32 endpoint perturbations. No native noise floor was measured.
- The capture is from a rejected Newton 23 candidate, not an accepted step or physical result. It does not qualify a full coupon, interface, assembly, or calibration.

An earlier full 1,728-row sweep across three tolerances and five perturbation scales was CPU-bound with no partial output. Its command, source hash, observed process state, and SIGTERM disposition are preserved in `tolerance-sweep-process-control.json`; it is explicitly excluded. It was replaced by the bounded ten-row sensitivity sweep in this package.
