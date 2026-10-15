# Bilinear 15-state local-projection diagnosis

This external, source-bound analysis compares captured native local return maps with a handwritten FP64 implementation of the authored 15-state equations. It is a constitutive-operator diagnostic, not a calibration result, full coupon qualification, corrugated-board validation, or physical test.

## Bound inputs

- Native source snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-stateless-compensated-and-validation-admission-077`.
- Exact material assets: `hajali2009_liner_hill_bilinear_fit.nmatter` (SHA-256 `a2720b90f2004f2eccf180063e7ecc4e0acdda93f700c4fd357914fc24d2255a`) and `hajali2009_medium_hill_bilinear_fit.nmatter` (SHA-256 `76f2bef552822708c36b5acc9f2bd44e7bb92c578e5e46adb64a1a2e82e47f9`).
- Production implicit evaluator: snapshot `matter/src/metal/common.metalinc`, SHA-256 `0509fe98df6afe6694dec59088ad5d4391177b63133bb38fb0caecb5e1ef7fba`.
- N0 capture is native case 258, SHA-256 `14246b2ade8e301d68b33d97f14eb6468e7f8f7dbc4a36249a83e33a25db2541`.
- N63 capture is native case 266, SHA-256 `13f264ff1bb9688f8ac48f3c940020ff1f1f45d7fbb1713cd2144ba5320b91c2`.
- The handwritten reference is `../bilinear15_reference.py`; the capture-seeded solver is `../diagnosis-001/captured_seed_reprojection.py`. Exact script, report, and prior-report hashes are recorded in `report.json`.

The material residual equations match the authored state order and parameters. All implicit rows carry the same `residual_scale=10`, which leaves the root unchanged. The production compensated-candidate projector uses `2e-8 * (1 + stateNorm)` as its local stopping bound; the material smoothing uses `fb_epsilon=1e-8`. The report re-evaluates the authored equations at each captured candidate and finds the independent scaled residual agrees with the shader diagnostic to about one part per million or better for the highlighted rows.

## Results

For N0 (root 1, Newton iteration 0), all 3,456 captured tets converge when independently reprojected from their captured local candidates at `2e-15 * (1 + state infinity norm)`. The FP64 tangent at the captured candidate agrees with the native tangent to `1.71e-8` global relative L2. Reprojecting to the tighter root changes the aggregate tangent by `1.1785%` global relative L2, with large local shifts concentrated in a small set of medium-paper tets near yield. The 2e-14 and 2e-15 independent roots agree to `6.69e-7` global relative L2; the largest per-tet difference is `2.59e-4`.

For example, N0 medium tet 758 has candidate yield gap `2.3720e-7`, `fb_epsilon≈1e-8`, `Δλ=0`, and scaled local residual `2.1070e-9` against a stop bound near `2.01e-8`. The tighter root moves the yield gap to `1.3683e-5` and `Δλ` to `3.6542e-12`, while the maximum local state change is only `3.44e-8`; the local tangent changes by `10.27` relative L2. Tet 751 is closer to the smoothed transition: its scaled residual is `1.9355e-8`, yield gap `2.4865e-8`, and `Δλ≈4.08e-14`; the stricter root gives `Δλ≈3.70e-12` and a `27.49` relative-L2 tangent change. These rows show a local stopping-accuracy sensitivity at the initial yield transition. The state changes are small, but the local derivative is highly sensitive there.

For N63 (Newton iteration 63), the all-row 2e-14 replay converges all 3,456 tets. Native versus strict-root tangent difference is `7.80e-8` global relative L2, with maximum per-tet relative L2 `3.31e-5`. A supplementary 2e-15 sweep converges 3,453/3,456; the three remaining rows stall at residuals from `2.03e-15` to `2.48e-15`. On the converged common rows, 2e-14 versus 2e-15 tangents differ by `5.20e-13` global relative L2. The complete 2e-14 result is therefore retained for N63; the partial 2e-15 result is only a stability check.

The older reference replay’s row-857 elastic-seed line-search stall is not evidence that the captured native local solution is invalid. That replay initialized from the old state and used a separate row-weighted merit metric (`1e4` for plastic-strain/multiplier rows), unlike the production unweighted authored-row merit. Starting the independent solve from the exact captured candidate and using the production merit converges for the N63 row. Both attempts and their outputs remain preserved.

## Interpretation and limits

The evidence supports a narrow finding: at N0, the local `2e-8*(1+stateNorm)` residual criterion can accept candidates whose smoothed Hill/FB state is close enough to yield that a stricter root produces a materially different algorithmic tangent. At N63, this local projection comparison does not show a material tangent mismatch. This does not establish that local projection is the only or dominant cause of the failed global solve. The separate N63 native linear solve remains underresolved (the bound native report gives true `b-Ax` relative residual about `0.841`, while offline sparse LU reaches about `1.8e-11`); that is a conditioning/Krylov issue and must not be conflated with the local projection result.

A next numerical test should keep the authored constitutive law, `fb_epsilon`, global tolerances, and operator unchanged, and compare a scoped tighter local projection policy against the same source-bound N0 and N63 captures. It must report achieved local residual floors, local tangent comparison, global history/certificate parity, runtime cost, and any nonconverged projection. A requested local tolerance below the device’s representable residual floor must fail explicitly; it must not be silently treated as convergence. No projection-policy change is made by this analysis.

## Reproduction

From the original retained case archive, run:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/bilinear15-fp64-replay-267/diagnosis-001/captured_seed_reprojection.py \
  /Users/home/cardboard-implementation-20261014/solver/sourcefit15-fg32-control1-capture-cycle-258/operator-capture \
  /path/to/new-n0-output

python3 /Users/home/cardboard-implementation-20261014/analysis/bilinear15-fp64-replay-267/diagnosis-001/captured_seed_reprojection.py \
  /Users/home/cardboard-implementation-20261014/solver/sourcefit15-fg32-control1-newton63-capture-266/operator-capture \
  /path/to/new-n63-output

python3 /Users/home/cardboard-implementation-20261014/analysis/bilinear15-fp64-replay-267/diagnosis-002/refined_root_stability.py
```

The first two commands reproduce the 2e-12/2e-14 all-row comparisons; the last command adds the strict 2e-15 stability check. Outputs are written to new paths and do not modify the captured native receipts or source snapshots.
