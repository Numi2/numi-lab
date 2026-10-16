# Native 469: independent high-precision center-projection audit

This package independently resolves the six source-166 FP64 center-projection misses from the exact captured inputs. Each row was solved separately at 70, 80, and 110 decimal digits, using the same authored 15-state material equations, the old accepted history, and the elastic trial stress as the initial guess. No native projected state is used as a seed.

The unchanged acceptance limits are a maximum scaled residual of `2e-12` and a maximum state-scaled Newton correction of `2e-11`. All six rows pass both limits at all three precisions. The largest 70/80/110 root spread is `1.83e-71` after scaling each component by `max(|old|, |candidate|, fb_epsilon)`. The exact 110-digit states are in `run-evidence/mp_states.npz`; the report retains decimal strings for the 70-, 80-, and 110-digit states and per-iteration histories.

| Tet | MP residual max | MP scaled correction max | Native-state relative L2 | Native stress relative L2 | Native tangent Frobenius relative L2 | Worst column relative L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 750 | 8.62e-14 | 8.54e-18 | 2.05e-12 | 1.88e-8 | 4.77e-8 | 5.54e-8 |
| 846 | 7.52e-14 | 2.53e-16 | 9.10e-9 | 1.46e-8 | 6.11e-7 | 1.45e-4 |
| 1571 | 2.86e-13 | 2.84e-17 | 2.04e-11 | 3.14e-8 | 1.50e-8 | 5.11e-8 |
| 1599 | 8.15e-24 | 8.07e-28 | 3.37e-11 | 4.28e-8 | 1.61e-8 | 3.88e-8 |
| 1601 | 1.53e-23 | 1.51e-27 | 4.57e-11 | 3.87e-8 | 2.51e-8 | 2.62e-8 |
| 1680 | 8.22e-23 | 8.20e-27 | 7.33e-11 | 2.17e-8 | 5.04e-8 | 5.77e-8 |

The native tangent columns are compared with the existing source-166 complex-step FP64 tangent evaluated at the MP root. This is a local comparison only; it is not a measured native finite-difference/noise-floor qualification. In particular, the tet-846 tangent comparison is provisional because the independent source-166 Jacobian has a demonstrated cancellation defect at that row's elastic seed.

## Tet 846 reference-Jacobian defect

At the exact captured elastic trial seed for tet 846, the rationalized Fischer-Burmeister residual denominator is `1.0953666745999648542e-14`, just above source-166's `1e-14` branch cutoff. Complex-step differentiation through that quotient reports `dFB/dlambda = 18,613,793.6167`; direct differentiation of the authored unsimplified expression gives `dFB/dlambda = -1` because the seed has `delta_lambda = 0`. This is a defect in the independent reference's quotient differentiation, not evidence of a production/native derivative defect. A follow-up reference-only patch is tracked separately; this package preserves the old result and comparison.

## Bindings and reproduction

The final source-bound report is `run-sealed002/report.json`, and its 110-digit states are in `run-sealed002/mp_states.npz`. Run from a fresh output directory with:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/native469-independent-projection-002/solve_mp_projection.py \
  --output /absolute/path/to/new-output
```

Then run the artifact checks:

```sh
cd /Users/home/cardboard-implementation-20261014/analysis/native469-independent-projection-002
/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v test_report.py
```

The solver verifies the SHA-256 of the original native capture, its source-166 arrays/report, reference and reader source, the Haj-Ali medium material asset, the authored constitutive implementation source, and every one of the 787 source-snapshot files recorded in source-166's replay report. `run-sealed002/report.json` records the exact hashes and maps each global tet id to its compact paper-array row. The direct authored FB residual is evaluated in MP; source-166 FP64 residual and complex-step tangent outputs are retained as separate comparisons.

Earlier attempts remain in this package without being overwritten. `run-final` preceded a stability-field naming correction; its artifact test failed because the test compared the threshold field as an observed error. `run-sealed` corrected that field but the test still targeted the stale `run-final` directory. `run-evidence` added the complete 787-file source recheck and is a successful run. `run-sealed002` is the final output bound by the five passing artifact checks and includes the final exact gate labels and source-snapshot binding.

The six original source-166 FP64 failure reasons are retained verbatim in `run-evidence/report.json`. The independent MP results do not repair native code, establish the six other deformation-state projections, close full-assembly derivative gates, or qualify the physical coupon or box trajectory.
