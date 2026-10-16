# native555-global-ideal-fd-002

Computes assembled ideal-direction FP64 finite differences around the exact source555 control19/Newton3 center using the same accepted history and captured F-direction/velocity correction, then compares residual secants with independent assembled -A delta_v.

## Result

All 20 endpoint projections passed strict local projection criteria, but the global secant did not resolve the small assembled action: ||A delta_v||=2.72927e-5 Ns while centered ||dR/depsilon|| plateaus near 6.0169e-4 for powers20–26 (about 21.97 relative discrepancy). At 2^-26 the absolute endpoint delta residual is about 8.97e-12 Ns before division by epsilon. This leaves the global finite-difference comparison inconclusive under projection/cancellation floor; it does not show a bad captured native residual assembly.

## Reproduction

From /Users/n/cardboard-implementation-20261014-host-macmini run:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 -B analysis/native555-global-ideal-fd-002/run.py

The command used the qualified remote Python 3.13 environment and one CPU thread. It performed CPU reference/reconstruction work only. Source, capture, and native run artifacts are bound by report.json and the file manifest.

The initial output-directory admission error is retained in attempt-001.json; it occurred before analysis. The corrected run completed successfully. No production files were edited.
