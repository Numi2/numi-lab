# native555-residual-delta-audit-001

Reconstructs captured 555 side residual differences from captured native stress and independently replayed strict FP64 stresses, with inertial mass separated; checks F/Fdot reconstruction from nodal delta-u/delta-v.

## Result

For env0, direct native stress gather plus mass reproduces captured side-minus-center free-row residual to 8.08e-7–3.01e-6 relative; independent FP64 stress reconstruction differs by 8.62e-7–3.28e-6. Captured F is consistent with nodal delta-u at 1.91e-8 relative and Fdot with nodal delta-v at roughly 1.1–1.5e-7. Quiet env1 has zero delta, so relative comparisons are undefined. No solver convergence claim.

## Reproduction

From /Users/n/cardboard-implementation-20261014-host-macmini run:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 -B analysis/native555-residual-delta-audit-001/run.py

The command used the qualified remote Python 3.13 environment and one CPU thread. It performed CPU reference/reconstruction work only. Source, capture, and native run artifacts are bound by report.json and the file manifest.

The initial output-directory admission error is retained in attempt-001.json; it occurred before analysis. The corrected run completed successfully. No production files were edited.
