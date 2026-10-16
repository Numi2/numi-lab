# Native535 first-failure adjudication

This read-only audit binds the source201 fresh-from-zero full-coupon attempt and locates its first actual failure. It does not promote the accepted prefix to completion.

The native run exited 2 at the expected failure, with no timeout, disk abort, termination error, or changed source/build inputs. It accepted 17 controls, then failed control step 17 in environment 0 (the bent case), at microstep.solverIteration 34. The status is nonlinear solver failure 10. The 64 FGMRES true-residual restart records for that Newton correction are healthy, and the final replacement residual is 1.19334089e-7 against RHS 1.63952757e-6.

The failure record maps directly to the FEM residual Armijo path in the bound source: a baseline free-row residual of 1.6395273405578337e-6, a last tested trial residual of 1.6396469391111168e-6, alpha 2^-23, and a 24-trial exhaustion count. The last measured trial residual is about 7.295e-5 above the baseline. The preceding 23 trial norms were not captured, so the package does not claim a complete line-search curve or explain the non-descent.

The first subsequent status-gated true-residual record appears at iteration 35 with solver status 10, operator status 4, and true_l2=-1; records through iteration 63 are invalid after the failure and are not physical residual evaluations.

For source198/native521 comparison, all 36 common observation rows and all 35 common accepted geometry files are byte-identical; both have the same 17-control bent accepted prefix. This is descriptive prefix parity only.

## Bound inputs

The audit script verifies all 817 source files against source201's frozen map and all build artifacts against its build record. It hashes the command package, runner receipts/logs, environment contract, output result/observations/manifests, rejected candidate files, full stderr log, and the source line mapping. compare-prefixes-vs521.json is the existing source535 comparison tool output.

Run:
 /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 -B /Users/n/cardboard-implementation-20261014-host-macmini/analysis/native535-first-failure-adjudication-001/audit.py

No GPU was launched by this audit. This failed run is not a completed coupon and does not establish convergence, calibration, assembly qualification, shipping-scale admission, or any of the six qualification axes.
