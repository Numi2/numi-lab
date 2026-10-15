# Source 175 cached-operator audit

**Verdict:** The selected rejected-Newton FEM operator is independently reconstructed with measured native action parity of 1.195e-06. The retrospective environment-0 RAP export is inadmissible for that operator: exported AP matches M·P (2.567e-08) but differs from A·P (119781). The native export's LU reconstructs its own exported coarse matrix at 8.037e-08; this does not repair the mismatch.

The exact failing environment had root 3 / Newton 23, residual Armijo rejected the correction, status 10, and 1024 FGMRES iterations. The overall run exited 2. The retrospective RAP manifest binds control step 3 and marks `accepted_physical_state=false`, but does not bind the RAP arrays to Newton 23 or a tangent-cache generation. The AP mismatch is measured; the suspected post-failure scratch lifetime issue is not proven.

Environment 1 is separate: its AP parity is 9.914e-08, but its captured direction is zero, so action parity is inconclusive and this does not qualify the failing environment.

A separate CPU-only study solved the audited 2538-DOF A with sparse LU (true relative residual 1.563e-12; normwise backward error 1.038e-17). The one-norm condition estimate is 3.84e+06, an iterative estimate rather than a bound. Exploratory overlapping graph-Schwarz tests at 16/32-node patches reached true relative residuals 0.824/0.730 after 64 iterations (restart 16). These separate CPU experiments are not Matter FGMRES, and observed runtime warnings are retained in the JSON.

The strict RAP consumer exited 2 at its A·P parity gate before running any native-RAP GMRES comparison. The exact failure transcript is `rap-admission-failure.json`. The full raw capture, receipt, prediction, stderr, and RAP binary files remain at their original source paths referenced in the report.

For a valid same-Newton comparison, add a read-only regional sidecar immediately after the selected `nm_fgmres_regional_factor_rap` dispatch and its barrier, before FGMRES correction. Bind root/iteration/environment, candidate high+low state and tangent generation, P/effective-P/AP/RAP/pivots/factor-valid/inverse blocks, dimensions/layout, fingerprints, and per-array hashes. Until then, native RAP conditioning is **inconclusive**.

Detailed source/binary/input hashes, metrics, and reproduction commands are in `source175-cached-operator-audit.json`.
