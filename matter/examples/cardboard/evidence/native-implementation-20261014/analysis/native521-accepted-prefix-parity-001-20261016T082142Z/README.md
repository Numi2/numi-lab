# Native521 accepted-prefix parity audit

This report copies only the shared accepted-control prefix (controls 0-16) from native521 and checkpoint480. Compact failure/solver telemetry for attempted, rejected control17 is kept separately; rejected geometry is excluded from the prefix.

All 34 accepted OBJ files and 34 observation rows are byte-identical. The first divergence is bent control17 (status 10, failing index 34). The held-reference row with status 0 is not a global acceptance.

521 is terminal, exit 2, with 17 accepted controls. At root17/Newton34 the FGMRES true residual reached 1.19334089e-7 with solver/operator status 0, but the nonlinear trial residual increased from 1.6395273405578337e-6 to 1.6396469391111168e-6 at alpha 1.1920928955078125e-7. The line search exhausted. Damped-CGLS event records have zero candidate norms; no applied damped direction is evidenced.

Source198 configuration, checkpoint480 configuration, source182-bound 497/498 resume metadata, accepted raw bytes, and selected stderr telemetry are bound in report.json and manifest.json. No full trajectory or physical qualification is claimed.
