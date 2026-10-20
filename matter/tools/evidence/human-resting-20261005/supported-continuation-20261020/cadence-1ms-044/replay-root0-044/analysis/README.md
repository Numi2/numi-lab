# Cadence 044 solver diagnosis

This compact analysis belongs to the 1 ms cadence failure and its exact diagnostic replay. The invocation and raw logs remain in the parent run directories. No production source or acceptance threshold was changed for this run.

The native run and diagnostic replay have byte-identical `common-field-failure.json` receipts. The replay trace records 24 Newton line-search decisions for microtick 0. Alpha is 0.25 at iteration 0 and 1.0 for iterations 1–23. The nonlinear assembled residual decreases from 0.0115443 to 3.13624e-4 by iteration 3, then stays between about 4.5e-5 and 1.56e-4; the latch stays false and the final nonlinear residual is 9.78602e-5. The unchanged target is 1e-5.

The first nonzero status appears after the microtick-0 vascular certificate snapshot, at the subsequent mixed certificate. `nm_mixed_certify` runs after FEM integration and validation and owns diagnostics `[relativeResidual, relativeCorrection, volumeResidual, pressureResidual]`. Its failure tuple `[1.37253126e-4, 3.74236901e-4, 0, 0]` is consistent with the certificate logic in `matter/src/metal/mixed_fem.metalinc`; the second component also matches the final FGMRES nonlinear correction telemetry. In this scene the certificate denominator is `max(fgmres.diagnostics.y, 1) = 1`, so `relativeResidual` equals the absolute assembled residual. The final gate fails the existing 1e-5 residual limit. The trace does not support an alpha synchronization diagnosis.

`trajectory.jsonl` contains the per-iteration compact values. `summary.json` binds them to the binary, app source, Metal library, and raw replay stderr hashes.
