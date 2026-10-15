# Source sign and dispatch proof for the joined Newton-center capture

This note supplements `report.json`; it does not change either native capture or the report's numerical results. The capture is observational and is not a solver or physical qualification.

## Bound inputs

- Frozen source: `/Users/home/cardboard-implementation-20261014/snapshots/source-center-action-twoenv-restart-099`.
- Runtime source SHA-256: `bc62df9185a73aea95bb9c557e8de8b173a446b09c8b0588d09355c276ebd0ba`.
- FEM Metal source SHA-256: `f4b5cd60dbe7ace7a54636746a00b5e7186d6c1f4890eb0718c138af9c753ae4`.
- FGMRES Metal source SHA-256: `eef8a0919636c513716528dcf04cf9d175801c25e7159125808cbb59a9b696d6`.
- Contact Metal source SHA-256: `3cef6ae3dbd384d7b26ff6b0eb4ecda6da7a32be74cdea0badc5975eef692132`.
- Center payload SHA-256: `ba17e850e84f9e4a7d3daa66e0159789d7cf2484485dc2ba1ebbaab2e1411b9e` (root 1201, Newton 12; manifest reports 145 arrays / 11,670,948 bytes).
- Private-trial payload SHA-256: `58206c37aa9cd897d490a7dd45979d887b0e2f44992fefad512e943fb10d17ff` (root 1201, Newton 12, trial 22; manifest reports 129 arrays / 19,467,248 bytes).
- The original sign-corrected report SHA-256 is `761d838757c470ad0c9085e5f420495b1e9f87ab79d4845fe89e8db29da5d4f3`.

## Sign derivation from the producer

`nm_fem_build_mechanical_residual` forms the mechanical right-hand-side lanes as

`b_FEM = dt * (internal force + external force + gravity force) - M * (v_current - v_accepted)`

at `matter/src/metal/fem.metalinc:2799-2821`. Therefore the physical momentum-balance residual is `R_FEM = -b_FEM`; prescribed rows are explicitly zeroed in the solve RHS after their reaction is published.

The rigid rows follow the same convention: the production residual assembly clears the generalized rigid rows, accumulates contact/support impulses and external-wrench impulses, then subtracts the inertial action (`runtime.mm` around lines 17430-17510; `fgmres.metalinc` around lines 4530-4645). Thus `b_rigid = contact impulse + external impulse - inertial impulse = -R_rigid`.

Before FGMRES reuses `femResidual`, `nm_fgmres_preserve_true_rhs` copies it byte-for-byte into the dedicated RHS arena (`fgmres.metalinc:5211-5221`; dispatched in `runtime.mm:17743-17764`). The rich center capture reads that preserved arena, not the post-FGMRES work buffer. Its manifest explicitly records `full_rhs_before_fgmres` as assembled Newton RHS `b` and physical residual `-b` (`runtime.mm:16343-16350`).

The private-trial capture's `residual_before_armijo_measure` is from the same residual assembly convention: trial mechanics, cohesive, contact, and free-body terms accumulate into the same residual/RHS arena. Consequently the observed physical change is

`R_trial - R_center = (-b_trial) - (-b_center) = b_center - b_trial`.

The center action is a same-generation production-operator mirror: after the final FGMRES restart, runtime clears a diagnostic action arena and re-dispatches the production FEM element, gather, cohesive, contact, and rigid action over the captured correction (`runtime.mm:20399-20405` onward). The captured semantics identify `J*delta` as the operator action, with `b - J*delta` the linear residual (`runtime.mm:16343-16345`). The Newton equation is `J*delta = b = -R`; hence the physical residual prediction is `R_trial - R_center ~= alpha * J*delta`. The sign used in the 003 report is correct.

This action is a fresh same-generation matrix-free operator dispatch, not an independent FP64 oracle and not a byte copy of a particular Arnoldi product. It uses the Newton-center candidate, correction, accepted constitutive/cohesive state, the selected lagged contact inputs, and the configured tangent-cache policy. The selected-trial report verifies that the center-used and trial-used contact and deformable-contact history bytes match. These records describe algorithmic lagged impulse/frame estimates; they are not accumulated physical tangential slip.

## Interpretation boundary

The joined trial 22 (alpha `2.384185791015625e-7`) is a healthy observational trial. The report's measured FEM and rigid blocks are retained verbatim; the prediction mismatch by itself does not localize an operator defect. Larger alpha trials 326-330 and bracket trials 333-334 fail during private deformable-contact rebuild, so their partial residual vectors are not admissible derivative samples. Those CCD failures are handled as a separate investigation and are not incorporated into this sign proof.
