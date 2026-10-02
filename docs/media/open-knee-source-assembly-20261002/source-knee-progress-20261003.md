# Source-knee progress, 2026-10-03

The complete source program remains assembled in the existing Matter runtime: 12 tissue volumes, 9 rigid bodies, 6 source cylindrical joints, the rigid spring, 29,427 source tissue ties, 406 discrete springs, 18 sliding-contact pairs, and the source prestrain/flexion curves. The runtime's sliding-contact admission guard remains fail-closed.

The current CPU cook-only run confirms those inputs compile, but reports the discrete spring curves and prestrain/flexion schedule as `not_stepped`, initialization as `not_solved`, and source equivalence as rejected. This is why an assembled program is not yet the requested source continuation.

## Matched XPLT state-1 comparisons

The reference is retained FEBio 2.9.1 XPLT state 1 at continuation time 0.05. Matter values below come from first assembly at the imported source pose or from a rejected Newton candidate. They are not Matter-predicted accepted states.

- The split-precision/tie-aware force comparison covers all 12 tissue domains. RMS Cauchy-stress differences range from 6.38e-6 MPa to 7.09e-5 MPa; MCL is 5.94e-5 MPa and LCL is 7.09e-5 MPa. See [the hash-bound stress comparison](full-tissue-state1-precise-tie-candidate-20261003.json).
- The rejected candidate contact comparison finds 617 active Matter faces versus 619 in XPLT, with a 2-face support difference and 1.934% relative RMS pressure difference over the active-face union. The TBB–MCL surface has the two support differences and 0.0147 mm RMS gap difference. See [the contact comparison](contact-state1-rejected-candidate-20261003.json).
- Archived patellar/rigid poses and prestrain fields are available as source references, but these imported-pose comparisons do not establish Matter's predicted reaction wrenches or displacement. Those outputs must be compared after an accepted Matter root.

## Solver diagnostics

The 14-Newton/20-FGMRES tie-aware diagonal diagnostic reduced the relative residual from 0.00128078 to 0.000923607 and the rigid-block residual from 0.00151316 to 0.00043479. The root still rejected with zero accepted microsteps; residual remains concentrated in MCL, medial meniscus, and the rigid block. Increasing the FGMRES budget to 32 with restart 10 did not improve the final residual. The dense six-coordinate tie block was worse and has been removed. Retained outputs and exact hashes are in the neighboring `tie-aware-precondition-*` and `dense-tie-block-precondition-*` receipts.

A full-restart diagnostic was interrupted when another Metal workload became active. It produced no result and is not evidence. The source tree was restored to the fail-closed contact guard and default restart, then the full knee target rebuilt successfully.

## Current qualification

No accepted Matter preload or flexion state exists. Therefore accepted-state rigid poses, reaction wrenches, contact pressure/gap, tissue stress/prestrain, and displacement have not been compared with XPLT. The passing compiler tests and imported-pose field comparisons do not close that gate. The next required result remains an accepted coupled source root under the certified source contact path, followed by checkpoint-by-checkpoint field comparison and continuation.
