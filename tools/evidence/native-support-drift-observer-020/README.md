# Accepted support-motion observer

This change adds an opt-in, read-only observer for the accepted whole-body support state. When `NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT=1`, the native run exports body-resolved mass/COM/twist/momentum rows and the already-selected skin-support vertex/point plus its pre-step `J·v` sample. The accepted-state and pre-dynamics step/time labels are both recorded. The observer reuses the existing support debug kernel; it does not alter contact geometry, forces, integration, Brain state, or the physical timestep. Default-off runs keep the prior output path.

## Verification

- The observer binary was built from the source snapshot bound in `build-pins.json` (base `41b254605e66035a4e3966e5a1ae790197431605`). Every compiled source file listed in that pin was byte-identical to the corresponding source file published in the Lab checkout; details are in `source-file-equivalence.json`.
- The focused C++ helper test passed. Its original command and output are bound in `build-pins.json` and `focused-helper-test.log`.
- The 20 s observer-on run passed with the frozen physical library014 and respiration metallib bytes. Its receipt verifies the loaded runtime and unchanged source files. `profile-fix-validation.json` records exact parity for the four legacy CSVs and eight shared geometry packs against the observer-off reference, plus finite observer outputs and corrected nonnegative nested profiler accounting.
- `observer-019-parity-validation.json` retains the earlier observer-on/off and reference comparison. It is supplemental history; the 020 profile-fix validation is the relevant post-fix run.

Build and test records: `build-attempt-002.log`, `build-pins.json`, and `publication-source.patch.gz` (the archived patch bytes are preserved exactly after decompression). The long 310 s drift-observer diagnostic is separate and is not used to claim static equilibrium, physical qualification, or performance improvement.


## Hip-capsule reference sensitivity follow-up (1193–1195)

The nominal 1.0, lower 0.5, and higher 2.0 native hip-capsule scales all completed their 40 s runs, but all three failed the exact final geometry gate. Across arms, 2,490 of 2,500 accepted observer samples were outside the neutral ab/ad flexion fit interval [-12°, 0°]; these are 8-step observer samples, not all physical steps or empirical fit data. No scale was adopted. See the compact follow-up bundle at hip-reference-follow-up-1193-1195/README.md for declarations, executions, metadata, final-step geometry counts, the 1193 region diagnosis, scripts, and hash pins.
