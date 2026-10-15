# Angular contact sample diagnostics overlay (source 069)

This checker-only overlay records raw production contact status/sample fields and continues through the other contact/inertia fixtures after an unusable contact sample. It does not change Metal code, contact thresholds, or acceptance logic. The test remains a failure if any required production sample is absent or invalid; it reports that sample as failed and marks its operator-action part skipped. Other action/inertia fixtures still run where their independent sample is valid.

## Basis

- Frozen source: `snapshots/source-graph-tail-guard-angular-instrument-069`.
- Failed native attempt: `native/free-box-angular-production-kernels-213` (preserved without modification). All five rotating path fixtures reported PASS. The run then exited at `face_pos did not produce a valid production contact sample`, before contact-action and inertia cases; its stdout did not include the sample fields needed to identify whether the failure came from status, valid flag, or branch marker.
- Exact production kernel used by the checker: `numi_matter_metal::nm_contact_evaluate` from the same frozen 069 metallib.

The overlay adds `contact_setup` records containing status code/object/failing index/diagnostics; sample flags, point, gap, normal, velocity, barrier and angular-branch markers; and the source node/rigid pose/mass values. Invalid samples do not enter derivative calculations. The checker still returns failure, so instrumentation cannot turn an invalid production sample into a pass.

## Validation

CPU Objective-C++ `-fsyntax-only` passed with `-Wall -Wextra -Wpedantic -Werror` against frozen 069 headers. No Metal build or GPU execution was performed. The new native run is required to identify the actual invalid-sample layer and to obtain the contact/inertia action results.

Rebuild using the frozen 069 CMake source and normal target `numi-matter-cardboard-free-box-angular-native-check`, replacing only its checker source with this overlay. Bind the resulting executable and metallib before running; do not reuse case 213's old binary as evidence for this checker.
