# Frozen-068 angular checker instrumentation overlay

This overlay changes only the checker
`matter/tools/cardboard_free_box_angular_native_check.mm`, copied from frozen
source `source-angular-native-check-and-capture-contract-068`. It does not
change the production contact, rigid-body, or FGMRES kernels.

## Diagnosis of native case 211

The sealed 211 run is still a failure and remains unchanged. It exited before
the contact-action and inertia-action suites because the checker threw on the
first rotating sweep fixture. The failure says the validator returned a
contact failure, its alpha-path flag matched the expected clear result, and
the check `diagnostics.y > diagnostics.z` disagreed with expected physical
collision. That comparison is not a valid physical-sweep classification:
`diagnostics.y` is the minimum gap among visited advancement samples, while
`diagnostics.z` is the collision floor. A positive sampled minimum does not
prove the continuous sweep clear when conservative advancement can reject on
its step floor or iteration bound.

For the first fixture, a separate FP64 sample of the authored box-point path
gives alpha-path minimum gap `+0.065 m` and physical-sweep minimum gap
`-0.005 m` at fraction `0.5` (the mid-path crossing). This agrees with a
fail-closed physical-sweep rejection. The frozen-211 stdout did not emit the
raw status fields because the old checker threw before its print statement;
the overlay now prints code, object/index, all diagnostic components, and the
FP64 sampled minima/fractions for every path fixture before summarizing them.
These are sampled FP64 geometry diagnostics, not a proof of continuous
clearance and not a replacement for production CCD.

The path checks now accumulate mismatches instead of throwing immediately.
They run all five path fixtures, then continue into the production contact
action and free-rigid inertia action suites. A real path mismatch still makes
the final process exit with code 2. An individual path-kernel command error
is printed and does not suppress the remaining path cases.

## Binding and validation

`source-overlay.json` binds the checker copy and the 211 prediction, receipt,
stdout, and stderr to their source hashes. The source-side hashes match the
input map sealed by the 211 prediction. The overlay passed Objective-C++
`-fsyntax-only` with `-Wall -Wextra -Wpedantic -Werror`. No Metal build or GPU
execution was performed. Root should build and run this instrumented checker
as a new evidence case; do not overwrite or reinterpret case 211.
