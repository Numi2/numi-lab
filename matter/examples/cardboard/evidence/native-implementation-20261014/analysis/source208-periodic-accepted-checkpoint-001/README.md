# Source208 periodic accepted-coupon checkpoint overlay

This isolated source overlay is based on the frozen source205 tree. It adds
--coupon-checkpoint-every N as a default-off, observer-only mode. For 60
scheduled controls and cadence 8, the capture indices are 7, 15, 23, 31, 39,
47, 55, and 59. For a 64-control schedule the final index is 63. The final
accepted boundary is always included; cadence at or beyond the schedule is
rejected because it would provide no intermediate checkpoint. The mode is
separate from the existing one-shot save-and-stop and resume options.

Each write is gated on the normal accepted-control path after accepted state,
solver certificate, and tooling audit. It flushes observation streams, sizes
the actual serialized accepted snapshot with the same serialization code,
bounds an individual checkpoint at 2 GiB, bounds the number of snapshots at 8,
and bounds all checkpoint-folder artifacts at 8 GiB. Periodic prefix schema v2
records accepted byte prefixes, so later appended bytes do not invalidate the
captured prefix. The existing v1 one-shot prefix format and stop behavior
remain unchanged. A capture failure is logged and disables further observer
writes while the physical schedule continues; the final result status is
computed by the original success/failure path.

A terminal checkpoint is an archive of the final accepted state. Its metadata
uses next_control_index equal to the schedule length and next_phase=completed;
the existing resume loader rejects cursors at or beyond the schedule length.
Thus the final snapshot is not presented as a resumable continuation.

CPU validation built the probe and both focused check targets with the
source-bound builder, then passed matter.compiler.cardboard_coupon_resume and
matter.compiler.cardboard_shipping_snapshot. The latter checks that predicted
serialized snapshot size exactly matches the emitted file. Probe help output
exposes the new flag. Build/test attempts 001–005, including an initial compile
failure and cadence expectation failures, are retained by source-build receipt
or test log in attempt-history.json. Build006 is the final successful
source-bound build and test.

No native GPU or full coupon run has been performed. This package does not
establish observer on/off parity, restoration of an intermediate checkpoint
after later output appends, or any physical/convergence qualification. Source210
integration still needs its source-bound build, observer parity run, and a
resume-checkpoint test for an intermediate (nonterminal) saved control.
