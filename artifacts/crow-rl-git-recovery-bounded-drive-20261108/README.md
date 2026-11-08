# Git recovery and bounded native replay

The manifest in `plan.json` retains every raw build, calibration, runtime and
crow trial file for the bounded-drive integration slice, including failed
instruments and rejected physical observations. Recovery reads the actual Git
objects, restores all manifest files into a temporary directory and runs nine
scoped native commands with relocated libraries and explicitly selected shaders.

The sealed plan fixes hashes, memory-on-disk budget, expected exits and a stop
rule before execution. Exact physical, numerical and protected-failure byte
comparisons are reported in `verification.json`. This establishes scoped recovery,
not scientific qualification or a successful learning environment.

Every restored or newly produced temporary file is inventoried. Original bytes
must resolve to the committed archive; new bytes are compressed into this Git
directory. Cleanup requires pushed HEAD parity, actual Git readback, unchanged
hash/mode/mtime inventories and zero open handles. The raw source archive is
pruned only after the restored execution and temporary cleanup succeed.

The experimental motor mode is not promoted. The final crow trial fails on tilt
and penetration; the default-mode regression remains byte-exact. See the
[physical study](../crow-rl-bounded-drive-crow-20261108/README.md) and
[requested optimizer checks](../crow-rl-post-step-rate-20261108/README.md).

## Executed recovery and cleanup

All 9 native commands passed their declared exits and all
50 byte comparisons were exact. The known bad shader
was rejected at its first invalid batch. All 396 original
files (89518819 bytes) were read from actual committed Git objects.
This includes reproducing the physical tilt failure, not qualifying it.

After pushed HEAD parity, actual Git verification, exact inventories and zero
open handles, 448 temporary recovery files
(90640420 logical bytes) and 396 original
raw files (89518819 logical bytes) were removed. Receipts:
`temporary-cleanup.json` and the constitutive study's `cleanup-receipt.json`.

A separate `local-scratch/manifest.json` preserves all 755
local authoring drafts, copied sources and derivative caches
(27555022 bytes). Every unique content was decoded from actual Git
objects before deleting 1497 source/export copies
(33208362 logical bytes). See
`local-scratch-verification.json` and `local-scratch-cleanup.json`. Draft
preservation makes no additional scientific or native-execution claim.

Two archival checks correctly blocked cleanup before any source deletion:
a raw backing script had been edited after capture, and a local transfer was
staged before it finished. Original backing bytes were restored, the changed
instrument received a separate filename, and the completed transfer was
recommitted. The repair and incomplete-stage receipts preserve both failures.

After Git verification and pushed HEAD parity, 960
duplicate archive checkout files (31039583 logical
bytes) were omitted. The Git objects remain reachable. Post-omission actual Git
readback again verified all 396 original files. Current library
and shader hashes match the tested runtime; no owned native worker remains.
Mac mini Data-volume free space is approximately 149.8 GiB. See
`omission.json` and `final-state.json`. The full research goal remains open.
