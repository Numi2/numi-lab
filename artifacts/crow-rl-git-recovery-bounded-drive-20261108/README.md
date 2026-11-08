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
