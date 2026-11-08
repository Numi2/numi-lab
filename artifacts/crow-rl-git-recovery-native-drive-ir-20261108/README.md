# Native motor-row Git recovery

The sealed plan restores all 176 raw files (35,563,714 bytes) from committed Git objects and executes nine native commands with explicitly relocated libraries and shaders. It requires 34 defined byte-exact file comparisons and 576 exact physical TSV records, excluding only the nondeterministic gpu_ms column while retaining both timing records. The old host-capacity rejection, old zero-gain shader rejection and current crow numerical rejection must reproduce.

This is recovery of existing observations. The original hybrid and split-quality crow trials are byte-verified and restored but not rerun. It does not qualify crow dynamics or the lifelong environment.

Cleanup requires committed and pushed recovery results, actual Git readback, unchanged hash/mode/mtime and zero open handles. The shared Git provider, protected deployment policy and current production build remain in place. The scientific result is recorded in the [native motor-row study](../crow-rl-native-drive-ir-20261108/README.md).
