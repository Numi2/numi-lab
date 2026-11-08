# First quality-failure Git recovery

The sealed plan restores all 157 raw files (34,741,071 bytes) from actual Git objects and executes nine native commands plus the independently calibrated FP64 analysis. It requires 41 byte-exact comparisons and all 576 physical TSV records, excluding only nondeterministic GPU timing while retaining both timing outputs.

The original diagnostic admission failure must reproduce. Current capture at one/eight controls must reproduce all public failure files and the complete protected record, followed by isolated native replay and the default 640-transition crow replay. FP64 analysis and its three reference arrays must also match exactly.

This is recovery of existing observations, not crow qualification. Cleanup requires pushed HEAD parity, actual Git readback, unchanged hash/mode/mtime and zero handles. The current production build, protected policy and shared Git dependencies remain in place. See the [scientific record](../crow-rl-quality-first-failure-20261108/README.md).
