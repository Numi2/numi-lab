# First quality-failure Git recovery

The sealed plan restores all 157 raw files (34,741,071 bytes) from actual Git objects and executes nine native commands plus the independently calibrated FP64 analysis. It requires 41 byte-exact comparisons and all 576 physical TSV records, excluding only nondeterministic GPU timing while retaining both timing outputs.

The original diagnostic admission failure must reproduce. Current capture at one/eight controls must reproduce all public failure files and the complete protected record, followed by isolated native replay and the default 640-transition crow replay. FP64 analysis and its three reference arrays must also match exactly.

This is recovery of existing observations, not crow qualification. Cleanup requires pushed HEAD parity, actual Git readback, unchanged hash/mode/mtime and zero handles. The current production build, protected policy and shared Git dependencies remain in place. See the [scientific record](../crow-rl-quality-first-failure-20261108/README.md).

Executed recovery passed: 157 files / 34,741,071 bytes restored, nine native commands plus one FP64 analysis, 41 byte-exact comparisons and 576 exact physical TSV records excluding GPU timing. The original admission rejection, all captured public failures and isolated protected record reproduced. FP64 report and all three reference arrays match exactly.

After pushed HEAD parity, actual Git readback and zero handles, cleanup removed 201 temporary recovery files (35,581,494 logical bytes), all 157 raw files (34,741,071 logical bytes), and 20 local draft/inventory files (1,136,438 logical bytes). Original payload adds 7,753,809 compressed bytes, reusing identical Git content. Three small recovery blobs retain changed timing/runtime output.

Final verification re-reads original and recovered contents from actual Git objects after duplicate archive checkout copies are omitted. See verification.json, temporary-cleanup.json, local-cleanup.json and the scientific record's cleanup-receipt.json.

Final verification passed after omitting 72 duplicate archive checkout files (7,939,493 logical bytes). It re-read all 157 original files (86 unique contents) and 201 recovery files (89 unique contents) from actual Git objects. All six modified/current production source files equal Git HEAD. The current native library, shader and motor probe match the sealed tested identities. Protected deployment97 is unchanged, all owned raw/recovery scopes are absent, and no owned native workers or open build handles remain. Data-volume free space is 149.75 GiB. See final-verification.json. The full crow goal remains active and incomplete.
