# Executed Git recovery of adjacent FP32 refinement

All 215 source files (25,620,799 bytes) were recovered and verified from actual
Git objects. The restored native binaries executed 20 commands; three numerical
analyses also executed. All 80 comparison records and 624 physical TSV records
match the original observations (GPU timing excluded). The old and new crow
rejections are retained and reproduce exactly. Recovery establishes these
existing observations only; it does not qualify the complete crow environment.

The first final handle check observed two open handles and prevented closure.
After that process exited, a separately sealed finalizer verified all 348
temporary files and their actual Git contents, exact hash/mode/mtime, the old
process absence and zero handles. Both handle observations are retained.
No cleanup occurred before the successful recheck. The restored/generated
workspace (27,713,309 bytes) and original raw inputs were then removed using
their exact committed inventories.

The source manifest is ../crow-rl-quality-representability-20261108/payload/manifest.json.
Plans, execution, comparisons, verification, temporary cleanup, local cleanup,
archive omission and final verification describe the bounded recovery.
Native runtimes and production sources remain identified in final verification.
No physical tolerance, training profile or policy was promoted.
