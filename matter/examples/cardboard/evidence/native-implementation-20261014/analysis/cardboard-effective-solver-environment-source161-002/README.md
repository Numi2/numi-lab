# Source-bound diagnostic environment overlay 161-002

This is a complete patch from `source-right-grip-telemetry-157`. It retains
the source161-001 environment-contract work, registers the two source160
selector-5 consumer-capture controls (49 native environment keys total), and
adds a diagnostic-run path. It does not change the native solver or its
acceptance criteria.

Diagnostic captures require `--diagnostic-run` and an explicit
`--diagnostic-capture KEY`. The runner creates an exclusive case-local output
prefix; arbitrary output paths are rejected. It inventories regular,
single-link files under that prefix, rejects symlinks and changed files, and
enforces 64 files and 64 MiB per file, with a 128 MiB aggregate runner limit.
The source160 native payload has its own 64 MiB cap. Diagnostic predictions
and environment records use distinct schemas. The cached-operator audit
accepts both schemas, but for a diagnostic prediction it requires the
diagnostic environment record, exact source registry, capture inventory,
terminal receipt, and an exact match between the retained capture and the
operator-FD capture path in the environment contract. Convergence and
resolution admission continue to reject diagnostic records.

The source160 read-only review found consumer readback disabled when both
capture variables are absent. With a valid selector it is limited to one
zero-rate, single-microtick event and selector 5; readback is after the
graph-apply barrier and before the operator consumes `z`. It copies to a
bounded shared snapshot and writes exclusive `.consumer.bin` and
`.consumer.json` files from the command-buffer completion handler. It does
not alter candidate or accepted physical buffers. The source464 size-only
preflight computes 39,825,484 bytes (37.98 MiB) for one selected environment,
including both 18,885,348-byte graph matrix arrays. This exceeds the old
32 MiB cap by 6,271,052 bytes and fits source160's 64 MiB cap with 27,283,380
bytes of payload headroom. This is a source-bound sizing estimate, not a
native allocation or peak-memory measurement. A native consumer capture has
not been run.

Validation used Python 3.10.12. The diagnostic/environment/convergence and
cached-operator suites previously passed 63 tests, and the source-bound
harness suite passed 7 tests. After the final exact-capture-path binding
change, the focused cached-operator suite passed 13/13 and the touched Python
modules compiled. The synthetic audit test also rejects a diagnostic record
whose capture path differs from the exact environment-contract path. The
earlier 63-test and 7-test receipts are retained as
their original runs; they were not rerun after that narrow binding change.
The patch-apply check applies the emitted patch to exact source157 files and
compares every output byte. These checks include synthetic diagnostic-run
records; they are not a Matter native run. No native execution, Metal build,
or GPU execution was performed for this overlay.

The first announced seal is preserved byte-for-byte under
`historical-first-seal-001/`. The current `source-overlay.json` supersedes it
with exact operator-FD capture-path binding and its corrected synthetic
fixture. Apply the current `source-overlay.patch` to the bound source157
snapshot with `patch -p1 < source-overlay.patch`. `source-overlay.json` is the
canonical manifest and records source hashes, dependencies, test receipts,
and the preserved first-seal hashes.
