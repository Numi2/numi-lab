# Source-bound diagnostic environment overlay 161-002

This overlay is a complete patch from `source-right-grip-telemetry-157`. It
retains the source161-001 environment-contract work, registers the two
source160 selector-5 consumer-capture controls (49 native environment keys
total), and adds a separate diagnostic-run path. It does not change the native
solver or its acceptance criteria.

Diagnostic captures can be requested only with `--diagnostic-run` and an
explicit `--diagnostic-capture KEY`. The runner creates an exclusive,
case-local output prefix; arbitrary output paths are rejected. It inventories
only regular, single-link files under that prefix, rejects symlinks and
changed files, and enforces 64 files and 64 MiB per file, with a 128 MiB
aggregate runner limit. The source160 native payload itself has a separate
64 MiB cap. Diagnostic predictions and environment records have distinct
diagnostic schemas. The cached-operator audit accepts either prediction
schema, but for diagnostic predictions it requires and validates the
diagnostic environment record, source registry, capture inventory, terminal
receipt, and exact captured-file hash. Convergence and resolution admission
continue to reject diagnostic records.

The source160 read-only review found its consumer readback disabled when both
capture variables are absent. With a valid selector it is limited to one
zero-rate, single-microtick event and selector 5; its readback is placed after
the graph-apply barrier and before the operator consumes `z`. It copies to a
bounded shared snapshot and writes exclusive `.consumer.bin`/`.consumer.json`
files from the command-buffer completion handler. It does not alter candidate
or accepted physical buffers. The size-only preflight for source464 computes
39,825,484 bytes (37.98 MiB) for one selected environment, including both
18,885,348-byte graph matrix arrays. That exceeds the old 32 MiB cap by
6,271,052 bytes and fits the source160 64 MiB cap with 27,283,380 bytes of
payload headroom. This is a source-bound sizing estimate, not a native
allocation or peak-memory measurement. A native consumer capture has not been
run.

Validation used Python 3.10.12. The diagnostic/environment/convergence and
cached-operator suites pass 63 tests; the cached-operator suite also passes
13 tests independently. The source-bound harness suite passes 7 tests. The
patch-apply check applies the emitted patch to exact source157 files and
compares every output byte. These checks include synthetic diagnostic-run
records; they are not a Matter native run. No native execution, Metal build,
or GPU execution was performed for this overlay.

The output package is `source-overlay.patch` plus the exact changed source
files under `source/`. Apply it to the bound source157 snapshot with
`patch -p1 < source-overlay.patch`. See `source-overlay.json` for file hashes,
dependency bindings, and test artifact names.
