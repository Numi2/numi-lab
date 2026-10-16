# Native 469 guarded runner v4

This is a runner-only repair package. It leaves the sealed v3 runner and all v3 receipts unchanged. It refers to the immutable source173 checker/build and the v1 input package; it does not compile or execute the Metal checker during its CPU tests.

The v4 fixes the two execution blockers found in v3: each invocation allocates a unique `runs/invocations/<UTC>-<UUID>` directory, so CPU-admission logs, native outputs, and receipts are never reused; and the native worker lease is acquired once and held through completion, without probing the same lock while owned. Worker processes are checked both before and after the lease is acquired. The child checker writes stdout/stderr directly to exclusive per-run files, avoiding pipe back-pressure. RSS, free-space, timeout, and log-size bounds remain explicit.

Default invocation performs hash verification plus CPU input admission only. Native dispatch requires `--execute`. A preflight `launch_allowed` result is a point-in-time observation, not a reservation; the execute path reacquires the lease and checks active workers immediately before dispatch. A refusal gets a per-run refusal record and does not start the native checker.

The input/source/binary hashes and runner command remain the v3-bound values. `INPUTS` and `EXPECTED` were compared against the v3 script: all eight input bindings and all eight original source/build/checker bindings match exactly.

## Review files

- `validation/run_guarded_source469_v4.py`: runner and report schema writers.
- `tests/test_guarded_source469_v4.py`: CPU-only namespace, lock, and file-backed-output tests.
- `validation/report-schemas.json`: field lists for v4 invocation, preflight, refusal, CPU admission, and native execution records.
- `validation/v4-tests.command.json` and `validation/v4-tests.stdout.log`: exact test command and receipt.
- `source-overlay.json`: parent v3 manifest binding, v4 source hashes, and limits.
- `manifest.json`: exact package payload inventory. The manifest excludes itself and any runtime `runs/` outputs.

No native/GPU dispatch was run in preparing this package. Six CPU tests pass.
