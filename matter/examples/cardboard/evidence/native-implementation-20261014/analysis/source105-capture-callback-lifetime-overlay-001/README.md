# Source105 coupled-capture callback lifetime overlay

This diagnostic-only overlay fixes asynchronous manifest reads that crossed the lifetime of the enclosing `encodeFEMCoupledNewtonCapture` lambda. The lambda is declared with reference captures. Its command-buffer completion block previously read `femCoupledCaptureRHSReady`, `state.mixedSolverValue.executionBudgets.w`, `omittedOptional`, and the capture byte cap directly. The first two are reached through state owned outside the block; the latter values also lacked a single explicit ownership boundary.

The candidate now snapshots those values into `detail::CoupledCaptureManifestContext` before creating the completion handler, and explicitly retains the Metal snapshot buffer in `capturedSnapshot`. The callback serializes only copied manifest values and reads the retained buffer. It preserves the RHS flag observed at callback construction and does not force `rhs_preserved_before_fgmres` to true.

The trigger is retained unchanged: the source105 run 369 manifest reports `rhs_preserved_before_fgmres: false`, while source104 run 362 reports `true`; both captures have the same full-RHS payload SHA-256 (`965d38af46af2bbc311cb1534193b8892581cb4d4548936dfc8d7ba9db0cc2e4`). The buffer bytes and production outputs are not rewritten by this overlay. The mismatch is consistent with the callback reading a reference-captured value after the enclosing lambda lifetime, but this candidate has not yet been rerun natively; that causal diagnosis remains to be verified with a fresh capture.

## Changes

- `matter/src/runtime.mm`: materialize the callback manifest values immediately before block creation and replace its four direct reads with the owned snapshot.
- `matter/src/coupled_capture_manifest_context.hpp`: value-owned scalar/vector context shared by production code and a delayed-callback test.
- `tests/test_callback_capture_contract.py`: checks the exact production callback contains no outer-state reads and verifies the snapshot is created before the block without normalizing the RHS flag.
- `tests/delayed_completion_capture.mm`: creates a completion block from the production context type, mutates all source values, then invokes the block asynchronously after the creator returns.

No solver, operator, acceptance, tolerance, rollback, or native physics code path changes. This package contains no GPU run and makes no claim that the previously failed physical fixture passes.

## Validation

- Python callback ownership contract: 3 tests passed.
- Objective-C++ delayed block test: passed; the completion reported the values copied into the context after its source scope returned and source variables changed.
- `runtime.mm` syntax-only compile: passed using source105 compile flags and source105 headers.
- Native GPU verification: pending. Preserve run 369 as the original mismatching observation; do not edit its manifest or bytes.
