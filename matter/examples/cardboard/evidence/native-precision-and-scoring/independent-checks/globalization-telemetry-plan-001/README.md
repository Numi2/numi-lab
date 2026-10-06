# Bounded FEM Newton trace patch

This is a scratch-only runtime patch and Python reader for one selected `request.controlStep`. It was prepared against the exact frozen source snapshot:

- `matter/src/runtime.mm` SHA-256: `a1c1bae6f0c6a61a703969a0994026f6fa46d7ec579bccf51d07d6b1107abda3`
- Runtime patch: `runtime-newton-trace.patch`
- Runtime patch SHA-256: `5b663cf3c5dcfea87392cd280eb5b2755e129c62141acc5b4646645a6c0dbb82`
- Decoder SHA-256: `ed30b1c17429c67ddd35bb0a0cc019a966f0284132be7059c659a1a139486a1d`
- Synthetic test SHA-256: `f408fce46832f17761a12d84f7d4ff3da15b6de9acb572beefdacb55245e9f11`

The runtime trace is opt-in through `NM_FEM_NEWTON_TRACE_ROOT=<control-step>`. It records four line-search boundaries for microtick zero: before contact limits, after deformable limits, after rigid limits, and after the final line-search limit. Each record copies the environment alpha, per-object volume line-search tuple, FGMRES diagnostics/state, and raw status. A separate snapshot copies solver certificates and status immediately after `nm_mixed_certify`, before candidate masks or commit.

The trace is fail-closed to static FEM-only generalized unknowns: exactly one explicit FEM object, one physics substep at index zero, maximum rate exponent zero, at most four environments and 4,096 FEM nodes, no MPM or mixed fields, adaptive/identification/mutation/learned modes, generalized rigid unknowns, vascular or human-support unknowns. The FGMRES arena must equal `2 * environmentCount * femNodeCount`, accounting for both runtime vector blocks. Rigid contact rows can still affect FEM alpha and residuals. Unsupported scope, invalid root selectors, iteration limits, allocation failures, and snapshot caps produce bounded skip/error records. The trace copies at most 128 stage snapshots plus one terminal snapshot; each is capped at 16 KiB (about 2.02 MiB maximum copy payload). It adds blits to the caller's existing command buffer and does not submit or wait on a queue.

The source establishes that `NMFGMRESStateGPU.nonlinear.y` is the full assembled generalized residual norm at the start of that Newton solve. Thus iteration *i*'s selected alpha aligns with iteration *i+1*'s `nonlinear.y`, which is the next reassembled residual. The final iteration uses the post-certificate relative residual rescaled by `max(fgmres.diagnostics.y, 1)`. The trace does **not** record an independently recomputed linear `b-Ax`, nor exact Krylov work for each Newton iteration. `diagnostics.x/y` is a final-restart Arnoldi estimate divided by the Newton-start norm; `diagnostics.z` means the cycle was accepted, not that the full nonlinear residual converged; `diagnostics.w` counts columns in the final restart. `status.fgmresIterations` is the microstep maximum `totalUsed`, not a per-Newton count.

The terminal certificate capture occurs before later candidate masks and commit. Its raw `NMMatterStatusGPU.code` is the status authority at that point: a nonzero code means the certificate-stage candidate failed, regardless of `certificate.validity.w`. A zero code means only that no failure was latched at that capture; it is not proof of eventual commit or accepted-state publication.

The decoder and tests are in `scratch/tools/cardboard_newton_trace.py` and `scratch/tools/test_cardboard_newton_trace.py`. The synthetic tests cover iteration alignment, nonconverged and zero-column restart-tail state, small-norm stage drift, malformed/truncated logs, missing/misaligned terminal records, explicit skips, one-object/layout admission, and a terminal failure whose certificate validity field is set. The CLI adds SHA-256 bindings for the input log and decoder and refuses to overwrite an existing report. All twelve tests pass with Python. The proposed CTest registration is `cmake-newton-trace-test.patch`.

No C++ build, runtime test, or GPU run was performed. The source patch is ready for parent review/application; its C++/Metal integration remains unverified.
