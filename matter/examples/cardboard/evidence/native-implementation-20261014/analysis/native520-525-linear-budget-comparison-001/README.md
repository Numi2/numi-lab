# Native 520 / 525 linear-budget diagnosis

This CPU-only report binds the source195 Newton9 capture and the later source195 extended-budget alias fixture. It recomputes the 520 residual directly from the captured float32 RHS and production matrix-free J-delta arrays, checks the capture payload and array hashes, and verifies the 525 audit artifact.

The 520 capture is a root2/control-step2/Newton9 state with FGMRES budget 64, restart 10, Newton budget 16, and line-search budget 8. It did not request CGLS damping. The independently formed residual is nearly unchanged from the RHS. The saved post-FGMRES residual workspace agrees closely with b-J-delta, so the discrepancy is not a mismatch between that action and workspace.

The 525 fixture uses FGMRES 256, restart 16, line search 24, Newton 32, and unchanged residual tolerance 1e-7. It accepts two controls, then rejects control 3 before a private trial because pretrial status is 10. Its exact baseline/alias parity and rollback are audited separately in native525-independent-byte-audit-001.

The 520 capture and 525 fixture share source, build artifacts, and material asset, but they are not a same-state paired replay. These results diagnose the short-budget Newton9 stagnation and show that a larger restart/budget advances the bounded fixture further; they do not establish full coupon convergence or universal contact/material tangent correctness.
