# Captured 419 CGLS right-preconditioning study

CPU-only conditioning study bound to captured source125 control 1 / Newton 48. It compares unpreconditioned CGLS, column right-scaling, and 3×3 per-node right-scaling on the same unsymmetrized 2538×2538 operator, without shifting or regularizing A. True residuals are always recomputed in the original unscaled A.

The 3×3 scale is S_n=(A_nnᵀA_nn)^(-1/2), computed from the 3×3 principal block of each free node. This is a diagnostic preconditioner; it has not been used in production. FP64 is a double-precision CPU recurrence. The FP32 variant uses float32 CPU CSR actions and source-shaped CGLS reductions but is not bitwise equivalent to Metal. Iteration 128 is outside the current native CGLS cap and is an offline diagnostic only.

Native residual-replay payloads are included for both precisions at 64/128 iterations for the two right-scaled methods, plus unpreconditioned 64-iteration controls. Each payload preserves the exact NMFDSWP1 header, source/correction bytes, fixed nodes, quiet environment, and w lanes; only the direction block is replaced. Run them only with the root-owned helper command in report.json and fresh output directories.

The same-generation regional/graph RAP preconditioner is unavailable: the 419 rich capture omitted its sidecar under the aggregate capture budget, and the later terminal rejected-conditioning export has no graph patch map or matching selected-generation identity. No regional result is inferred from that later export.

This is linear conditioning evidence only. It does not establish native parity, nonlinear acceptance, a production policy, or coupon convergence. See report.json and manifest.json for complete numbers and hashes.
