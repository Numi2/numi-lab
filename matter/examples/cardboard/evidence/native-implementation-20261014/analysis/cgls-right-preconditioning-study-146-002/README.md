# Captured 419 CGLS A_ii right-preconditioning study

CPU-only conditioning study bound to captured source125 control 1 / Newton 48. It compares unpreconditioned CGLS with `S[j,j]=1/A[j,j]` and per-node nonsymmetric `S_n=(A_nn)^−1` right scaling on the same unsymmetrized 2538×2538 operator. No operator term is shifted or regularized; solve coordinates are `x=S*y`, and the transpose recurrence applies `Sᵀ`. True residuals are recomputed in the original unscaled A.

The preceding 146-001 package preserves global column-norm and `(A_nnᵀA_nn)^−1/2` controls. This package focuses the cheaper A_ii-based scales requested for native feasibility. FP64 is a double-precision CPU recurrence. The FP32 variant uses float32 CPU CSR actions and source-shaped CGLS reductions but is not bitwise equivalent to Metal. Iteration 128 is outside the current native CGLS cap and is an offline diagnostic only.

Native residual-replay payloads include the diagonal and 3×3 A_ii inverses in FP64 and emulated FP32 at 64/128 iterations, plus unpreconditioned 64-iteration controls. Each payload preserves the exact NMFDSWP1 header, source/correction bytes, fixed nodes, quiet environment, and w lanes; only the direction block is replaced. Use the root-owned helper command in report.json and fresh output directories.

The same-generation regional/graph RAP preconditioner is unavailable: the 419 rich capture omitted its sidecar under the aggregate capture budget, and the later terminal rejected-conditioning export has no graph patch map or matching selected-generation identity. No regional result is inferred from that later export.

**Erratum for companion 146-001:** its prose labeled `K` as `-d(f_internal)/d(v)`. The captured assembled convention is `A=M+dt²K` with `K=-d(f_internal)/d(x)`; the 146-001 CSR calculations and direction outputs were unchanged by that label error. This package uses the corrected convention.

This is linear conditioning evidence only. It does not establish native parity, nonlinear acceptance, a production policy, or coupon convergence. See report.json and manifest.json for complete numbers and hashes.
