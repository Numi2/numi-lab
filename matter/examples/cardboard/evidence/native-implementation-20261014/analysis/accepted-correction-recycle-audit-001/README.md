# Accepted-correction recycle capture: coarse inverse audit

This CPU-only audit extends the selected source-085 capture at root 0, Newton 1. It rebuilds the nonsymmetric coarse matrix from the independently assembled `A = M + dt^2 K` action and the captured effective basis, then applies the captured FP32 LU factors and row permutation to every unit right-hand side. It retains the previous RAP reconstruction and adds per-column true-operator residuals for scalar FP32 substitution (each multiply, subtract, and divide rounded) and FP64 `math.fsum` substitution using the same captured factors. This models the source solve order, but is not a native Metal solve replay.

The true RAP singular values are `3.1825384e-5` through `8.3718237e-2`, with condition estimate `2630.55`. The native LU reconstructs pivoted true RAP with relative Frobenius error `1.72037e-7` and maximum entry error `1.71637e-8`; the minimum absolute U diagonal is `3.81947e-4`. Across all 27 unit RHS columns, the maximum true-RAP residual 2-norm is `7.04427e-5` for scalar FP32 substitution and `3.20118e-5` for FP64 substitution. Since each RHS has unit 2-norm, these are also the maximum relative residuals. The largest solution error versus a direct solve of the independently rebuilt true RAP is `9.96258e-6` (FP32) and `9.43617e-6` (FP64). The worst residual is column 2. Every unit-RHS residual is below `1e-3`, the existing prepared local-solve diagnostic scale; this comparison is not a physical or trajectory acceptance criterion.

Adding captured recycle mode 26 changes the 2-norm condition estimate from `2476.58` for the 26-mode base RAP to `2630.55` (+`6.217%`). Its Schur complement is `0.00691896`, about `48.82%` of its diagonal `0.014172`. The mode is normalized, orthogonal to the other captured effective modes to `1.38e-8` maximum dot product, and its independently rebuilt `p^T A p` is `0.0141720`. These measurements do not show a singular or grossly inaccurate coarse inverse in this captured linearization; they do not establish that the changed preconditioner improves Krylov convergence for the actual residual.

The full native run 284 still failed its strict certificate at step 0 (`1.77112e-4` residual versus `1e-6`; zero accepted steps). This single Newton-1 linearization cannot identify why the opt-in trajectory fails or establish that recycle improves a full solve. Captured tangent/operator consistency is not an independent validation of the material tangent.

## Reproduction

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 -W error \
  /Users/home/cardboard-implementation-20261014/analysis/accepted-correction-recycle-audit-001/audit_recycle_capture.py
```

The script fails closed on source/capture/graph identity mismatches. `audit.json` contains the per-mode AP comparison, full true RAP matrix, per-column inverse results, source bindings, run receipt, and limits. `manifest.json` binds the script, report, selected native case, capture bytes, compiled package, and static source files by SHA-256.
