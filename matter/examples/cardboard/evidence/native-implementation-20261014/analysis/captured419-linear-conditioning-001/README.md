# Captured 419 linear conditioning audit

This is a CPU-only FP64 analysis of the exact source125 root 1 / Newton 48 rejected-candidate capture in `../captured-base-state-cauchy-419-001`. It does not change source, the physical operator, constitutive response, tolerances, or solver. No GPU/native run was performed.

The captured operator is an unsymmetrized free-DOF CSR matrix `A = M + dt^2 K`, shape 2,538×2,538, with 71,046 nonzeros. The environment and capture bindings, file hashes, timestep, node ordering, and prior validation are in `report.json` and the original package manifest. Inputs are hash-checked before the analysis runs.

## Result

The complete economy FP64 SVD gives `sigma_max = 1.094732845241445`, `sigma_min = 2.3851704931191304e-8`, and `kappa_2(A) = 45,897,467.2`. All 2,538 singular values exceed the conventional double-precision numerical-rank cutoff `eps * max(m,n) * sigma_max = 6.16936e-13`; the matrix is full rank under that threshold, but strongly ill-conditioned.

The weakest right singular mode contains `99.999846%` of the captured Newton correction's L2 norm. The weakest left singular mode contains `2.43286%` of the base RHS norm. For that mode, the captured coefficients are `uᵀb = -4.48725667e-6` and `vᵀdelta = -188.1201223`; multiplying the latter by `sigma_min` gives `-4.48698565e-6`. Since A is nonsymmetric, its weakest left and right vectors are distinct. Across the weakest 16 modes, the RHS projection norm fraction is `2.628%` and the correction projection norm fraction is `99.999989%` (full values are in `report.json`).

The captured correction has `||delta||₂ = 188.1204 m/s`, `||b||₂ = 1.8444380e-4 N·s`, and `||b-A delta||₂ = 2.4927731e-6 N·s`. The FP64 full-SVD least-squares solution has correction norm `188.1318 m/s` and residual `2.1573e-14 N·s`; it differs from the captured correction by `0.03732 m/s` (`1.9841e-4` relative). That indicates this snapshot's huge correction is consistent with solving its captured unregularized linear system; it is not evidence by itself that the production constitutive or nonlinear update is correct.

Diagnostic alternatives show the expected correction/residual tradeoff, not a solver recommendation. For example, Tikhonov `lambda/sigma_max = 1e-8` predicts `||x||₂ = 155.40 m/s`, `||b-Ax||₂ = 7.81e-7 N·s`; truncated SVD at `1e-6*sigma_max` predicts `0.2685 m/s` and `4.49e-6 N·s`; truncating at `1e-4*sigma_max` predicts `0.04665 m/s` and `6.06e-6 N·s`. The exact Euclidean steepest-descent step from zero is only `1.4295e-4 m/s` and leaves `1.6006e-4 N·s`; the separately retained column-equilibrated Cauchy diagnostic is `0.003342 m/s` and leaves `1.2957e-4 N·s`. These `b-Ax` values are linear predictions only.

## Reproduction and retained attempts

The authoritative script computes an SVD from a separate dense workspace while retaining the original CSR matrix for all residual matvecs. It uses `VECLIB_MAXIMUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1` before NumPy import. Full U/V factors are not written; `spectral_compact.npz` stores the singular values and RHS/delta modal coefficients.

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -W error /Users/home/cardboard-implementation-20261014/analysis/captured419-linear-conditioning-001/run_analysis.py
```

Two superseded attempts are retained rather than silently discarded. Attempt 001 let LAPACK overwrite the dense matrix later used for residual calculations; its candidate residuals are invalid, while its singular spectrum/modal data came from the intended matrix. Attempt 002 fixed that but treated Accelerate's spurious FP warnings on dense products as errors before report generation. Neither is authoritative; the final script uses a preserved CSR and bounded vector contractions.

No regularized or truncated candidate was run through nonlinear Armijo or physical acceptance. This package diagnoses one exact captured Newton linearization only; it neither changes nor qualifies the production solver.
