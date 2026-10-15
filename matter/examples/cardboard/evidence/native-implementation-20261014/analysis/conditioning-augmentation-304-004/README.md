# Same-Newton extra coarse-mode test for capture 304

All inputs are bound to source091 capture 304, the same root-2/Newton-63/environment-0 rejected transaction. The comparison uses the exact captured free-free A, RHS, effective P and production source prediction/receipt. It does not use an accepted state or a cross-Newton recycle.

`run.py`/`report-v2.json` test the proposed extra mode `q = x - Q(Q^T x)`, where Q is a rank-checked double-precision QR basis of the exact captured 26-column effective P, and `P_aug=[P,q/||q||]`. `AP_aug=A P_aug` and `R_aug=P_aug^T A P_aug` are rebuilt from the same captured A. RAP factors and patch matrices are cast to FP32 for the approximate production-shaped offline Schwarz replay; Krylov and true residual calculations are CPU double and are not native Metal parity.

Results:

- The a-posteriori captured-native correction q gives 27 independent Euclidean basis columns, but RAP has condition about 1.17e7, smallest singular value 6.16e-9, a negative symmetric-part minimum about -6.16e-9, and only 26/27 singular values above a 1e-7 relative FP32 rank threshold. Its FP32 coarse-only correction from zero worsens relative residual to 6.21; after 16/32/64 restarted steps the residual is 0.2304/0.0693/0.000441. This q comes from the end of the same failed solve and is unavailable at its start.
- An online-feasible q generated from the first 16-step same-Newton restart has similar RAP conditioning (1.18e7; 26/27 at the FP32 rank threshold). After applying the FP32 RAP update at that restart, relative residual rises to 5.97, then the next 16 and 32 steps reduce it to 0.241 and 0.000560. This does not reach the 1e-6 solve gate and uses a near-singular coarse factor that should fail closed absent native conditioning/rollback evidence.
- A direct-solution-derived q is only a retrospective upper control. It has similarly ill-conditioned RAP; FP32 coarse-only residual is 0.00358 and restarted 16/32/64 residuals are 1.68e-5/5.30e-6/2.02e-8. Its source vector is not available online.
- The exact P-only offline restart baseline already plateaus near 0.9522 by 64 steps. That earlier no-augmentation result is preserved at `../conditioning-candidates-304-003/report.json`.

No added mode is recommended for native integration from this evidence. The captured correction is a near-dependent direction after applying A and eliminating the original coarse space; it creates a RAP pivot around 6e-9 and an FP32 rank defect. Scaling/replacing the requested normalized mode or using a different Petrov-Galerkin left basis would be a different candidate requiring separate evidence. No physical operator, tolerance, material or accepted history was changed. The 304 step remains rejected.

The original `report.json` and `run-v1.py` are preserved as v1. V1 mislabeled the coarse-only update as R16 in the from-zero sequence; use `report-v2.json`, whose R16/R32/R64 fields are true restarted-iteration counts. V1 is retained for traceability, not as the final comparison.

`report-v2.json` SHA-256: `4f7ab62630b93daaf338661baaafb9b9c65e5d624ac0bb98dfdb7c80599eb7a2`.
`run.py` SHA-256: `020d943b36289ad09bac0cdb7bab56e868b9d6d13ce113cfb6451db1da5082c4`.
