# Superseded attempt 001

The initial SVD run called `scipy.linalg.svd(..., overwrite_a=True)` on the same dense matrix later used for residual products. LAPACK was therefore free to overwrite A, making subsequent `A @ x` candidate residuals invalid. The singular spectrum itself and U/V modal projections were computed from the intended bound matrix, but all candidate residuals in this attempt are superseded and must not be cited. The analysis script was corrected to preserve A and use a separate SVD work copy before the authoritative run.
