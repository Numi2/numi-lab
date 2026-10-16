# Native497 damped least-squares candidates (CPU only)

This package reconstructs the control-20/Newton-63, environment-0 mechanical free-XYZ operator from the captured production tangent cache and creates exploratory regularized search directions. It does not run Metal, alter source, or qualify a solver step.

The operator has 2,538 free XYZ rows and 71,046 nonzeros. Dense binary64 SVD of that reconstructed matrix gives the six smallest singular values from 3.10e-7 to 1.19e-6 and largest 0.276965. A sparse one-norm condition estimate is 3.134e6. These describe this one cached operator only.

Each LSQR solve minimizes norm(A p - (b - A x0)) squared plus lambda squared times norm(p) squared. Saved *_delta.npy files are p and *_total.npy files are x0+p. Lambda 0, 1e-5, and 1e-4 reached the 5,000-iteration limit (status 7); lambda 1e-3 stopped with LSQR status 2 after 798 iterations. Status, residual, and vector hashes are in lsqr-candidates.json. The reconstruction helper action-parity comparison was 1.2397e-4 relative for this captured action; no claim is made that the binary64 assembly reproduces native rounding.

A bounded PROPACK svds attempt with k=6, which=SM, tol=1e-9, maxiter=2000 was interrupted after about five minutes without a result. It is retained as a non-result in validation.json; the dense SVD is the only reported spectrum.

Nonlinear reference evaluation is pending separately. These vectors are not native Armijo trials, and they do not transfer the original Newton certificate. They do not establish a full-coupon or physical result.
