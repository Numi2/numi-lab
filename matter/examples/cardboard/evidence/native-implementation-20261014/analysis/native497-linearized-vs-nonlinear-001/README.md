# Native497 captured linear action versus CPU nonlinear residual (CPU only)

This report joins the exact CGLS v3 scratch capture to the Newton63 operator-FD node capture. The source-defined v3 payload keeps total candidate X=x0+S*y, seeded fresh residual B, and post-CGLS residual R. It reconstructs A*X from the captured production A*x0 plus B-R, then compares the linearized norm ||R0-alpha*A*X|| with the strict CPU material/force residual curve. It does not run native trial kernels or certify a native trial.

Input and source hashes are in report.json. The companion nonlinear replay and three small-alpha direction reports remain separate immutable packages. Strict local material projections passed all 1,728 paper tetrahedra for all reported points; this remains a single captured Newton state, not a physical or whole-coupon qualification.
