# Native497 small-alpha curves and linear action (CPU only)

Three separate CPU nonlinear curves use the same accepted control-20/Newton-63 base: warm CGLS total X, independent exact-LU total direction, and captured original FGMRES correction. Alpha is 0, 1/1024, 1/512, 1/256, 1/128. The alternatives are not added together.

For warm CGLS, captured v3 semantics are X=x0+S*y, B=b-A*x0, and R is post-CGLS residual, so A*(X-x0)=B-R and A*X=A*x0+(B-R). The linear residual prediction is ||R0-alpha*A*X|| on selected free mechanical rows, using captured production A*x0 and FP32 B/R and host binary64 norms. No linear action is claimed for exact-LU or FGMRES.

All 15 CPU strict-reference trial points pass material projection with zero failed tets. This is not native trial execution, whole-coupon qualification, or physical cardboard qualification. Inputs and sources are SHA-bound.
