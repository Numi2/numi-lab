# Newton 9 linear completion versus no-op gate (capture 385)

This report uses the existing source-bound 385 center capture and independent same-generation residual replay. It does not change solver policy or qualify the full-contact run.

The captured action is a useful approximate linearized descent: `||b|| = 3.7630833378917694e-6`, `||J delta|| = 3.7548819182694667e-6`, and independent `||b - J delta|| = 8.176561191043932e-8`, a 97.8272% reduction. Native Arnoldi estimate is `8.176974120033265e-8`, differing by `4.12928989332969e-12`. The scheduled forcing coefficient is 0.015625, so its target is `5.87981771545589e-8`; both estimates exceed it by about 39.1%.

The finalizer has a distinct outer-target admission: `max(regularization.x, residualTolerances.x * max(state.nonlinear.x, 1))`. With floor `1e-8`, tolerance `1e-7`, and first Newton norm `0.019994786009192467`, this is `1e-7`. The default 385 path has no true-residual-replacement selector, so this admission used the Arnoldi estimate, not the offline replay. The replay independently confirms the resulting error is below that absolute outer target. The solver's nonlinear-converged latch remains false, and Armijo is still run.

The no-op is separately gated on base and trial norms both being no more than `residualTolerances.x = 1e-7`, plus first trial index, zero alpha, and healthy statuses. Newton 9 base is `3.7630829865520354e-6`, 37.63 times that bound, so the no-op condition is correctly false.

At Newton 9, 22 earlier trials fail before measurement because the deformable CCD path does not certify the sweep. The only two status-clear trials use alphas `2.384185791015625e-7` and `1.1920928955078125e-7`; both produce a residual one FP32 ULP above base (`2.2737367544323206e-13`) and unchanged by halving. The accepted-step result is a failed nonlinear line search, not proof of a residual noise floor or a successful physical transaction.

Source paths and exact line ranges are bound in `report.json`: `fgmres.metalinc` finalizer at 5811–5818, true-residual replacement gate at 5328–5331, and `fem.metalinc` certified no-op gate at 4490–4495.
