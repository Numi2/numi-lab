# Source125 run 419 FP64 directional sweep with tightened diagnostic projections

This CPU-only diagnostic binds the exact source125 run419 Newton48 rich capture, fixed accepted history, the 422 production-term replay, the 424 native residual-sweep report, and immutable baseline 005. It evaluates ideal unquantized FP64 endpoints F0 +/- epsilon*dt*H from the captured FP32 high+low center for epsilon 2^-14 through 2^-30.

Each endpoint first runs the unchanged 005 projector, then continues the same residual equations and complex-step Jacobian with residual-descending Newton steps to diagnostic-only gates of 2e-12 scaled residual and 2e-11 dynamic correction. These are not production tolerances. Failed rows remain unavailable; no endpoint is borrowed from 005.

The rowwise audit requires two adjacent fixed-grid points with both endpoint projections passing, the recorded lambda/yield/FB predicates matching center, and local centered secant error <=1e-6. It separately reports the 1% assembled A-action gate. The two scopes are distinct. The compressed NPZ records per-tet errors, projection gates, force/stress scale, and Jacobian condition.

This is a local constitutive diagnostic, not a native tangent qualification, universal noise floor, coupon convergence, calibration, or physical qualification. See report.json and manifest.json for source hashes, warnings, row failures, and limitations.
