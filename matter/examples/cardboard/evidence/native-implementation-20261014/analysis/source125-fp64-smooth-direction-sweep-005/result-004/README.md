# Source125 run 419 ideal FP64 directional epsilon sweep

This CPU-only diagnostic binds the exact source125 run419 Newton48 rich capture, fixed accepted history, the 422 production-term replay, and the 424 native residual-sweep report as provenance. It evaluates ideal unquantized FP64 endpoints F0 +/- epsilon*dt*H from the captured FP32 high+low center, for epsilon 2^-14 through 2^-30. It reports plus, minus, and centered local secants.

Every independent constitutive projection must pass the existing reference residual and dynamic state-correction gates. Failed rows are retained, and a global assembled action is unavailable unless all required paper rows pass. The assembled action keeps M*dv and the signed -dt^2*K contribution separate. Branch summaries cover endpoint lambda-increment sign, yield-residual sign, and reference FB arithmetic path; they do not reconstruct every internal return-map Newton branch.

All endpoints are ideal FP64 inputs; no captured FP32 side endpoints are substituted. The 424 result is bound as separate provenance only. This sweep is a local diagnostic, not a native tangent qualification, universal noise floor, coupon convergence, calibration, or physical qualification. See report.json and manifest.json for source hashes, projection outcomes, and limitations.
