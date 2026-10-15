# Free-box angular FP64 inertia-chain checker overlay (073)

Overlay 073 is based on the corrected, instrumented checker 072 (`base-908255.mm`, SHA-256 `908255e8aabbca96f1c1ca535de6390ae6f6ddb948f4e66fb87178d89b1f7bd1`). It carries forward the 072 swept-path classification repair and adds a distinct FP64 verification of the free rigid-body angular inertia chain. It changes only `matter/tools/cardboard_free_box_angular_native_check.mm`; the production solver and its tolerances are unchanged.

The inertia oracle uses the actual fixture `MRBodyStateGPU` fields at execution. It records the exact FP32 words for source quaternion, initial angular velocity, dt, world inverse-inertia rows, candidate angular increment, and each direction, then promotes those words to double. It evaluates `q = Exp(dt*(omega0 + DeltaOmega))*q0`, rotates the captured world inverse inertia, forms `R = -I_world(q)*DeltaOmega`, and computes `A = -dR` with the spatial left Jacobian. Each direction has an independent FP64 central-difference refinement sequence. Gates are `1e-6` for the smooth FP64 derivative and `1e-3` for production Metal action versus the FP64 chain, both with vector-relative and component-over-vector-norm checks.

The existing native-FP32 candidate finite-difference results remain a separate quantization-sensitive diagnostic. Their unresolved status remains inconclusive; the independent smooth FP64 result does not convert them into a pass. The swept path output also distinguishes production rejection/uncertified from a clear certificate and continues through contact and inertia diagnostics.

Validation performed: Objective-C++ `-fsyntax-only` passed against the source-071 headers with `-Wall -Wextra -Wpedantic -Werror`. No native/GPU run was performed for this overlay.

Files: `cardboard_free_box_angular_native_check.mm` is the replacement; `checker.patch` is the diff from `base-908255.mm`; `source-overlay.json` binds hashes, equations, gates, and validation; `base-908255.mm` is the exact checker base.
