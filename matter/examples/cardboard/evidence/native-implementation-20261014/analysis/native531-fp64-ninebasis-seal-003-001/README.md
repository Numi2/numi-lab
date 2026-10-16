# Native531 nine-basis FP64 finite-difference audit seal

This seal binds, but does not modify, analysis/native531-fp64-ninebasis-003. Exact package and output hashes are in seal.json. A verifier checked every file declared by the original package manifest, input identity, center strict gates, bitwise center/tangent parity, and all-row outcome counts.

The audit compared nine local deformation-gradient basis directions at all 1,728 paper elements, with 15 symmetric perturbation sizes from 2^-10 through 2^-24 and fixed accepted history. All 17,280 center tangent vectors match the parent FP64 reference C exactly; all center states are valid and match their parent state bitwise. All endpoints are strict-valid and branch-sign classifications remain the same across the grid. 15,529 of 15,552 tet-direction pairs have adjacent same-branch centered differences at relative L2 <= 1e-6. The remaining 23 are unresolved; none is classified as invalid, branch-crossing, or zero-action.

Tet 222 is a substantive exception, especially basis columns 2 and 6: best centered relative differences are about 0.3113 and 0.3059 at the smallest declared step. Those are not a noise-floor pass. The independent reference works directly with Green strain and PK1=F*S, without an eigenbasis/polar decomposition. High-precision checks must focus on the smoothed Hill/Fischer-Burmeister residual and the implicit 15-state Jacobian/tangent chain, including effective smoothing variables and one-sided slopes.

Observed caveats: the successful run emitted a NumPy eager-where divide warning at the rationalized FB residual line for an inactive arm. The resulting tested outputs were finite and all projections valid, but the warning was not captured in a dedicated ledger and no universal harmlessness claim is made. Three failed preflight attempts were reported in execution history but their exact logs were not retained; this seal labels them as history only.

This is a CPU-only local constitutive self-consistency test on one captured state. It does not establish native FP32 noise, a full assembled tangent, a converged coupon, calibration, or physical qualification.
