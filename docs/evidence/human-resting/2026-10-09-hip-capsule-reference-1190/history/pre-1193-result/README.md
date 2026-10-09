# Reduced periarticular hip reference: implementation and bounded checks

This increment adds an optional four-term passive periarticular hip reference to the existing Human stand solver. The flag is absent by default; the default-disabled 1192 rebuild was byte-identical to 1191 for the compared CSVs and eight saved MRV packs, and its transaction rejection/replay probe passed. This verifies the disabled path, not the enabled model.

The enabled law uses paired unilateral rotational branches whose slack boundaries shift with flexion. Each branch has the convex C2 toe potential
\[
U(g)=A[g]_+^3/3+B[g]_+^4/4
\]
and contributes restoring force and the corresponding positive-semidefinite tangent to the existing implicit stand operator. The implementation leaves muscle passive terms untouched, adds no extension branch, and does not clamp pose. Source identity includes the option, scale and canonical term descriptor. The absent-option path preserves prior arithmetic.

The nominal toe is inferred from rounded aggregate figure reads in the pinned 2015 cadaveric periarticular study: a 14.5 degree F0 toe and 0.8 Nm/degree tangent at the 5 Nm endpoint. The flexion-dependent slack envelope uses an approximate graph-derived center and width over extension −12 to 0 degrees near neutral ab/adduction. This is an inferred reduced reference for aggregate capsule/labrum restraint, not an isolated capsule measurement, patient-specific calibration or validated human stiffness law. The copied [parameter-and-axis-review-v3.json](parameter-and-axis-review-v3.json) is the pre-implementation parameter review; its historical status field does not describe this implementation. Its limitations remain applicable.

## Validation

- CPU passive-joint tests: 658 checks passed, including potential-gradient, Hessian/tangent, slack, descriptor validation and source-axis constraints.
- Production Metal fixture on Apple M4 Pro: 1,993 checks passed against the actual stand-step and split-mass kernels. It compares active, slack-to-active and active-to-slack states to an independent double reduced-mass oracle at 2, 1 and 0.5 ms; same-20-ms trajectories use 10, 20 and 40 steps and converge monotonically against a 15.625 μs double reference. Energy drift and GPU/oracle energy error are reported; exact discrete energy conservation is not claimed. This is a synthetic two-coordinate fixture, not a full Human run.
- Default-disabled native parity: 1192 completed with return code 0. All compared CSV rows and eight saved MRV packs matched 1191 byte-for-byte, and the transaction rejection/replay check passed. See [the retained comparison](disabled-reference-1192/parity-comparison.json). It does not validate enabled-model behavior.
- Enabled nominal native study 1193 was running when this bundle was assembled; no enabled native result is claimed here. It must be assessed separately when complete.

Earlier fixture attempts are retained under [metal-fixtures](metal-fixtures). Attempt 001 records the unspecialized Metal-function failure; the final harness specializes the production entry points with the required function constants. Attempts 002 and 003 retain the production-step assertion failure logs; the final harness uses the accepted two-step sequencing and sets the split-mass preparation flags. The logs do not support a narrower causal claim. Attempt 004 records an incorrect metallib path and did not execute a kernel. Attempt 005 passed the initial 312 checks but used only two steps per timestep, so those different-duration cases were not a temporal convergence test. Attempt 006 adds the equal-horizon refinement.

The runtime/build artifacts and full native run data remain in their retained external paths rather than being duplicated in Git. [external-artifacts.json](external-artifacts.json) records exact paths and SHA-256 pins; [SHA256SUMS.txt](SHA256SUMS.txt) covers each file in this publication directory.
