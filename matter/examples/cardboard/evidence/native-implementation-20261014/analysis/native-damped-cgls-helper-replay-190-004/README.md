# Damped CGLS production-kernel replay fixture (source 190)

This revision prepares a source-bound Metal checker for the damped CGLS recurrence in the immutable source-190 snapshot. It compiles and links the checker at Metal O0/O3 and compiles the host harness at both optimization levels. The helper executable is not run by this package; it performs no GPU dispatch.

The fixture calls the production initialize_state, initialize_vectors, measure_initial, form_alpha, update_xr, complete_iteration, form_beta, and update_p kernels. A fixture-only kernel supplies A p and A^T r for a full-rank nonsymmetric 3-by-3 matrix. The host binds one environment with two float4 rows: one xyz mechanical row and one zero field-backing row. Right scaling is disabled. Each run iterates until the production status becomes terminal or the requested cap is reached. The eight requested cases combine caps 32 and 64, operator scales 1 and 2, and damping mu=0 or 1e-3 kg.

The independent standard-library CPU oracle solves (A^T A + mu^2 I)x = A^T b and checks the regularized normal equation. The native harness independently forms b-Ax from the returned x, reads the production recursive residual directly from rBuffer (which already contains b-Ax), computes the vector mismatch ||r_native - (b-Ax)||_2, and gates that mismatch against 512*float_epsilon*(||b||_2 + ||A||_F ||x||_2). It reports the actual terminal status and completed iteration count. Clean terminal statuses iteration_cap and zero_normal_residual require zero error bits. A breakdown is reported as an explicit numerical terminal only when status is breakdown and the sole bit is BadDenominator (32); malformed input, nonfinite state, other error bits, or a still-running state fail.

This is a checker design, not a result: it has not been run natively. It does not qualify runtime integration, the captured FEM operator, Newton correction certificates, nonlinear Armijo behavior, or a full coupon trajectory.

## Revision history

The immutable 002 attempt records native 518's pre-kernel failure: the harness requested unqualified names while the metallib exports numi_matter_metal::-qualified names. Revision 003 fixed those lookups, and native 519 later ran O0/O3. Native 519's candidate comparison was useful, but its reported recursive-residual metric was wrong because it subtracted b from an rBuffer that is already the residual. It also stopped after six iterations even while status remained running. Revision 004 fixes the residual comparison and runs to the requested 32/64 cap. The original 003 package and native 519 output remain unchanged.

## Reproduction and source binding

The checker source and compile commands are in source/. build/validation.json binds the compile-only commands, nine exported functions, 815 frozen source-file hashes, and six source-build artifact hashes. The source snapshot is snapshots/source-damped-full-zero-cgls-190-001, frozen-manifest SHA-256 1189d65c3d7f29d326155eee687979c8dd5fc1c79360a333bd569a08ad30dde7; source-build manifest build-source190-002/source-build.json, SHA-256 54e65bb00fd5050a4613eae5438345efe9ef5297e687ba98f3bfd44eddf2e0bf.

No production project source, runtime file, acceptance setting, or build snapshot is modified by this package. A future native execution must be separately authorized and bound to the exact O3/O0 metallibs and helper executable listed in validation.
