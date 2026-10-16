# Damped CGLS production-kernel replay fixture (source 190)

This package prepares a small Metal checker for the changed CGLS recurrence kernels in the immutable source-190 snapshot. It has been compiled and linked at Metal -O0 and -O3, and the host harness has been compiled at both optimization levels with -Wall -Wextra -Werror. The checker executable was **not run**; no GPU dispatch or native-kernel result is claimed.

The Metal translation unit includes the source-190 base_cgls.metalinc and calls its actual initialize_state, initialize_vectors, measure_initial, form_alpha, update_xr, complete_iteration, form_beta, and update_p functions. A fixture-only kernel supplies A p and A^T r for a full-rank nonsymmetric 3-by-3 matrix. The host binds one environment with two float4 rows: a single xyz mechanical row and a zero field-backing row. Right scaling is disabled. Each case starts from zero and performs at most six recurrence iterations with the source kernel's supported cap set to 32.

The four cases use mu = 0 and mu = 1e-3 kg at two operator scales. The base matrix is

    [[ 2.0e-3, -0.75e-3, 0     ],
     [ 0.5e-3,  3.0e-3,  0     ],
     [ 0,       0,       1.25e-3]]

and b = [1.25e-4, -2e-4, 0.75e-4]. An independent standard-library Python oracle solves (A^T A + mu^2 I)x = A^T b, checks the regularized normal-equation gradient and confirms the damping direction is numerically resolved at both scales. It passed all four CPU cases. This CPU result verifies the reference fixture only; it does not exercise Metal.

The host harness reports the recurrence state, iteration count, native candidate, CPU reference, physical residual, recursive-residual consistency, and damped objective. It allows a running state after its six-step diagnostic budget. A native invocation, if separately authorized by the parent GPU owner, would be:

    /Users/n/cardboard-implementation-20261014-host-macmini/analysis/native-damped-cgls-helper-replay-190-002/build/O3/damped_cgls_helper_replay \
      /Users/n/cardboard-implementation-20261014-host-macmini/analysis/native-damped-cgls-helper-replay-190-002/build/O3/damped_cgls_fixture.metallib

That invocation dispatches Metal work and was deliberately not run here. Even a passing native fixture would qualify only these recurrence kernels and this synthetic operator. It would not qualify the source-190 runtime integration, the captured FEM operator, a Newton correction certificate, nonlinear Armijo behavior, or a full coupon trajectory.

## Binding and reproducibility

The source dependency is snapshots/source-damped-full-zero-cgls-190-001, whose frozen-source manifest SHA-256 is 1189d65c3d7f29d326155eee687979c8dd5fc1c79360a333bd569a08ad30dde7. The exact source build is build-source190-002/source-build.json, SHA-256 54e65bb00fd5050a4613eae5438345efe9ef5297e687ba98f3bfd44eddf2e0bf. build/validation.json records the 815 frozen-source file hash checks, all six source-build artifact checks, the seven compile/CPU-oracle return codes, and the no-GPU status.

The standalone Metal compile command is represented by source/compile_and_verify.py; it uses the source snapshot's matter/include, repository include, matter/src, and matter/src/metal include roots with -std=metal4.0 -fno-fast-math. No project source, runtime source, build snapshot, or production acceptance setting was changed by this package.
