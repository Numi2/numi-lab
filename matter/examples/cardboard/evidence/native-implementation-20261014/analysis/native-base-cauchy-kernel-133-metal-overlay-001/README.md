# Source 125 base-Cauchy Metal kernel overlay

This is the kernel-only source 133 candidate for an opt-in raw Cauchy fallback after ordinary Newton Armijo exhaustion. It preserves the primary Newton FGMRES direction and its certificate; the fallback is a new direction whose admission is through the existing nonlinear Armijo and final checks. Runtime wiring is intentionally absent from this package and is owned separately. No GPU kernel run is claimed.

The patch modifies only `matter/src/metal/fem.metalinc` and `matter/src/metal/fgmres.metalinc`. It adds cached tangent transpose/forward operator modes 3/4 scoped by the selected-environment marker, captures the full `2*E*N` assembled RHS while masking constrained/nonpositive-mass mechanical rows, and rejects any nonzero or nonfinite field-backing RHS before building a mechanical-only direction. The selected free mechanical RHS is also checked for finite values in the builder so a preparation kernel cannot sanitize NaN/Inf to zero. The direction is `p = tau A^T b`, `tau = ||A^T b||^2 / ||A A^T b||^2`. The builder records original FGMRES convergence, clears its latch for this distinct direction, resets line-search/trial state, and writes a zero diagnostic trace for unselected environments.

The CPU test is an independent small-matrix reference for mechanical masking, full field-half preservation/rejection, finite free mechanical RHS, fixed rows, descent, and zero/singular action rejection. It is not a substitute for the parent-owned native Metal checker.

Base: frozen source 125 at `/Users/home/cardboard-implementation-20261014/snapshots/source-status5-vtccd-diagnostics-125`. `overlay.patch` was applied to copies of the exact base files and the resulting SHA-256 digests matched the overlay bytes. The package contains no runtime changes.

Validation: 5 CPU reference tests pass. The complete `matter/src/matter_kernels.metal` translation unit compiles at Metal 4.0 with `-O0 -fno-fast-math` and `-O3 -fno-fast-math`; only pre-existing `sharedValid` and `refinementValid` warnings appear in `fgmres.metalinc`. Commands, logs, and AIR hashes are in `validation/`. No GPU execution or physical/full-solver qualification is claimed.

To reproduce the Metal compile, set `SOURCE_ROOT` to a source-125 checkout with `overlay.patch` applied, then run `validation/compile_metal.sh`.
