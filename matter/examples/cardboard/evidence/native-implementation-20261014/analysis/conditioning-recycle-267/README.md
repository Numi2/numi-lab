# Accepted-correction coarse-mode overlay

This is an isolated, opt-in solver-conditioning experiment against snapshot
`source-runtime-diagnostic-return-repair-079`. Apply
`accepted-correction-recycle-079.patch` from that snapshot root. The exact base
and overlay file hashes are in `manifest.json`.

Set `NM_FEM_REGIONAL_ACCEPTED_CORRECTION_RECYCLE=1` only with preconditioner
selector 5. Any other value, or use with another selector, fails during runtime
admission. With the variable unset, the existing source mode count and solver
path remain unchanged. The opt-in bit is included in the execution fingerprint.

The host adds one independent static `y^4` bending trial field after the
existing authored modes. Before the first Newton update, that field remains in
the reserved slot. Once `nm_fem_apply_solution` has applied the chosen
line-search factor, a scratch buffer stores its already-scaled FEM xyz
correction. At the next Newton linearization, the selector first builds a
per-environment, two-pass modified-Gram-Schmidt basis from the candidate-masked
source modes. Dynamic grips zero rows after the host-authored basis was
orthonormalized, so those effective rows may no longer be orthonormal. The
orthogonal workspace is separate; the P used for AP/RAP is unchanged. Exactly
dependent rows are skipped. A finite but nonzero row below the relative squared
rank threshold `1e-10` is treated as rank-ambiguous and selects the static
fallback, avoiding admission of a direction in an omitted near-null subspace.
The recycle correction is projected against the normalized independent rows
and checked for relative orthogonality to every original effective row before
it can replace the fallback. Zero, nonfinite, dependent, or rank-ambiguous
input keeps the fallback. The current candidate still rebuilds AP/RAP through
the ordinary production path on every Newton iteration. No physical state,
tangent, residual, line search rule, tolerance, or FGMRES restart length
changes.

The optional regional diagnostic payload includes
`accepted_correction_recycle_mode` (`uint32[environment]`): `0` means the
static fallback was selected and `1` means the stored accepted FEM correction
was selected. This records selection only; it does not demonstrate that the
conditioning change improves convergence.

The opt-in reserves a checked `E*N` correction buffer and a checked `E*baseModeCount*N` orthogonalization workspace, both accounted in resident bytes. The host regional-scratch admission includes the latter; all products are checked before allocation.

CPU checks:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v test_recycle_basis.py
/Users/home/.pyenv/versions/3.10.12/bin/python3 check_recycle_basis.py --output run-006/cpu-recycle-basis-check.json
c++ -std=c++20 -Wall -Wextra -Werror -Ioverlay/matter/include regional_bending_fallback_check.cpp -o run-006/regional-bending-fallback-check
run-006/regional-bending-fallback-check
```

The source-bound CPU check uses the snapshot-079 N63 operator capture as a
retrospective vector-algebra control. Its correction is not a captured
preceding-Newton accepted line-search increment. It shows only that this one
same-Newton control is independent of the existing source modes after
projection. A synthetic CPU fixture exercises dynamic masking that makes
source rows nonorthogonal, dependent and near-rank-ambiguous rows, FP64 QR
agreement, and disjoint environment-major scratch ranges. These are algebra
and indexing controls, not solver-performance evidence.

The overlay passed Objective-C++ syntax checking and full Matter Metal AIR and
metallib compilation against source-083 headers in an isolated external compile
tree. The Metal compile emitted the same two existing uninitialized-threadgroup
warnings as source 083 without the overlay. No GPU execution has been performed;
a native trial should compare the unmodified selector-5 trajectory against
this opt-in using the same source, inputs, and full accepted/rejected history
evidence.

`regional_recycle_masked_basis_native_check.mm` is a separate kernel-level
checker source for a later native run. It calls the production candidate-mask
projection and recycle-selection kernels with two environments: one mask makes
the retained modes nonorthogonal but independent, and one makes them dependent
and must preserve the static fallback. It checks the raw selected vector and
that authored P, candidate nodes, status, and retained effective P rows are
bitwise unchanged. Its candidate-bound SPD action is synthetic; the checker
does not evaluate or qualify the production FEM operator A. The source has
passed Objective-C++ syntax checking only, and has not been built into a target
or executed on a GPU.
