# Opt-in reuse of the accepted 15-state projection in the FEM tangent cache

This overlay adds `NM_FEM_CACHE_PROJECTION_REUSE=1` for the tightly bounded, single-object static FEM coupon: one active FEM object, one or two environments, exact nonempty topology, and every active tetrahedron using the compensated and scaled generic 15-state material program. It computes the accepted material projection once per tetrahedron and reuses that state while evaluating the nine directional tangent columns. The physical tangent columns are still evaluated and stored by the existing production equations.

The default (`NM_FEM_CACHE_PROJECTION_REUSE` absent or `0`) continues to dispatch `nm_fem_build_algorithmic_tangent_cache` exactly as before. Any other selector value fails initialization. An explicit `1` request fails initialization if any admission predicate is unsupported; it never silently falls back to the old kernel. The original kernel body is byte-identical to the source135 body. The new kernel is a sibling copied from source134 with a fail-closed check that the runtime-admitted 15-state material policy is present.

The overlay does not change the physical operator, residual, acceptance criteria, tolerances, or default behavior. It is an implementation candidate only. It has not been run through the native full-coupon off/on history comparison and is not a performance or physical qualification.

## Source and validation

Base bytes are bound to the retained source135 snapshot (`source-135.json.gz`, uncompressed SHA-256 `e134f53fefb45c0b066b68b4556d1b576f299dade26fcc672fa0f9c6a62a4b55`). The shared tangent-evaluator helper is copied from source134. `verify_overlay.py` checks source and candidate hashes, verifies the original cache builder is unchanged, verifies the sibling against source134 after removing only this overlay's admission guard, and checks the runtime selector wiring.

Validation completed:

- Full Metal translation unit compiled at `-O0` and `-O3`; the `-O3` AIR linked into a metallib.
- Runtime Objective-C++ passed strict `-fsyntax-only -Wall -Wextra -Wpedantic -Werror` against source135 headers.
- The Metal compiler emitted two warnings in unchanged `fgmres.metalinc` about potentially uninitialized threadgroup locals (`sharedValid` and `refinementValid`). They are outside this overlay and did not prevent AIR generation or linking.
- Native off/on history parity and full-coupon qualification remain pending.

See `manifest.json` for exact hashes and commands. `overlay.patch` applies the three candidate files to the source135 base.
