# Retained crack-plane face-ambiguity gate (source115 overlay 002)

This isolated overlay adds a fail-closed check for retained crack-band planes that classify an entire tetrahedron face as on-plane. The geometry routine now computes one `TetCrackPlaneClassification`; polygon construction and the opt-in full-face policy consume that same normal, centroid plane point, signed-distance array, and relative tolerance. The retained matrix builder opts into `RejectWithinClassificationTolerance` for each of its four selected planes. Other geometry callers keep the default `Allow` behavior.

The gate is strict: for each of the tetrahedron's four faces, at least one face vertex must have `abs(signedDistance) > distanceTolerance`. Equality remains in the on-plane category because the existing classifier treats `[-tolerance,+tolerance]` as side zero. Nonfinite distances fail closed. This is a numerical ambiguity check; it does not reject a plane merely because it intersects a face along an edge or line.

The CPU fixture demonstrates the gap left by the existing volume floor. With an axis-aligned tet of x/y side 1 m and z height `2e-12 m`, the centroid plane has a base-face distance `5e-13 m`, below the existing `1.4142135623730952e-12 m` classifier tolerance. The old default geometry path returns that complete triangular face as a positive-area cut, and the volume is still `4.1465` times the existing degeneracy floor. The strict retained policy rejects it. A boundary case at exact tolerance is rejected; one representable height step outside is accepted. A separate 1 mm-side, `2e-15 m`-height case exercises rejection through the actual retained `buildElementProgram` call.

The valid ordinary and rotated-plane geometry controls compare the patched default polygon, normal, area, volume, and characteristic length bit-for-bit against a namespace-renamed copy of the original source115 geometry header. Those controls pass. The full retained-matrix CPU checker also passes, including its source-material, generated-program, tangent, and work checks. That checker was compiled from the overlay source and linked to the prebuilt compiler archive from snapshot 091; this is not a fresh full-project build. The isolated boundary smoke test was compiled and run directly against the overlay header with `-Wall -Wextra -Wpedantic -Werror`.

The earlier report-only memory audit 001 is preserved unchanged (manifest SHA-256 `ae644c2b55dee8539e51705b906503f6d6389845fe5f1aba8413b63c5ecbdf08`). This patch adds no mesh-wide `FaceRecord` or other ownership workspace: classification uses fixed-size per-plane values. The existing generic ownership-check vector remains absent from the live retained shipping path. This does not show that the full shipping mesh passes the new gate; no full-mesh traversal, shipping build, Metal run, or physical qualification was performed.

## Files and integration

`overlay.patch` is a modification-only patch against source HEAD `4fb44c44511609d66eb4443fd8aa3730a39d516b`. It changes only:

- `matter/tools/cardboard_fracture_regularization.hpp`
- `matter/tools/cardboard_retained_matrix_damage.hpp`
- `matter/tools/cardboard_retained_matrix_damage_check.cpp`

The existing CMake target `numi-matter-cardboard-retained-matrix-damage-check` already includes the changed checker; no CMake edit is required. Apply from the repository root with `git apply overlay.patch`, then rebuild and run that CPU target against the intended source snapshot. `git apply --check` passed against the base tree; the shared checkout itself was not modified.

## Verification records

- `tests/face_boundary_smoke.cpp`: PASS. It checks bitwise baseline geometry equivalence for ordinary and rotated cuts, full-face classification of a volume-floor-valid sliver, strict equality behavior, and one-ULP-outside admission.
- `tests/retained_matrix_damage_check.stdout`: PASS (`cardboard_retained_matrix_damage_cpu_passed`). Includes builder-level sliver rejection. stderr is empty.
- Overlay checker syntax-only compile: PASS with `-Wall -Wextra -Wpedantic -Werror`.
- `git apply --check`: PASS.
- GPU/runtime initialization/full shipping mesh: NOT RUN.

Exact file hashes and command records are in `manifest.json`, `report.json`, and `verification-commands.txt`.
