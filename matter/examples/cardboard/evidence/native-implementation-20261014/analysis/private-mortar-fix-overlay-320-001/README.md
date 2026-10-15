# Exact Q25 mortar tiling and paired-seam witness fix

This is an isolated patch against source snapshot `source-private-mortar-witness-098` (parent revision `35bfb518c542a306f563cc364ef8496837068b9a`). It addresses FP32 clipping slivers for exact, small mortar tilings and limits seam-witness suppression to a provably identical material-point pair. It does not change accepted histories, force laws, contact tolerances, or physical material parameters.

## Change

`mortar_shared_edge_guard.h` adds a conservative exact Q25 predicate for barycentric XY maps. It accepts only coordinates exactly representable on the `2^25` integer grid; it never rounds coordinates to make a proof pass. For 2–4 triangles it requires positive nonzero orientation, exact reversed cancellation of internal directed edges, exact contiguous coverage of all three parent boundaries, and pairwise separating-axis proof of no positive-area overlap. Unsupported, malformed, or non-Q25 maps decline the fast path and remain subject to the existing clipping checks. There is no floating-point area-sum shortcut.

`contact.metalinc` uses the proof to preserve the authored covered triangles directly rather than generating a numerically clipped complement with tiny residual cells. The seam CCD exemption is pointwise: it applies only when an exact paired boundary/vertex maps to the same 3D material point on both parents and those two points are bitwise coincident at the uploaded sweep start. It does not skip the whole pair of triangles or adjacent cells; all other vertex/triangle and edge/edge witnesses continue through CCD.

For that single fixed material-point pair, linear endpoint interpolation gives

`r(t) = x_left(t) - x_right(t) = (1-t) r(0) + t r(1)`.

The guard establishes `r(0) = 0`, so `r(t) = t r(1)`. Thus the same two material points cannot separate and later re-collide during this linear sweep: if `r(1) != 0`, their distance grows monotonically from zero; if `r(1) == 0`, they coincide throughout. This argument does **not** authorize suppressing a near-paired point, an initially offset pair, another barycentric point on either triangle, a rotating/nonlinear path, or any unresolved map. Those remain CCD candidates or fail closed. A finite-thickness seam is intentionally treated as the same bonded material-point pair, not as a second ordinary self-contact pair; other points on the surfaces are still tested.

## Captured-input evidence

The accompanying CPU decoder replays the exact uploaded face records from native witness package 319:

```sh
python3 audit_mortar_witness_maps.py \
  /Users/home/cardboard-implementation-20261014/analysis/private-mortar-witness-319-001/cases.jsonl \
  /tmp/native-witness-map-audit.json
```

It requires all four selected records and exits nonzero if any side fails. Both sides of the selected three-patch fan and four-tile cases satisfy the exact Q25 edge-cancellation / parent-boundary coverage proof and pairwise no-positive-overlap check. The captured maps are interpreted in the contact partition's XY coordinates, with `z` reconstructed as `1-x-y`. The source FP32 `z` for thirds is `0.3333333432674408`, whereas XY reconstruction is `0.333333283662796`, a residual of `5.960464477539063e-8`; the report preserves this normalization discrepancy rather than calling the raw 3D weights identical.

Those 319 cases had 12/16 expected outcomes pass before this overlay. Four expected-positive seam/closure cases were rejected. The map audit proves the relevant 2D maps are exact covers; it does not prove the new native Metal path passes those cases. No GPU run has been performed for this overlay, and the 319 outcomes remain the pre-patch evidence.

## Validation performed

CPU exact-geometry check:

```sh
clang++ -std=c++20 -Wall -Wextra -Werror -I matter/include \
  matter/tools/mortar_exact_tiling_check.cpp -o /tmp/mortar_exact_tiling_check
/tmp/mortar_exact_tiling_check
# mortar_exact_tiling_checks=11 exact_fan_and_tiles=true exact_feature_identity=true
```

The same 11 checks passed with AddressSanitizer and UndefinedBehaviorSanitizer. The full-contact Objective-C++ target compiled and Metal shader compilation succeeded in the isolated build tree:

```sh
cmake --build build-private-mortar-320 \
  --target numi-matter-cardboard-full-contact-armijo-check -j2
```

The patch applies cleanly (`git apply --check`) to the exact 098 snapshot. CPU checks and compilation are software evidence only. The patch still needs root integration and native regression execution; it does not qualify shipping assembly or physical corrugated-board behavior.

## Files

- `private-mortar-exact-tiling.patch`: modification-only patch.
- `native-witness-map-audit.json`: decoded source-319 input maps and exact-cover results.
- `audit_mortar_witness_maps.py`: reproducible decoder/auditor for that captured input format.
- `build-source/`: isolated patched source tree used for compilation.
- `manifest.json`: file hashes and validation bindings.
