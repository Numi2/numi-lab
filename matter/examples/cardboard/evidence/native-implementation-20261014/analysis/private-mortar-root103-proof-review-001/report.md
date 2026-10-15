# Source 103 whole-parent mortar skip review

**Verdict:** no actionable unsafe false exclusion found in the integrated source 103 whole-parent skip. This was a read-only CPU/source audit; it is not a native Metal or physical qualification.

## Bound source and test

Snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-whole-mortar-proof-103`.

The frozen-source manifest binds the reviewed contact and FEM Metal files and the shared guard. The source review also checked active-patch admission and the production cohesive-force paths. The standalone CPU checker was built from this snapshot with:

```sh
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -I snapshots/source-whole-mortar-proof-103/matter/include \
  snapshots/source-whole-mortar-proof-103/matter/tools/mortar_exact_tiling_check.cpp \
  -o analysis/private-mortar-root103-proof-review-001/mortar_exact_tiling_check
analysis/private-mortar-root103-proof-review-001/mortar_exact_tiling_check
```

Result: `mortar_exact_tiling_checks=61 exact_fan_and_tiles=true exact_feature_identity=true whole_parent_sweep_certificate=true patch_identity=true axis_normal_relative=true` (exit 0). Binary SHA-256 is recorded in `manifest.json`.

## Why the shortcut is bounded

`activeMixedCohesivePatchesForPrimitivePair` only collects matching parent-face references whose face is active, paired, mixed-mode, non-dormant, and bilinear; malformed incidence and more than four candidates reject. Thus an inactive/dormant or unsupported face cannot contribute to a successful coverage certificate. The exact-cover gate then requires each side's partition to consist entirely of covered cells and prove a non-overlapping exact Q25 tiling of the parent.

After that, the production early return requires one of two sweep certificates: (1) exact 3D identity over the complete linear parent sweep and same-face patch-cell paths, or (2) horizontal, nondegenerate parent triangles with exact common translation per side, coincident XY footprints at both endpoints, and same-face projected patch-cell paths with coherent shared vertices. Since the checked trajectories are linear, these conditions keep the complete footprint and each patch correspondence invariant over the interval. Lateral slip, shear, rotation, nonplanar geometry, partial coverage, cross-wired face IDs, unresolved arithmetic, and excess patch count do not meet the guard and remain on ordinary CCD or fail closed.

The shortcut suppresses only ordinary fallback contact for a fully covered parent pair. Cohesive force/history kernels still process each activated face. Their mixed-mode branch assigns normal compression traction as `stiffness.z * trialGap.x` when the local normal gap is nonpositive, so covered normal closure remains owned by the cohesive interface law. This review verified control flow and equations, not numerical closure behavior on GPU.

## Scope limits

The axis-normal proof allows common XY translation of each parent (the two projected paths must coincide), plus independent normal translation. It is not a general deforming, rotating, or sliding-mortar proof. Full coverage is limited to 2–4 active patches. No GPU/native trial or full box qualification is asserted here.

## Reviewed locations

- `matter/src/metal/contact.metalinc`: active face admission (1589–1642), exact-cover predicate (2099–2114), same-ID sweep construction (2190–2230), skip and unchanged CCD fallback (2284–2344), production call paths (2594 onward, 2735 onward, 3859 onward).
- `matter/include/numi/matter/mortar_shared_edge_guard.h`: exact common translation/permutation (592–657), axis-normal whole-parent proof (689–731), projected patch identity/coherence (784–867).
- `matter/src/metal/fem.metalinc`: active/dormant face checks and mixed cohesive compression branch (1788 onward; 1344–1365 for local traction).
