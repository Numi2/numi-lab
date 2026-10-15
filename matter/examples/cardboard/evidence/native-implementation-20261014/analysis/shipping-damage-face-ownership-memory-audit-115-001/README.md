# Shipping crack-band face-ownership memory audit

This is a source-bound accounting adjudication for the source115 tree at `4fb44c44511609d66eb4443fd8aa3730a39d516b`. It makes no source, runtime, or memory-limit change.

The new `validateNoDuplicateSharedFaceCrackBandArea()` workspace is not allocated on the shipping retained-matrix damage path. `cardboard_probe.mm` builds that path through `retained_damage::buildMeshProgramPool(uniqueRequests)`; the retained builder in `cardboard_retained_matrix_damage.hpp` does not call the generic face-ownership validator. Repository-wide call tracing shows the validator is used by the generic `damage::buildMeshDamageProgramPool` and `buildMeshDamagePrograms` only. Therefore, the source115 `FaceRecord` vector contributes **zero bytes** to the active shipping prebuild/runtime arena estimate. It must not be added to GPU-resident Metal bytes.

The hypothetical generic-builder workspace is still quantified: each request contributes four records, and the host ABI layout of `{uint64_t, size_t, size_t, uint8_t}` is 32 bytes on this 64-bit host. For 1,477,728 requests, `reserve(4*R)` is 5,910,912 records or 189,149,184 bytes (189.15 MB / 180.39 MiB), before allocator metadata. The vector is local to the validator, filled and sorted in place, and destroyed on return. Its live range overlaps the request array and host mesh preprocessing, but not the subsequent element-program loop, `compileWorld`, or `Runtime::initialize`. The four plane-geometry objects and their polygon vectors are local to the currently processed request, so they add only request-local temporary storage, not another mesh-wide array.

The retained builder independently computes four selected-plane sections per element. At the call site each plane has `offsetFromCentroidM == 0.0`, and the geometry function rejects degenerate tetrahedra. In exact geometry, a plane through a nondegenerate tetrahedron's strictly interior centroid cannot coincide with a boundary face; the shared-face full-area duplicate caught by the generic guard is therefore excluded by this construction. The geometry routine uses a relative classification tolerance of `1e-12`, however; this audit did not measure a minimum altitude for every shipping tetrahedron. It does not claim that tolerance cannot misclassify an extreme sliver.

The prebuild gate is run before mesh construction. The later exact source/compiled-world budget is computed after the overlay builder has returned. Neither is a measured host peak. The custom retained builder's request/map/program work is separate from the `FaceRecord` workspace and remains represented only by the existing planning reserves. This audit does not certify total host preprocessing peak or revise any existing admission limit.

If a future shipping implementation calls the generic builder, account `4*R*sizeof(FaceRecord)` as a checked **host preprocessing transient**, with `R` equal to the actual generic builder request count or a conservative prebuild bound. For a phase-aware peak model it belongs with mesh/request preprocessing, not in Metal resident bytes; use the maximum of overlapping phase inventories rather than claiming the temporary persists into runtime.

## Reproduction

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/shipping-damage-face-ownership-memory-audit-115-001/audit.py \
  --root /Users/home/numi-cardboard-20261006 \
  --requests 1477728 \
  --output /Users/home/cardboard-implementation-20261014/analysis/shipping-damage-face-ownership-memory-audit-115-001/report.json
```

The checker reads sources and computes the ABI-sized record count; it does not compile, initialize Metal, allocate the vector, build the probe, or measure memory. All source-contract assertions passed. See `report.json` for exact file hashes and line bindings.
