# Bounded multi-patch mortar union overlay

This is an isolated source patch against the frozen source-086 snapshot. It adds native same-parent-pair union/complement partitioning for up to three active mixed-mode cohesive patches. The source patch is `mortar-union.patch`; apply it only to the exact base listed in `manifest.json`.

The contact builder collects the active mixed-mode faces for the exact canonical primitive pair, orders them by stable face identifier, maps each face patch into the parent triangle's barycentric coordinates, and rejects invalid mappings or positive-area patch overlaps. It sequentially subtracts each patch from the uncovered cells, leaving each covered cell labeled with its authored face index. A left/right covered-cell pair is excluded from ordinary contact only when both cells have the same patch identity. Covered cells from different patches remain contact candidates, including after sliding. The same partition builder is used by the ordinary triangle-pair contact path and the private swept-CCD validator. The point-feature suppression path likewise requires both points to be covered by the same exact face.

Each authored mixed-mode face continues to be evaluated and stored independently by the existing cohesive force/history path; this patch does not merge cohesive state. The compiler rejects a fourth mixed-mode face on the same canonical parent primitive pair when self-contact is enabled. Runtime partition capacity is 64 triangles per side; reaching that bound, invalid maps, or overlapping active patches fail closed. The area-overlap threshold is `1e-10` in parent-barycentric area. Clip and edge tolerances remain numerical implementation thresholds, not manufacturing tolerances.

For three patches, one stored partition cell is 32 bytes (three `float2` barycentric vertices and two `uint`s); each 64-cell partition is about 2,056 bytes including its count/alignment. The caller holds two partitions, while partition construction also needs a temporary partition, mapped patch coordinates, and clipping polygons. That is approximately 6.5 KiB of explicit per-thread local storage at peak before unrelated contact locals and compiler register/spill decisions. This is a source-derived estimate, not a measured Metal stack/register or occupancy figure.

The shipping source generator creates one starch paired face per manifold paper/glue tetrahedral face pair; for that default source topology, the parent-pair multiplicity is one. Optional HMA closure patches are a separate path and their multiplicity was not measured here. The three-patch bound therefore does not qualify arbitrary HMA closure meshes or other geometries with four or more patches on one parent pair.

## Checks completed

- `python3 -m unittest -v test_mortar_union_partition.py`: 14 CPU geometry/identity tests passed. These test the reference implementation, not the Metal kernel.
- Strict C++ and Objective-C++ syntax checks passed for the modified compiler and native checker.
- Metal 4.0 compilation of `matter_kernels.metal` passed. It emitted two existing warnings in `fgmres.metalinc` (`sharedValid` and `refinementValid`), outside this overlay.
- The CMake target `numi-matter-cardboard-full-contact-armijo-check` built from a source-086 copy carrying this overlay.
- Its `--cpu-only` admission run passed, reporting 36 FEM nodes, 48 tetrahedra, 72 contact pairs, and zero native FEM transactions. It exercised the compiler controls for three distinct pairs, three disjoint patches on one parent pair, fourth-patch rejection, overlap reaching runtime fail-closed validation, self-contact-disabled admission, and paired-interface/mortar admission.

No GPU/native contact transaction was run for this overlay. In particular, the private self-sweep controls and `--same-parent-mortar-union-parity` remain for the parent task's native run. No physical cardboard result or qualification is claimed.

## Native commands for the integrating task

```sh
numi-matter-cardboard-full-contact-armijo-check BULK.nmatter --private-self-sweep
numi-matter-cardboard-full-contact-armijo-check BULK.nmatter --same-parent-mortar-union-parity --diagnostic-state-output FRESH_OUTPUT_DIR
```

The union parity mode owns its solver/material settings; it compares the legacy and tiled arms on the same three-patch full-contact fixture, including activation, closure, loaded rejection, and rollback, and checks that all three stable face records remain separate. The contact checker is synthetic and does not calibrate a real corrugated grade.
