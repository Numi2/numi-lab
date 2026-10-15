# Bounded same-parent mortar union overlay (source 091)

This package implements a narrow shared-boundary repair for the private mortar-complement CCD path, raises the explicit same-canonical-parent-pair patch ceiling from three to four, and adds a bounded 4-patch admission/CCD fixture. The patch is against frozen source snapshot `source-rationalized-paper-and-native-checkers-091` and is in `mortar-union-four-patch-092.patch`.

The repair retains each covered cell's face/patch identity. It does not skip a covered-cell pair wholesale. For a pair of distinct mortar face IDs, it derives the candidate common edge from each authored face's mapped coordinates on both parent primitives. It filters a single vertex/triangle or edge/edge CCD witness only when both witness barycentric points lie on that common edge, their edge parameters match, and relative normal displacement is non-approaching within the source epsilon. Inward normal motion, a mismatched edge parameter, off-edge points, and interior witnesses remain contact candidates. Both production narrowphase and private Armijo sweep callers use the same code path.

The CPU predicate check is `cpu_boundary_contract_check.py`; its output is `cpu_boundary_contract_check.json`. It uses the exact float32 sector and midpoint maps authored by the fixture, with native clip/overlap/edge tolerances recorded. It verifies shared-edge endpoints/interior, separating/tangential/inward motion, parameter mismatch, off-edge, positive-area cyclic cross-mapping interior, and 4-triangle tiling. It is CPU-only and does not execute Metal CCD.

The C++ source/compiler check and shader build pass. The CPU-only executable reports source admission success with zero native FEM transactions. Native private-sweep and full-contact GPU tests have not been run by this worker. Native run 301 remains preserved and only reports an aggregate closure error; it did not capture an exact first pair/witness, so the pair attribution remains a CPU reconstruction. Do not treat compile/CPU tests as a native or physical qualification.

## Current shipping geometry ceiling

`shipping_hma_parent_pair_multiplicity.json` is a separate source-bound CPU reconstruction of the current default shipping source. It uses the actual default design (300×200×200 mm inside, 25 mm manufacturer joint, 1 mm slot kerf, 16 subdivisions per band, literature2009 recipe), 8 corrugation intervals per pitch and one through-thickness slice, then performs starch node duplication and the source HMA closure builder. Starch and HMA faces are grouped together by the same canonical parent primitive IDs used by the compiler.

The resulting mesh has 528,444 nodes, 1,477,728 tets, 97,984 starch paired faces, and 162,851 HMA mortar patches across 9 HMA groups. Combined records group into 183,033 canonical primitive pairs with multiplicity histogram 1:106,021, 2:76,227, 3:780, 4:5; no pair exceeds four. This supports a four-patch capacity for this exact geometry only. It does not prove actual glue-law behavior, assembled-box execution, other mesh resolutions, or any altered HMA footprint. Five-patch input remains rejected.

## Reproduce

From the overlay directory:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 cpu_boundary_contract_check.py
cmake --build build-source/build-overlay --target numi-matter-cardboard-full-contact-armijo-check -j2
build-source/build-overlay/matter/numi-matter-cardboard-full-contact-armijo-check build-source/matter/materials/hajali2009_liner_hill_asymmetric_c1_transfer.nmatter --cpu-only
```

The exact shipping pair-multiplicity program is compiled against frozen source 091 headers using the command in its JSON report. `manifest.json` binds source, patch, executable, metallib, CPU outputs, and hashes. No native GPU run is included.
