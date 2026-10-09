# Contoured bed source-pose feasibility

This is a read-only CPU feasibility artifact built from the accepted step-0 `NHSKIN` surface in the 1191 native scene. It does not modify the scene, manifest, contact solver, or any body state; it is not a native contact test or anatomical qualification.

Reproduce on the Mac mini with:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python3.13 \
  /Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196/analyze.py
```

The source script, three imported owner files, accepted skin pack and receipt, original skin asset, scene manifest, and NHCNT payload are SHA-pinned in `analysis.json`. The compact Float32 row-major height arrays are in `heightfields.npz`.

The builder covers every accepted skin triangle by its closed world-XY AABB. Each touched bed cell is capped to the lowest triangle vertex Z minus the scene's 10.0014554 µm source gap; each grid node takes the minimum cap of all incident cells. It caps the remaining surface at 0.25 m, sets the finite perimeter to exactly z=0, and applies a four-neighbor min-plus slope envelope. Grid edges are capped at `0.9999 × min(spacing)/sqrt(2)`; the actual p10-p01 triangle slopes are measured. Float32 node values are rounded downward before export. A separate common-Float32-lattice triangle audit reports zero AABB candidate pairs and zero intersections for both grids.

The 5 mm candidate is the more faithful source-envelope sample: 260×460 cells, 261×461 nodes (120,321 heights), 239,200 triangles, height range −0.116 mm to 240.386 mm, and maximum measured slope 0.9998884. Its minimum same-XY skin-vertex gap is 10.0015 µm; the conservative per-cell AABB lower-bound gap has a 0.275 mm median, with some much larger bounds where triangle AABBs over-cover. The 10 mm candidate has 130×230 cells, 30,261 nodes, 59,800 triangles, maximum height 235.950 mm, slope 0.9999013, and the same minimum vertex gap. Both remain inside the existing 1.3×2.3 m bed footprint and below the 262,144-height limit.

The arrays use rows indexed by y and columns indexed by x, with x varying fastest; each heightfield uses the p10-p01 diagonal matching the requested `00_10_01__10_11_01` manifest triangulation. The 5 mm fields are `x_0.005`, `y_0.005`, and `heights_yx_0.005`; the 10 mm fields use the `_0.01` suffix.

The current 1191 manifest still describes a flat four-vertex/two-triangle bed and has no `bed.heightfield`; its NHCNT1 payload still has an old z=0 plane. This analysis deliberately records legacy NHCNT1 witness locations only as a reference. The intended replacement whole-skin support query must be implemented and verified separately before these arrays can be used as physical contact geometry. No force, support reaction, or motion result is inferred here. The 0.25 m target is a cap, not a medically or anthropometrically validated contour.
