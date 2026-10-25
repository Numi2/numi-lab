# Connected boundary refinement 114

This isolated increment contains a topology-only owner for conforming subdivision of an oriented O-surface edge after a caller has independently selected and certified a detailed-surface midpoint. It does not discover midpoint paths, run TetGen, change material/reference state, or claim collision or physical clearance.

`human_skin_fat_boundary_refinement.py` requires exactly one appended midpoint per selected original edge, a source-surface provenance record, two oppositely oriented incident O faces per selected edge, and a unique positive parent marker per source face. It handles triangles with one, two, or three selected edges, reuses a midpoint across every incident face, preserves the input coordinate prefix, retains child-to-parent face rows and source markers, and verifies that the selected parent edge is gone while each replacement subedge has two incident faces. The caller-provided provenance must identify detailed source faces/path and emitted sample coordinates; ambiguity and opposed/unresolved normals fail closed. Source parent marker propagation is exact array indexing from the input O face table.

The synthetic suite passes 20 tests, including shared-edge two-face conformity, all one/two/three-edge split cases, orientation, parent marker ancestry, and explicit O/I/D per-column midpoint construction with independent layer-edge maps. It rejects missing/mismatched layer edges, malformed registered offset provenance, boundary edges, ambiguous or opposed source paths, and malformed samples.

The bounded input inventory is `baseline-edge-incidence-114.json`. It cross-checks the pinned baseline-103 exact edge witness set against the compact 084 `source_O_faces` and the 083 PLC marker owner: 147 candidate O-layer edge pairs form 166 edge-bank groups; 145 are actual interior O edges covering 164 groups. Two pairs, (206,215) and (903,1184), are not edges of the source O surface and are excluded. The 084 builder’s marker rule is `100000 + source_O_face_row + 1`; the subdivision output carries each child’s exact parent marker.

The single mapped (329,340) path sample is not inserted by this increment. The source-path witness is deterministic but does not establish shortest-path uniqueness or source-normal compatibility. Its bounded exact segment report checks only the two named banks, not all 185 banks. The full 145-edge source midpoint set and its source patch/path/orientation certificates remain outstanding. No geometry candidate was emitted and no TetGen or native/physics run occurred.

Run the focused suite with:

```sh
/Users/numi/numi-human-prep-venv-20261005/bin/python -m pytest -q matter/tools/test_human_skin_fat_boundary_refinement.py
```


The explicit three-layer entry point is `subdivide_registered_column_edges`. It accepts named coordinate frames, separate O/I/D face/marker tables, a caller-provided `column_id -> {O,I,D: point_row}` map, and selected records with explicit endpoint column IDs and explicit O/I/D edge endpoints. It creates separate appended midpoint rows for each layer. For each frame it evaluates I/D as the detailed O path midpoint plus a linearly interpolated registered endpoint offset field, using the supplied path arclength fraction. This does not assume a `3*N` row layout or infer correspondence from geometric proximity. Each emitted I/D point carries both endpoint offsets, source column weights/IDs, detailed O source path/face IDs, and field/map SHA pins. It remains a topology/reference construction primitive: capacity112 and whole-volume/exact self-intersection audits must run on the generated candidate before any remesh or physical use.
