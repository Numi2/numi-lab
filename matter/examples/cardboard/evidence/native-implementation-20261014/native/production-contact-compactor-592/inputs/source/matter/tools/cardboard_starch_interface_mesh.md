# Finite starch-layer interface mesh preparation

`cardboard_starch_interface_mesh.hpp` prepares an already-conforming tetrahedral
board mesh for an eventual native cohesive-face law. Material indices are
explicit: `0` liner, `1` medium, `2` finite glue. Every tetrahedron must carry
its source density and a proper material-to-reference rotation; the helper has
no glue-density default and never rewrites material frames.

The preparation pass groups tetrahedron faces by their sorted source node IDs.
Only shared `liner/glue` and `medium/glue` triangles become paired-interface
records. Direct liner/medium adjacency is rejected, as is a face with more than
two incident tetrahedra or a boundary/cut-slot tag on an interior face. A
face-level boundary inventory retains each exterior face, its local
opposite-vertex slot, region, and caller-supplied boundary/cut tag. The mesh
must be conforming by shared node IDs: geometrically coincident faces with
different IDs are not guessed or paired.

`PreparedForNativeLaw` duplicates each source node used by a paper/glue
interface once on the glue side, rewrites glue tetrahedra only, and emits a
paired triangular face record. Each record binds paper/glue tet IDs, local
face slots, matching node IDs and reference coordinates, opposing outward
normals, paper region, and reference area. The records are dormant:
`constitutiveLawInstalled` remains false. A native caller still has to bind
accepted/candidate state, traction, energy, tangent, and rollback behavior.

Two topology controls are available. `TiedControl` keeps shared interface
node IDs and emits comparison records. `UnbondedControl` duplicates the glue
side but installs no traction law. The latter is a topology control, not a
complete physical open-interface model; contact after separation is outside
this utility.

The helper computes reference volumes and masses from the source and prepared
tetrahedra. It preserves volume and mass by region, reports total consistent
P1 mass-matrix mass (`sum_ij M_ij = rho V` per tetrahedron), and recomputes
row-sum-lumped nodal masses from the split connectivity. It does not copy the
old nodal masses onto both sides of a duplicated face. This helper does not
apply a glue strength, fracture energy, cure rule, or damage law and does not
claim calibrated starch behavior.

The CPU check constructs a small conforming four-tet fixture with one internal
glue/glue face and both paper/glue interface types. It checks independent and
tied controls, matching face slots/coordinates and normals, preserved cut-slot
tags, unchanged source material frames, regional volume and total mass,
recomputed split-node mass, nonmanifold-face rejection, direct paper-paper
rejection, and invalid material/density/frame/tag rejection. Its densities
and geometry are synthetic topology fixtures. Standalone command:

```sh
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  matter/tools/cardboard_starch_interface_mesh_check.cpp \
  -o /tmp/cardboard_starch_interface_mesh_check
/tmp/cardboard_starch_interface_mesh_check
```
