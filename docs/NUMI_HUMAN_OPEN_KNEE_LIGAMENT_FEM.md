# Open Knee(s) exact ligament FEM preflight

The native Apple mechanics preflight now consumes the exact `ACL`, `PCL`,
`MCL`, and `LCL` tetrahedral regions from the bilateral `NHKNEE1` payloads.
It does not replace the source topology with straps, lines, or generated
primitives.

## Device result

Both sides passed on Apple M4 Pro with 38,159 FEM nodes and 159,416
tetrahedra. Every ligament retained source rigid-node attachments on both the
femur and tibia, generated nonzero reactions on both bodies, and transferred
those reactions into the generalized-force buffer in the same Matter/rigid
transaction. A rejected step restored the accepted FEM state, and replay from
the initial snapshot was bitwise identical.

- Left: maximum displacement `2.31927e-7 m`, determinant range
  `0.999719-1.00025`, femur/tibia reaction L1 `0.00831492/1.18204 N`.
- Mirrored right: maximum displacement `2.99231e-7 m`, determinant range
  `0.999774-1.00029`, femur/tibia reaction L1 `0.00827715/1.05561 N`.

The source right-payload compiler initially reflected node positions without
reversing connectivity parity. Matter correctly rejected every mirrored
tetrahedron as inverted. The compiler now swaps the first two indices of every
mirrored tet4 and tri3, preserving volume and surface orientation without
changing topology or attachment membership. The passing right transcript is
from the corrected payload.

Raw device transcripts are retained in
[`numi-human-open-knee-ligament-fem-v1`](media/numi-human-open-knee-ligament-fem-v1/).
That directory also contains four visually inspected native frames for each
side. In those frames the ligament surfaces are world-bound to the accepted
`NHKFEM1` nodes; the renderer fails closed if the snapshot is incomplete or is
combined with a different articulated pose.

## Evidence boundary

This is an exact-topology, two-body attachment, reaction-transfer,
rollback/replay preflight under a prescribed sub-micron tibia displacement.
The per-region isotropic matrix uses the Open Knee `c1` and bulk parameters,
but Matter does not yet apply the source transverse-isotropic fibre field or
initial prestretch. The result is not a loaded flexion validation, clinical
validation, production-cadence solve, cartilage contact solve, or visual proof
of a flexed ligament. The earlier single-rigid-body flexion images remain
rejected evidence until accepted deformable nodes own their rendered surfaces.
