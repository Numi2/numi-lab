# Open Knee(s) exact ligament and patellar-tendon FEM preflight

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

## NHKFEM2 patellar-tendon extension

`NHKFEM2` adds the exact `PTL` region: 9,280 nodes, 35,616 tetrahedra,
553 tibial attachment nodes and 424 patellar attachment nodes. The full
transaction therefore covers 47,439 nodes and 195,032 tetrahedra with femur,
tibia and patella as distinct reaction owners. Bilateral accepted/rejected/
replay receipts and eight inspected frames are retained in
[`numi-human-open-knee-tissue-fem-v2`](media/numi-human-open-knee-tissue-fem-v2/).

After correcting the femoral axial sign, the left PTL generated tibia/patella
reaction L1 values of `1.02410/0.00368505 N`; the mirrored right generated
`1.05174/0.00345063 N`. Total femur/tibia/patella reaction L1 was
`0.0166385/2.14116/0.00368505 N` left and
`0.0087464/2.19477/0.00345063 N` right. Both runs transferred nonzero patellar generalized
force and retained bitwise replay plus rejected-step rollback.

The old accepted-looking images had passed topology and FEM checks but used an
ambiguous flexion-axis sign that placed patella posterior and fibula medial.
They are retained only under
[`rejected-axial-sign`](media/numi-human-open-knee-tissue-fem-v2/rejected-axial-sign/).
The replacement source compiler requires anterior-axis alignment plus explicit
patella-anterior and fibula-lateral offsets before a payload can reach this
mechanics preflight.

The quadriceps tendon (`QAT`) is intentionally not reassigned to a bone. Its
distal nodes tie to the patella, while the source model uses a separate rigid
quadriceps-origin (`QSO`) construct proximally. Completing it requires a
distributed quadriceps muscle-load boundary, not a cosmetic femur anchor.

## Evidence boundary

This is an exact-topology, two-body attachment, reaction-transfer,
rollback/replay preflight under a prescribed sub-micron tibia displacement.
The per-region isotropic matrix uses the Open Knee `c1` and bulk parameters,
but Matter does not yet apply the source transverse-isotropic fibre field or
initial prestretch. The result is not a loaded flexion validation, clinical
validation, production-cadence solve, cartilage contact solve, or visual proof
of a flexed ligament. The earlier single-rigid-body flexion images remain
rejected evidence until accepted deformable nodes own their rendered surfaces.
