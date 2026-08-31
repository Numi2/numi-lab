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

The quadriceps tendon (`QAT`) is not cosmetically reassigned to the femur. Its
distal nodes retain the exact patellar attachment, while the source model's
quadriceps-origin (`QSO`) construct remains the proximal force source. The live
extensor-chain implementation below replaces the complete source quadriceps
generalized-force row with an explicit distributed attachment transfer.

## Live full-Human passive continuum transaction

The visual probe now creates the exact `ACL`, `PCL`, `MCL`, `LCL`, and `PTL`
Matter FEM worlds directly inside the persistent 157-body Human step. It
captures anchor reactions before rigid integration in the same command buffer,
adds those passive reactions to the retained NHTENDON3 source-route force, and
then verifies a rejected-step rollback plus bitwise replay. It does not claim
that passive tissue reaction replaces an active source tendon force.

The bilateral Apple M4 Pro qualification used one `0.1 ms` step and a
`3e-6 rad` knee-flexion perturbation:

- Left: maximum node displacement `2.54740e-7 m`, determinant range
  `0.9997598-1.0002606`, and femur/tibia/patella reaction L1
  `0.529482/2.309817/0.916469 N`.
- Mirrored right: maximum node displacement `2.04857e-7 m`, determinant
  range `0.9998230-1.0002322`, and femur/tibia/patella reaction L1
  `0.630589/2.391164/1.054424 N`.

The accepted presentation combines full BodyParts3D femur, tibia, fibula and
patella meshes with the exact Open Knee articular bone ends. That composite
removes the joint scan's truncated bone ends while retaining the Open Knee
attachment/contact surfaces under the same material. Bilateral four-angle
frames and raw receipts are in
[`numi-human-live-open-knee-composite-v1`](media/numi-human-live-open-knee-composite-v1/).

## Active QAT-patella-PTL-tibia extensor chain

The four source quadriceps keep their nonlinear compliant Hill-type tendon
force law, but no longer apply their complete source row directly to the rigid
patella. The same Human command buffer now:

1. restores each source proximal endpoint;
2. distributes its patellar terminal force over 378 exact QAT attachment
   nodes using attachment-area weights;
3. applies a zero-resultant PTL force couple over 424 patellar and 553 tibial
   attachment nodes; and
4. returns the accepted QAT/PTL fixed-node reactions to the articulated bodies
   before rigid integration.

The left M4 Pro qualification transferred a `1955.27 N` quadriceps resultant
through a `1973.99 N` PTL resultant. The right mirrored qualification
transferred `2048.40 N` through `2068.43 N`. Assembled external-force relative
errors were below `1e-6`; QAT/PTL reaction residuals were below the measured
passive femur-reaction allowance (`0.532618 N` left, `0.630435 N` right).
Both sides retained positive near-unity deformation Jacobians, bitwise replay,
and rejected-step rollback.

Machine-readable receipts and all eight inspected M4 Pro frames are retained
in
[`numi-human-open-knee-extensor-chain-v1`](media/numi-human-open-knee-extensor-chain-v1/).
The right knee is a sagittal mirror registered into the live right-body frames,
not an independently segmented right subject.

## Evidence boundary

The original `NHKFEM1/2` result is an exact-topology attachment,
reaction-transfer, rollback/replay preflight under a prescribed sub-micron
tibia displacement. The live result additionally proves a same-command-buffer
passive continuum reaction changes the full Human rigid state for one bounded
step; it is not a sustained or production-cadence rollout.
The per-region isotropic matrix uses the Open Knee `c1` and bulk parameters,
but Matter does not yet apply the source transverse-isotropic fibre field or
initial prestretch. The result is not a loaded flexion validation, clinical
validation, production-cadence solve, cartilage contact solve, or visual proof
of a flexed ligament. The active extensor-chain result proves reduced nonlinear
tendon force transfer through exact QAT/PTL entheses; it is not an active
volumetric tendon solve. The earlier single-rigid-body flexion images remain
rejected evidence until accepted deformable nodes own their rendered surfaces.
