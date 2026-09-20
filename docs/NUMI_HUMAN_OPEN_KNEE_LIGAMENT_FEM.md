# Open Knee(s) exact ligament and patellar-tendon FEM preflight

## ABI 2 source-directed material upgrade

`NHKNEE1` ABI 2 now admits the exact homogeneous fibre direction and the
`c1`, `c2`, `c3`, `c4`, `c5`, `lam_max`, bulk-modulus, and in-situ-stretch
values for `ACL`, `PCL`, `MCL`, `LCL`, `PTL`, and `QAT`. The source identity
gate now includes `FeBio_custom.feb` SHA-256
`00b6efb53ad7e7330296cbb9569d358d48ed60819e22732e6149db6fb98a158a`;
the prior ABI omitted that file even though it owns the fibre vectors.

The live Matter material consumes each registered/mirrored unit fibre axis and
uses the source deviatoric `c1` matrix plus a smooth tension-only exponential
fibre term. A compact `expm1_minus_x` constitutive opcode keeps its stress and
tangent differentiable without duplicating the full fibre-stretch expression.
This is a source-shaped Apple-GPU approximation, not the exact FEBio
piecewise exponential-linear energy: Matter does not yet provide FEBio's
exponential-integral primitive or its straightened-fibre branch.

The first bounded attempt to jump directly to the final ACL/MCL/LCL in-situ
stretch rejected at LCL with nonlinear-solver status 10. Raising the iteration
budget kept the Apple GPU at 99--100% utilization for 12 minutes without an
accepted transaction, so that path was rejected as both unstable and
inefficient. The admitted source values remain in ABI 2, while the current live
solve applies neutral stretch (`1.0`) until a staged prestrain/equilibrium ramp
owns initialization.

On Apple M4 Pro, the left neutral-stretch directional preflight passed all
47,439 nodes and 195,032 tetrahedra with bitwise replay and verified rollback:
maximum displacement `2.93542e-7 m`, determinant range
`0.999768--1.00017`, femur/tibia/patella reaction L1
`0.0602688/2.11899/0.0197677 N`, compile time `249.975 ms`, accepted-step wall
time `166278 ms`, and peak RSS `706871296 bytes`. The whole accepted/rejected/
replay qualification took `476.37 s`.

The mirrored-right gate passed the same topology and transaction boundary:
maximum displacement `2.35136e-7 m`, determinant range
`0.999543--1.00025`, femur/tibia/patella reaction L1
`0.0475741/2.08083/0.0181325 N`, compile time `247.656 ms`, accepted-step wall
time `158776 ms`, and peak RSS `706838528 bytes`. Its registered fibre x
components have the expected opposite sagittal sign while y/z and unit length
are preserved.

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

## Source-directed transverse-isotropic v3 qualification

Revision `5882143` admits the six Open Knee ligament/tendon regions only when
their FEBio-authored homogeneous fibre axes and material parameters survive the
`NHTENDON3` ABI. The Matter law is an explicitly labelled smooth,
source-shaped approximation; source in-situ stretches remain serialized but
the qualified step uses neutral stretch pending staged prestress equilibrium.

The left live Human qualification transferred `2765.11 N` from the four
quadriceps routes through QAT and `2787.97 N` through PTL. Patellar/tibial PTL
reactions were `2787.83/2788.34 N`; the `264442`-tet transaction retained a
Jacobian range of `0.99976-1.00028`, bitwise replay, and rejected-step
rollback. The seven articular pairs evaluated `69701` samples with force and
moment residuals below `7.7e-9`. Four inspected `512x512` live frames and a
machine-readable receipt are retained in
[`numi-human-open-knee-transiso-v3-left`](media/numi-human-open-knee-transiso-v3-left/).

The bilateral nonvisual probe separately passed for left and mirrored-right
source axes. The right-side fibre x components reverse sign while y/z remain
preserved. This is a reflection/registration check, not independently
segmented right-knee evidence.

## Exact passive axial fibre-transfer v4

Revision `3dc8777` replaces the neutral smooth axial approximation for `PCL`,
`ACL`, `MCL`, `LCL`, and `PTL` with the exact source FEBio piecewise
exponential-linear fibre-stress branches. Each reduced element connects the
centroids of the two exact source enthesis attachment-node sets. Its effective
area is the source tetrahedral volume divided by the reference centroid
separation. The exact source tetrahedral volume remains live as neutral
matrix-only FEM, and its `fiber_scale` is zero so the axial response is not
counted twice. `QAT` retains its deformable directional material and active
extensor-chain ownership.

The reduced force is applied as an equal/opposite pair to the two bones through
their point Jacobians in the same borrowed Human command buffer. A GPU audit
requires five active regions, an accepted transaction marker, finite positive
force, bounded stretch, and force/moment closure. The CPU reference test covers
the slack, exponential, and straightened-fibre linear branches plus stress
continuity at `lam_max`.

The bilateral source cook measured fibre-axis/centroid-axis alignment from
`0.99644` through `0.999931`. At the source pose the nonzero tensions were
`3.34181 N` ACL, `55.4378 N` MCL, and `8.40188 N` LCL, giving an expected
two-endpoint L1 force of `134.36298 N`. The final mirrored-right Apple M4 Pro
transaction measured `134.384201 N`, `55.453167 N` peak tension, effective
stretch `0.999998629-1.034004688`, zero force residual, and
`6.143906e-8 N m` moment residual. Its `264442`-tet solve retained Jacobians
`0.999714375-1.000294328`, bitwise replay, and rejected-step rollback.

An initially rigid host gate rejected the physically valid `0.999998629`
minimum stretch even though the tension-only law correctly permits slack. The
gate now accepts positive bounded slack while retaining the force, closure,
determinism, and rollback requirements. Bilateral focused MCL probes also
passed with `15693` exact nodes and `62712` tetrahedra per side.

All eight `512x512` runtime frames were inspected. Patella remained anterior
and seated, fibula remained lateral, the QAT-patella-PTL chain was continuous,
and no detached or inverted tissue was visible. Frames, raw transcripts, the
rejected validator receipt, hashes, and the machine-readable qualification are
retained in
[`numi-human-open-knee-exact-axial-v4`](media/numi-human-open-knee-exact-axial-v4/).

This production path is deliberately narrower than a volumetric prestress
claim. FEBio defines the exact stress branches used here, but its complete
transversely isotropic continuum also defines a fibre strain energy and a
three-field formulation. A homogeneous in-situ stretch jump and rate-staged
ramp both failed the bounded nonlinear/performance gate. For an image-derived
fixed reference geometry, the superior continuum path is an iterative,
generally per-element compatible prestrain-gradient solve rather than a larger
homogeneous jump. See the
[FEBio material definition](https://febiosoftware.github.io/febio-feature-manual/features/solid_material_trans_iso_mooney-rivlin/)
and the [general prestrain framework](https://pmc.ncbi.nlm.nih.gov/articles/PMC7651410/).

## Evidence boundary

The original `NHKFEM1/2` result is an exact-topology attachment,
reaction-transfer, rollback/replay preflight under a prescribed sub-micron
tibia displacement. The live result additionally proves a same-command-buffer
passive continuum reaction changes the full Human rigid state for one bounded
step; it is not a sustained or production-cadence rollout.
The current passive axial owner applies the exact source FEBio stress branches
and source in-situ stretch between source enthesis attachment-node centroids.
It is a reduced force-transfer law in parallel with neutral matrix FEM, not the
exact FEBio fibre energy or a compatible prestressed continuum field. It also
does not provide a spatially varying fibre field. The result is not a loaded
flexion validation, clinical validation, production-cadence solve,
full-resolution cartilage volume solve, or visual proof of a flexed ligament.
The active extensor-chain result proves reduced nonlinear tendon force transfer
through exact QAT/PTL entheses; it is not an active volumetric tendon solve.
The earlier single-rigid-body flexion images remain rejected evidence until
accepted deformable nodes own their rendered surfaces.
