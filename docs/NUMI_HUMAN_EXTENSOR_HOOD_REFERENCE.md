# Numi Human extensor-hood tensile-network reference

Status: **bilateral digits 2-5 FP64 mechanics and source-posed preflight**. The
conservative nonlinear network now consumes eight explicit `NHHOOD2` ray
records compiled from the pinned MyoSim hands. It has not yet entered the live
whole-body Metal force transaction.

## Mechanical contract

`NumiHumanTensionNetwork` represents a tendon sheet as an explicit graph of
axial collagen bundles. Each element uses

`T = (E A / L0) max(L - L0, 0)`

and therefore carries tension only. The solver minimizes total strain energy
minus external-force potential, computes fixed-bone reactions, and fails
closed unless free-node equilibrium converges. It reports force and moment
closure, strain energy, active/slack bundles, deterministic replay, and
transactional rollback. It produces attachment forces rather than direct
joint torques.

## Anatomical and material basis

The reference topology contains the extensor medial band, radial and ulnar
interosseous lateral bands, extensor hood/sagittal anchor, medial and terminal
bone attachments, and four intercrossing fiber bundles. The model receives
separate EDC, radial interosseous, ulnar interosseous, and lumbrical forces.

The topology and material family follow the open primary study *The Bundles of
Intercrossing Fibers of the Extensor Mechanism of the Fingers Greatly Influence
the Transmission of Muscle Forces*:
<https://pmc.ncbi.nlm.nih.gov/articles/PMC12271525/>. Its reported starting
ranges are 65--157 MPa Young's modulus, 0.3--1.8 mm2 band area, and 0.01 mm2
individual intercrossing-fiber area. The reference uses 90--110 MPa,
0.3--1.0 mm2, and 0.01 mm2 respectively. Each of the four muscle inputs is
2.9 N, matching one studied loading condition.

The canonical geometry is literature-scale and deliberately transparent. Main
bands start at 1% shortening and intercrossing fibers at 5% shortening; the
latter is sub-millimetre in this layout and lies within the study's reported
rest-length perturbation scale. These are preflight parameters, not identified
subject-specific tissue properties.

## Apple M4 Pro result

The full 11-node, 13-element network converged with 9 active tensile elements:

- maximum free-node residual: `1.09454e-9 N`
- force-closure residual: `1.09210e-9 N`
- moment-closure residual: `4.47423e-13 N m`
- strain energy: `0.0287704384 J`
- middle attachment reaction: `2.88127415 N`
- distal attachment reaction: `8.69156346 N`
- bilateral mirrored tension error: exactly zero
- replay: bitwise
- malformed-topology rollback: verified

Removing the four intercrossing fibers while retaining identical muscle loads
changed the combined middle/distal attachment reaction by `0.02643295 N`.
This is the required nonvisual witness that the hood network transfers force;
it is not decorative geometry or an added coordinate spring.

## Promotion path

1. Replace only each declared distal MyoSim route share with solved attachment
   reactions; retain one rigid generalized-force authority.
2. Add an FP32 Metal implementation in the owning Human command buffer and
   qualify against this FP64 oracle.
3. Re-run whole-body recruitment and require finger/wrist residuals to fall
   without degrading wrist, PIP/DIP, root support, replay, or rollback.
4. Build a separately sourced thumb extensor apparatus; do not clone the
   non-thumb expansion onto the thumb.

Machine-readable evidence:
[`m4-pro.json`](media/numi-human-extensor-hood-reference-v1/m4-pro.json).

## Source-posed promotion result

`numilab-human` revision `f09a782` compiles 24 named MyoSim route sites, 28
literature-inferred tensile bundles, and ten exact EDC5/EDM/RI5/lumbrical/UI5
route bindings and their cut ordinals into a 2,036-byte `NHHOOD1` payload. Its header embeds the
pinned MyoSim archive hash; the runtime rejects source, ABI, record-size,
range, laterality, or topology drift before solving.

On Apple M4 Pro, both source-posed fifth-ray networks converged under the
published 2.9 N input condition with 24 active elements, `0.096332941933 J`
total strain energy, `2.71920e-8 N` maximum free-node residual,
`5.66165e-10 N` maximum force-closure residual, and `4.72477e-10 N m`
maximum moment-closure residual. Replay was bitwise and malformed topology
left the previously published result unchanged.

The artifact also preserves MyoSim's source-default force at every binding
(maximum `56.4591 N`). A second solve applied those forces and projected the
paired interface/attachment reactions through exact analytic point Jacobians.
It retained force closure (`2.87e-11 N` root residual), moment closure
(`1.60e-6 N m` root residual), and bitwise replay.

That projection is also a useful negative result. It generated only
`1.15978e-6 N m` at each fifth-MCP ab/adduction coordinate, versus the
approximately `0.074 N m` residual in whole-body recruitment. Distal hood
force transfer alone therefore cannot close that gap and must not be inserted
as a cosmetic correction. The next investigation returns to upstream route
replacement, activation/recruitment constraints, and intrinsic/extrinsic
moment-arm ownership before any live transaction is changed.

Source-posed evidence:
[`m4-pro.json`](media/numi-human-extensor-hood-source-v1/m4-pro.json).

## All non-thumb rays

`numilab-human` revision `48bb878` replaces the hand-ambiguous `NHHOOD1`
record with eight explicit side/digit `NHHOOD2` rays. Digits 2-4 bind EDC,
radial and ulnar interosseous, and lumbrical routes. Digit 5 alone adds EDM;
the compiler does not fabricate that branch for the other fingers. The 7,044
byte artifact contains 84 exact source-posed nodes, 100 inferred tensile
bundles, and 34 exact muscle-route inputs.

On Apple M4 Pro, all eight networks converged under both the 2.9 N reference
load and source-default MyoSim forces. Source lengths span `4.74472-50.5758 mm`.
Every ray has distinct metacarpal, middle-phalanx, and distal-phalanx fixed
anchors and a nonzero generalized-force projection. The source-force solve
reported `1.44026e-7 N` maximum free-node residual, `1.41851e-7 N` force
closure, `9.50881e-8 N m` moment closure, bitwise replay, and verified
malformed-topology rollback.

The maximum per-ray internal generalized responses were `0.478663`,
`0.150819`, `0.657259`, and `0.0375946` for digits 2-5 respectively, mirrored
to numerical precision. The fifth-MCP ab/adduction projection remains only
`1.15978e-6 N m`; therefore the earlier conclusion still holds: distal hood
transfer alone does not explain the whole-body fifth-MCP residual.

Both hands were rendered on M4 Pro from front, oblique, side, and rear with 17
source muscles and 89 route segments per hand. A separate display-only pass
projected 99 source sites per hand to the nearest BodyParts3D triangles to
inspect endpoint contact. Those spheres are visual diagnostics; the exact
unprojected MyoSim sites remain mechanics authority, and the cyan centrelines
are not tendon surfaces.

All-ray evidence and the multi-angle frame paths are recorded in
[`m4-pro.json`](media/numi-human-extensor-hood-all-rays-v2/m4-pro.json).
