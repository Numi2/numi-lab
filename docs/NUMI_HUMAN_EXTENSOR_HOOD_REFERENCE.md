# Numi Human extensor-hood tensile-network reference

Status: **FP64 mechanics and source-posed preflight**. The conservative
nonlinear network now consumes a bilateral `NHHOOD1` artifact compiled from
the pinned MyoSim hand. It has not yet entered the live whole-body Metal
force transaction.

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

1. Compile right-hand MyoSim EDC5/EDM, RI5, UI5, and lumbrical route sites plus
   BodyParts3D phalanx attachments into this graph.
2. Mirror from source geometry and verify bilateral material/topology parity.
3. Replace only the declared distal MyoSim route share with solved attachment
   reactions; retain one rigid generalized-force authority.
4. Add an FP32 Metal implementation in the owning Human command buffer and
   qualify against this FP64 oracle.
5. Re-run whole-body recruitment and require the fifth-MCP residual to fall
   without degrading wrist, PIP/DIP, root support, replay, or rollback.
6. Extend the topology to all non-thumb rays, then build a separately sourced
   thumb extensor apparatus.

Machine-readable evidence:
[`m4-pro.json`](media/numi-human-extensor-hood-reference-v1/m4-pro.json).

## Source-posed promotion result

`numilab-human` revision `71f638f` compiles 24 named MyoSim route sites, 28
literature-inferred tensile bundles, and ten exact EDC5/EDM/RI5/lumbrical/UI5
route bindings into a 2,036-byte `NHHOOD1` payload. Its header embeds the
pinned MyoSim archive hash; the runtime rejects source, ABI, record-size,
range, laterality, or topology drift before solving.

On Apple M4 Pro, both source-posed fifth-ray networks converged under the
published 2.9 N input condition with 24 active elements, `0.096332941933 J`
total strain energy, `2.71920e-8 N` maximum free-node residual,
`5.66165e-10 N` maximum force-closure residual, and `4.72477e-10 N m`
maximum moment-closure residual. Replay was bitwise and malformed topology
left the previously published result unchanged.

The artifact also preserves MyoSim's source-default force at every binding
(maximum `56.4591 N`) but does not apply that force in this preflight. The
next promotion is to solve the hood from the live muscle-tendon state and
project its attachment reactions through exact point Jacobians once, inside
the owning whole-body force transaction.

Source-posed evidence:
[`m4-pro.json`](media/numi-human-extensor-hood-source-v1/m4-pro.json).
