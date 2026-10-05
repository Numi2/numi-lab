# Explicit rigid digits for the resting reference

`numi human resting-run --rigid-hands` (native option
`--resting-rigid-hands`) fixes the reference adult's forty digit coordinates
at source qpos0 through the existing Metal joint-equality solver. The wrists,
arms and floating root remain free. Every body mass and inertia, MyoSim muscle,
compliant tendon and registered attachment remains present. Constraint reactions
are internal to the articulation; bed contact still supplies external support.
The native viewer labels the approximation. Hand physiology and manipulation
are not simulated by this mode.

The source-bound compiler admits only the selected MyoSim archive, its 129/128
q/v layout and 51 source equalities. It retains those rows byte for byte and
adds forty fixed-coordinate rows in the existing runtime ABI. The selected
coordinates are q43–62 and q81–100; every target must lie inside its source joint
limits, and any overlap with a source equality dependency rejects admission.
The added rows enter accepted-state identity. A standing initialization cache
cannot be reused for this reduction. Default launches remain unchanged.

This is a declared mechanical resolution choice, not measured infinite joint
stiffness or a claim that a healthy hand has fused joints. Rigid hand segments
also appear in [Rajagopal et al.'s whole-body model (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5507211/).
That is precedent for the reduction, not validation of this resting model.
The existing optional wrist/finger passive-stiffness law is a separate option;
its aggregate measurements must not be described as isolated capsule forces.

The alternative skin-weight repair was tested at the retained six-second and
sixty-second native poses. Local geodesic reweighting preserved neutral geometry
within 0.083 micrometre but did not remove the hand crossings. An offline static
preview using Numi's existing FP64 kinematics reproduced the native crossing
counts and then changed only the forty digit q values:

| Recorded pose | Original right/left self-crossing pairs | Rigid-digit preview |
| --- | ---: | ---: |
| 60 s dynamic release, cap 0.01 | 943 / 910 | 0 / 0 |
| 6 s release with existing upper passive law | 328 / 395 | 0 / 0 |

The native position oracle error was at most 0.837 micrometre. These are exact
triangle tests within explicitly defined dominant-owner hand subsets, including
the repaired skin caps; they do not establish whole-skin clearance. The static
preview advances no physics and is not acceptance of the new dynamic mode.
Native contact, accepted-state, rejection/replay and full-horizon results are
recorded separately before promoting a resting scene.

The subsequent source018 native run completed 3,000 steps (6 s) on the Mac
mini's M4 Pro. All forty digit q values matched their initial references
exactly and their final velocities were zero. Complete outer-skin exact tests
at steps 639, 2783 and 2999 found zero forbidden pairs. The body retained 72 kg,
bed contact supplied support, and the rejection probe preserved body, muscle,
circulation, gas and Brain history; retry matched uninterrupted replay.
This closes the sampled hand defect, not the full-duration anatomical gate.

The reduction took 192.362 wall seconds, or 0.03119 times real time, before
optimization. Its 91 equality rows exceed the existing 64-row fast path.
The paired prior 320-second body-release diagnostic retained hand crossings,
eight final forearm crossings and slow lateral drift; it is not a successful
anatomical baseline. The known lung/rib and cardiac-wall defects also remain.
Exact launch/source hashes, traces, recording identities, full-skin results
and the failed first audit wrapper are retained under
`matter/tools/evidence/human-resting-20261005/hand-native-032`.
