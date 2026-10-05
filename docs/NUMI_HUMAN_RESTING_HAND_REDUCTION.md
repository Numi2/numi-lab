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

The subsequent existing-solver optimization keeps larger equality factors in
their existing device buffer while parallelizing independent elimination and
forward-substitution rows. Pivot selection and each row's FMA order remain
unchanged; the 64-row threadgroup cache remains in use for smaller blocks.
The paired six-second native result is bit-for-bit identical for both traces,
the initial and three accepted geometry packs, and terminal body state.
Total measured GPU time fell 39.53%; wall time was 119.966 s (0.05001 times
real time). A 32-step control/candidate check also preserved the original
51-row configuration exactly. This is a local comparison with concurrent CPU
asset preparation, not a general benchmark. See `equality-cooperative-039`.

The following 24-second run completed 12,000 accepted steps at 0.05065 times
real time. Complete outer-skin checks at 10, 18 and 24 seconds found zero
forbidden pairs. Its 18–24 s center-of-mass slopes were 0.0446, 0.0401 and
0.00275 mm/s; it is a settling-body diagnostic, not demonstrated equilibrium.
The recording retains 376 unretimed frames over 476.14 wall seconds.

The existing projected-contact solve now also handles blocks larger than the
64-row factor cache, using their existing device factors. The 91-row resting
case previously fell back to interleaved equality corrections. This changes
the finite-iteration body trajectory: it is not a bitwise-equivalent rewrite.
In the paired six-second native run, measured GPU time fell another 29.98%,
to 86.194 wall seconds (0.06961 times real time). Physiological trace columns
were identical; body/contact columns changed. The three accepted full-skin
checks remained clear, all digit coordinates stayed fixed, and rejection/retry
checks passed. The 51-row native regression remained bit-for-bit identical.

The production-kernel test now exercises 64, 65 and 91 equality rows against
an independent reduced-mass and energy oracle, including physical support;
5,068 checks passed on the Mac mini. A packed-triangle cache experiment made
no meaningful measured difference and was not adopted. Exact evidence and
both successful and failed trials are retained in `projected-equality-044`.
The long-duration, complete breathing-cycle and cardiac anatomical gates
remain open; these solver results do not qualify them.

The ordered source-limit solve subsequently extends each SIMD lane's existing
equality ownership from two rows to three, covering the 91-row resting block.
It retains the original limit order, per-DOF FMAs and existing buffers. Against
source022, the six-second run preserves both traces, initial and three accepted
geometry packs, and terminal q/v bit for bit. Measured GPU time falls 22.34%,
with 69.164 wall seconds (0.08675 times real time). The original 51-row run is
also bit-for-bit unchanged; its short timing difference is not a claimed gain.
See `limit-lanes-047` for exact compiled identities and retained evidence.
