# Synthetic costal ownership native gate

This is an explicitly invoked, experimental synthetic integration gate, not a
registered anatomical thorax or full NumanX v5 runtime admission. It derives
its four-node tetrahedron and floating, revolute two-body model from the CPU
fixture. The in-memory version-2 ownership assigns tissue mass across both
bodies and puts a fixed attachment on each. It uses cooked attachment records
to create the passive Matter adapter, rather than decoding an `NHTBIND2` file
or invoking NumanX asset admission. The prescribed motion uses the rebased
model and its analytic kinematics and Jacobians. It is not a dynamic breathing
simulation and has no physical calibration.

Run the built target explicitly with the matching shader library as its sole
argument. Admission checks both mass partitions and the common runtime
attachment ownership validator, then initializes the Matter runtime and
adapter. The accepted trial imposes a 0.0001 rad joint angle and checks a
successful solver status, nonzero FEM displacement, and finite nonzero
reactions at both attached nodes. It deliberately injects a failed owner
status for the next transaction and verifies bitwise restoration of FEM node
state. Finally it restores the initial Matter snapshot and checks bitwise replay
of FEM nodes, attachment reactions and generalized force outputs. It does not
claim byte-for-byte rollback of every runtime state or full NumanX admission.

Real rib/sternum body provenance, separate anatomical ownership, physiological
motion and experimentally calibrated tissue properties are still missing.
An in-memory synthetic two-body map cannot supply those inputs.

## Initial native trial, failed

On Apple M6, the initial probe (source SHA-256
`583eb19ae55d36852c48d4191fe8621c2466a5228acafa0aea52b5699b3f6383`,
executable SHA-256
`6f9d87cc0b26215ad553711605afcefdb1eecd255cffaeec6e5076ca5f52ea5f`)
reported Matter and adapter initialization, then failed `encode transaction`
(exit 1). The first version omitted the adapter diagnostic, so the cause was
not determined from that run. No movement, rollback or replay is qualified.
The metallib SHA-256 was
`6f3182e1d78071c20afc1da7efb54927da4c3f724a758e552e5293fe8ea52f99`.

The executable directly compiled `src/core/NumiHumanTissueBinding.cpp` from
`bc71d385`, so the ownership compiler and validator are present. It linked
ArticulatedDynamics, EngineModel and NumiHumanTissueMass from retained archive
SHA-256 `ae265d71c8b20b04dc4bd02078bbe6a657a911d2cfd9aa1d4635614a09c5fb1c`,
and Matter compiler, runtime and Human adapter from retained archive SHA-256
`c206b9912670eb0187c113a615d8380c8637928e5fc24ae916fd962fff71a105`.
The NumanX runtime admission changes are not linked in this trial.

A diagnostic-only rerun (probe SHA-256
`106b2a05e630853e720060de3ad131a2b7a78d27f2d1dd8505b0c0a48b7e4dca`,
executable SHA-256
`a8fe729d804cab57862c2a06dd9ed10dcb719d0f839616c59153cc5405f8c183`)
identified `Human body-contact BVH node buffer is unavailable`. The baseline
adapter only allocates that buffer when body interaction exists, yet checked
it unconditionally. Its always-dispatched status kernel also binds the optional
neighborhood buffers; the experimental correction provides typed empty rows
when there is no body interaction and still requires the BVH buffer for a real
interaction. All other BVH consumers are gated by body sample count or body
interaction. The revised native probe adds a second mode with an active
body-contact sample and requires an accepted, nonzero normal-force audit.

## Guarded adapter trials, still unqualified

Both trials used probe SHA-256
`282d5d24b330dc19aa6da4de0b06427c9415eed5b45e093d01612228774b7dbe`,
the locally compiled guard adapter SHA-256
`5f0a506039bfd6ce4ceb7e3571fee35ca2423072bcb73b4bf6fd8b2553066d29`,
executable SHA-256
`2b7a8cf31702557983556f76651841bb774d6ac612c9cc1362cba6938d8e697f`,
and the metallib above. The linker map binds the current ownership compiler and
adapter directly; articulated mechanics and Matter runtime/compiler remain the
retained baseline archives. The full NumanX admission implementation is not
linked.

Both the empty interaction and one-body-contact variants passed adapter buffer
allocation and then failed pre-dynamics encoding, exit 1:

```
articulated Matter IPC requires a compatible primal coupled-candidate service plus exact q/v and candidate-body capacity
```

The one-contact variant reports one uploaded sample, but its command did not
reach a completed step. Neither variant establishes loaded movement, actual
contact force, rollback or replay. The fixture supplies prescribed poses and
Jacobians only; cooked Human attachments require the actual coupled-candidate
service and corresponding q/v and body arenas. Supplying a fake service or
weakening this production check would not qualify the runtime. The guard itself
also remains unqualified until a valid coupled execution exercises both modes,
including nonzero accepted body-contact force. No runtime repair is promoted.

## Standalone adapter controls, passed

The same locally compiled guard was tested with a separate ordinary-fixed-node
Matter world and the same prescribed two-body pose. This world does not contain
cooked Human attachments and is not the Human ownership runtime trial. The
binary SHA-256 was
`62a04cab8cf4902a98e8338ac08ad3b03aef05ac2052c4b80a500daaff9933f8`;
the probe source SHA-256 was
`f288232035c51812d24eca7379058d35ae7b683a5aea01c1bfd5add3ed392bc7`.

Both controls passed on Apple M6. With no body contact, the imposed joint motion
produced 1e-6 m maximum displacement and nodal reaction magnitudes 0.000334903
and 0.00482041 N. With a real body-contact sample, its accepted audit recorded
9.90054e-05 N normal force and one closed sample; displacement was 1e-6 m and
nodal reaction magnitudes 0.000376295 and 0.00482045 N. Both checked bitwise
rollback of FEM nodes after an injected owner rejection and bitwise replay of
nodes, reactions and generalized forces after restoring the initial snapshot.
No full-state or NumanX transaction restoration is established by these controls.

The probe output text was subsequently corrected to label the standalone mode
as an adapter control. Source and binary identities above identify the exact
versions actually run, not the subsequently edited source.

## Real NumanX owner preparation

The separate real-owner probe extends the retained 160-DoF owner fixture (155
articulated bodies, plus a preceding world body) with a synthetic two-donor
binding. It uses the existing owner force-reject proof/apply path. Its initial
binary `cd71724b16011ec12620f3817edc58e9850a55b1b69aed0319e0eec1e068eddc`
stopped before device work because the inherited material path did not exist.
The corrected material path had SHA-256
`d3094476cefa1a28ca8b5f35b28e2586d5a44ce63f268e57e96812de32de4a5f`,
and the corrected binary
`5b74fa2e4a199f779a0c57a3917f693d58a2ea11aa139952e0d4017e8779eb3e`
stopped before device work with `costal cook failed: costal source kinematics
failed`. Both were linked against `libmetalrobo.dylib` SHA-256
`552ef872612e8f38f2f6e190dde8487aad137e47b9e4ca3aecef8d8303473a31`,
MetalRobo metallib SHA-256
`66c6947d3db340c98526b9ac03a9c27cddee6b5b6f0ae53be3a5939af10fe0bf`,
and the Matter metallib above. The two failures did not exercise the owner.

The kinematics failure exposed a real world-index issue: the costal compiler
and v5 admission previously passed the entire world body array as output to a
query requiring exactly the articulation body count. The corrected paths pass
a slice beginning at `firstBody`, retain global indices in the backing array,
and reject donor or attachment bodies outside the articulation. The CPU
regression passes with an actual leading world body, retains attachment indices
1 and 2, and rejects an outside-world donor, an out-of-range attachment and
an invalid articulation range. It does not execute v5 runtime admission.

The corrected real-owner probe passed the existing force-reject lifecycle on
Apple M6, with two cooked donor partitions and two Human attachment records,
inside the 155-body, 160-DoF synthetic articulated chain. It retained the real
MetalArticulatedOperator, NumanX owner, Matter proof and force-reject apply;
the existing equality check reports restoration of accepted Matter authority
and rejection of HumanIO publication. The probe source SHA-256 was
`09b5b161d44dd0b1ed972f3f734beadeb1881813150d0f84db329574c388a68f`,
the compiler source was
`57a750186bd10b6a68bb8c50581e6cf266ad339e131808d8d4773d52ac1bdbfc`,
and the executable was
`e88de5ea7cc39e8d32a9afb02e053462e5241b8286006a0aa9600133aaa4b4d0`.
It used the retained dynamic library and both metallibs identified above, plus
the directly linked guarded adapter and current ownership compiler. This is a
force-reject result only: it does not establish an accepted loaded step, full
NumanX v5 construction, replay of a committed step, a two-body owner or
anatomical validity. The v5 indexing change compiled in a syntax-only check but
was not linked into this probe.

## Accepted owner trial and wrong-channel instrumentation

With the same 155-body synthetic model and 0.001 rad/s initial relative joint
velocity, executable
`0c165864063faab8028a2f183cb6ced0f7d882ad7e784ab1cbab562d4c8417b3`
published an accepted step but failed its two-anchor force gate. A diagnostic
repeat, executable
`672d97040a11f1f43461993a84d0d16848c2cbf35dc9f720d624b87a259fbd5b`,
reported four zero vectors from the ordinary fixed-node reaction buffer while
the staged generalized load had L1 magnitude 0.00532348 and joint-6 component
-0.000102367. Its node displacement vectors in metres were:

```
0:  0,          -0.00017029,   1.49012e-08
1:  7.45058e-09,-0.00017035,   1.49012e-08
2: -5.21541e-08,-0.000172734, -5.34505e-05
3: -8.19564e-08,-0.000110447,  1.3113e-06
```

The rejected instrument used the static fixed-node channel. The public Matter
API explicitly identifies the tag-2 Human attachment raw-reaction channel as
the force on Human; its rows correspond to the cooked attachment records.
The two failed instrument source versions are preserved under
`docs/evidence/human-costal-20261107/` with SHA-256
`c4277960e044bda3fd9bd7fc5a1c8611400a5fac2a94d5a96d1f2020598b004a` and
`80215bf11636a8e869fabe91fe4efeb311abf9a1bed690c9a47cb8a4f5149e83`.

The corrected-channel executable
`040af78320a191cacef454fb29a0ec5fc5234ccd9722820d5d977ebacf444409`,
source SHA-256
`6adf1ea6fe0ab30f58b32bf3ac336ca75b034e3982c91a5b0ac79ff6bff5894a`,
passed the same accepted publication and nonzero-load gates. The synthetic Brain
witness and ACK enabled real owner publication; they do not qualify a Brain
integration. Maximum displacement was 0.000180815 m. Raw attachment reactions:
node 0/body 1 = (-0.00414287, -0.00369722, -0.00176924) N, magnitude
0.00582778 N; node 1/body 2 = (0.0040943, 0.0000282371, 0.0000204071) N,
magnitude 0.00409445 N. The same generalized load values above were observed.
This establishes a nonzero synthetic load, not a force accuracy test, native v5
asset admission, committed replay, physiological behavior or anatomical ownership.

## Committed Matter restore and loaded replay

The replay probe, source SHA-256
`3859c3d61422433eec43159c9280afdf8e0f9287eb9603f0caeecdf4d76e8447`
and executable SHA-256
`ecf7f0f14bec5b234568b4458e554da3a86fc49d47bb1c5b764ccd6dcf214789`,
passed on Apple M6 with the same library, metallibs, material and fixture.
Each of two fresh real-owner contexts completed an accepted step with the
synthetic Brain witness, ACK and exactly one HumanIO publication. Both had the
same nonzero motion and attachment reactions recorded above. Each restored its
committed Matter snapshot and compared the explicitly enumerated accepted
Matter authority fields; each also restored the tendon adapter snapshot and
compared its authority digest. The second owner restored the first run's
initial Matter snapshot before executing its step, then compared accepted
Matter authority, both raw attachment-force records and all 160 generalized
reaction floats bitwise to the first run.

The reconstructed owner started from the same authored Human input; no live
articulated-owner q/v/MyoSim snapshot was restored and no next step was
continued from the committed state. This is not complete owner restoration,
force calibration, native v5 construction, an anatomical thorax or a
physiological breathing result.

## Two-step device-resident continuation

The `--continue` probe, source SHA-256
`7f8ac65a15eb9653fb2fe1623b3a1e9d12315c4fd80fb02f614213b2a462d1a4`
and executable SHA-256
`59759648d9ecf1b5d7cbfbc637d5d993822ae005e2ca9f0b414974981419ea64`,
passed on Apple M6 using the same 155-body synthetic owner and binaries above.
After its first accepted, published step, it named the preceding transaction
and generation and submitted a second step in the same owner and Matter
contexts without host q/v/MyoSim reset. Both synthetic Brain witness/ACK
transactions published exactly once. The second step had maximum FEM
movement 0.000359071 m relative to its own starting snapshot. Its raw Human
attachment forces were (-0.000504344, -0.000446525, -0.000505431) N at node
0/body 1 and (0.000479609, 0.0000150794, 0.00000956645) N at node 1/body 2;
magnitudes were 0.000842145 N and 0.000479941 N. Generalized reaction L1 was
0.000221391 and its joint-6 component was -0.00000258352.

The earlier independent force-reject test restored its Matter authority, but
it was not a rejection following this two-step run. This result establishes
only two consecutive synthetic native owner steps, not sustained loading,
force calibration, native v5 admission or anatomical fidelity.
