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
