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

## Four-step resident continuation and forced rejection

A separately compiled `--bounded` probe used source SHA-256
`c1082b74175fef5faba56f8a96b89a071d19e7f32c31cacc56b1e0dddab746e4`
and executable SHA-256
`a81e3268b4568d171de565d4cb0024e7ae548760e8a95cde31351c42cda45413`,
with the same linked owner, metallibs, material and 155-body synthetic fixture
identified above. Its exact native output is retained as
`docs/evidence/human-costal-20261107/bounded-continuation.log`, SHA-256
`aa2f993b9c1781fe8d63893204fc98d9a5bfb7f4c6d62c596593d7a639c75462`.
The shell wrapper attempted to assign zsh's read-only `status` variable after
the executable finished and failed with `zsh:1: read-only variable: status`.
The executable exit code was therefore not retained and is not inferred.

The log records four accepted device-resident owner steps. Transaction IDs
18178–18181, generations/slots 1–4, and control steps 47–50 each passed the
nonzero two-attachment-reaction and exactly-once synthetic HumanIO publication
gates. The last two steps had maximum per-step FEM displacement 0.000541304 m
and 0.000723634 m, with raw attachment forces respectively
(-0.000201561, -0.000111627, -0.000121111) and
(0.000166537, 0.0000209524, 0.0000138376) N, then
(-0.000448025, -0.000175646, 0.000102172) and
(0.000376944, 0.0000459895, 0.0000239346) N. Per-node displacement and
all transaction identities are in the log.

The fifth attempt used transaction 18182, generation/slot 5, control step 51,
and intentionally forced rejection. Its final log line reports the enumerated
Matter authority restored and HumanIO rejected, after the checks for a resolved
prepared capability, no HumanIO publication, and the expected tendon adapter
encode and borrow counts. This does not compare the live articulated owner's
q/v/MyoSim state after rejection. It is four short synthetic steps, not sustained
loading, a force-accuracy result, a v5 admission test, or anatomical validation.

## Resident physical-state rejection observer

A separately built `--owner-rollback` probe retained the four-step fixture and
used the actual owner physical-state observer immediately before and after the
forced fifth rejection. Its source SHA-256 is
`bd995dc5f871982b30cc2660b062bb2fa470b04df842620dec597cddf1cc1832`; executable
SHA-256 is `04c8e6156168ea09d391e6301225c4979d84f8dde8301e2835e1645d972d544c`.
The exact output, including start/end UTC and exit code 0, is retained at
`docs/evidence/human-costal-20261107/owner-physical-rejection.log` with SHA-256
`cecbc4ef2b4df0e330e14b5d431c6457a540da298f30f9741198e2fd87c8ff12`.

The prior accepted root retained transaction 18181, generation 4 and accepted
token 11323465167329197731. Its four physical arenas compared bitwise before
and after the rejected transaction 18182: root translations 48 bytes, q 644
bytes, v 640 bytes, and MyoSim state 16 bytes. The existing enumerated Matter
authority and exactly-once HumanIO rejection checks passed in the same run.
This tests these named authorities for one synthetic forced rejection; it does
not establish a restorable full cross-owner checkpoint, all derived state,
physical force accuracy, native v5 admission, or anatomical validity.

## Synthetic native v5 construction

A bounded v5 constructor probe was run on Apple M6 at 22:50:21 UTC and
returned exit 0. Its synthetic NHTBIND2 package has 14 decoder-coverage
regions, 448 attached nodes and 112 disconnected tetrahedra, with artificial
owner bodies 20 and 25. It is deliberately not an anatomical thorax. The
fixture generator, synthetic registration, binding, cartilage, cooked world,
CPU cook output and exact native output are retained in the source tree under
`apps/` and `docs/evidence/human-costal-20261107/v5-synthetic/`. The generator
was invoked with `--rib-body 25`. The original candidate used body 21 and the
CPU cook rejected it because its donor was not a positive dynamic mass
distribution; its failure is retained. No result was attributed to that failed
candidate.

The CPU cook reports two partitions, eight rebase pose checks and maximum
attachment rest error 5.7742002290694927e-08 m. The native test rejected a
wrong binding fingerprint with asset-failure status 3, then constructed the
v5 runtime with the exact synthetic package and verified the combined source
fingerprint `8757514333639552933`; world fingerprint is `8141185403901143359`.
The binding SHA-256 is `0a8c24b4e1ae919188095858763698d9f2cfa5991a64954015030da201c2a59d`,
the cartilage SHA-256 `4241e17f251f70b279d25dcf97b2484ed81f77bf6597f5ca39c7e9b97d3d8374`,
and the package SHA-256 `0187d2ce30cac21273b1cdbaee6c00ed373a4c428354d2295719e8fe39975a6a`.

Native admission source SHA-256 was
`6a77119e8769787f68de3b318eec437f3b0f35de57ee7885dc271d05b83ab71d`;
fixture generator SHA-256 was
`0098fbf251e651f4f9e18c2b808bf4a39614a3d4a34efd708f8bbef8dd21fbe8`;
executable SHA-256 was
`6f74c8b07fc8dac3962df7552cf919133080340642c7544ed202768adf70806f`.
The link map (retained alongside the log) records the v5 constructor from
`NumanXRuntimeV1.mm` object SHA-256
`04038c4730fc73e859c0e66e84b5853af3d8229f6707c76203ed4f7b80ac347b`
(source SHA-256 `6eccdf71f40352146c2437a3512c3b73722f85930317e97eeabb9a7a27949e2f`),
and the NHTBIND2 compiler/decoder object SHA-256
`7a8768f21a49aea1fb55d579b07ab40ec080847d2ec30679c8fa70b80be4228a`.
Remaining services came from `libmetalrobo.dylib` SHA-256
`4400a6ce7f3605fa7c7accc5a4a4fb9814219a5cbe8aaa0b3dc790f8b6872b23`;
that shared library was rebuilt at 22:48:15 UTC, before this executable was
linked. This identity is different from the dylib in the earlier owner trials;
those results are not relabelled. A copy of the executable, dylib, both objects
and link map is retained on `ssh numi` at
`/Users/numi/numi-v5-admission-artifacts-20261107/`. The map SHA-256 is
`5dcef13de013b491af0d07429fac1a3bafb70dcec35ea7766b7e247a03d2bfdc`.

The source rigid, muscle, support and source-compliance equality assets had
SHA-256 respectively `2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4`,
`e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b`,
`4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907`, and
`c99f69b6c217e0bce518cb4db7b27416be0d36f9a1cef26490cf8773850847df`.
The visual pack and vision profile were copied from the retained macmini
v5 workflow, hash-verified at both ends; their SHA-256 values are
`b8ab7cd8deead1e2af3137a727eaad74cd7e4238d77d9116d774bbf304936e37` and
`166220533c6c78695f6e55412f2e832eaaa4ec94ddf791b476111dc323c652a6`.
The Matter and MetalRobo metallib identities remain those recorded above.
This is native constructor admission and identity qualification only: it has
not executed or restored a v5 step and does not establish realistic anatomy,
connected cartilage mechanics, force accuracy, or a surgical task.

## Brain publication probe preparation and build failures

The bridge exposes public aggregate readers, but no full accepted physical
runtime snapshot-and-restore operation. A public aggregate is not a substitute
for q/v/MyoSim, Matter and Brain authority restoration.

A separate attempt to run the historical real-Brain joint-publication test
copied the retained `numi-brain-2to1-source-20260928` snapshot to an isolated
`ssh numi` directory. The unmodified test source SHA-256 was
`612dd01a28155b9c6af99e4a678a465f2839392cc2db28ab3ce0ffd19cc551f0`;
its package SHA-256 was
`72ec253e1adf9802cb2495b81a812a43988078c607a040339b084f6168e9317c`.
The first build failed when Swift 6.4 attempted to compile
`BorrowedHumanMotor.metal` alone despite its stated dependency on concatenated
`NumanXMotorReady.metal` source. The second isolated build changed the shader
resource from `process` to `copy` and the historical test's expected 2871
anchors to the synthetic fixture's 448; it failed because this machine's
Command Line Tools lack XCTest and the Xcode developer frameworks. Both exact
build logs and the test-only difference are retained in the `brain-standalone`
evidence directory. No native publication was executed in either build.

An isolated research dylib was linked from 141 retained integration build
objects with only the previously identified current v5 runtime and NHTBIND2
objects substituted. Its SHA-256 is
`5bcb3160a155e20713197f5a5360706833c9677e758350d7f3a396383cc40a37`;
its exact link command and input hashes are retained alongside the logs. It is
not the library used by the earlier native admission. A standalone extraction
of the actual Brain motor, witness, preflight, ACK and joint-publication
sequence is being built without XCTest. Its first build failed on the internal
factory visibility and two throwing check expressions; that build is retained.
A test-only factory wrapper calls the existing factory unchanged. This is
preparation, not evidence of an accepted v5 step or complete restoration.

## Failed real-Brain publication attempt and stage diagnosis

The standalone XCTest-free extraction was built after adding an isolated empty
C translation unit for the header-only ABI target and forwarding wrappers to
the existing internal Brain factory, ACK and applied-validation methods. No
witness or transaction implementation was replaced. Its source SHA-256 was
`c9cb81cc9e5511b8eedf723873d2db257744a82a2f95542971cb4fc77479e413`; its
executable was `9f0348e34661a05c7c4543bdb45ef7a9ff9c99ed642d77ca8513f646b850ccf2`.
It dynamically loaded the research dylib identified above. The run at
23:01:32–23:01:38 UTC failed with exit 139 after native world admission and
Matter initialization. Its crash report identifies EXC_BAD_ACCESS in Swift's
error formatting, rather than the original failing operation. The raw log and
crash report are retained. It did not establish accepted publication.

For this run the shared `MetalRobo.metallib` had been rebuilt at 22:55:58 UTC,
after the earlier 22:50:21 v5 admission. Its SHA-256 was
`55b021d249bc1bf3e40f94158556c93910d85d3611d4d7fc7ab4bda800d6b2f3`.
The prior admission's `66c694...` shader identity remains unchanged. Exact
copies of the failed executable, shader, Matter metallib, visual/profile and
crash report are retained under `ssh numi` in
`/Users/numi/numi-v5-admission-artifacts-20261107/first-publication-assets/`.

A separately built diagnostic added stage markers and did not format caught
errors. Its executable SHA-256 was
`c71266c7ddd48dbfdd3d692b8774e946325944d0f3365900be5b6582bb31120e`.
All 32 copied model, shader, library, executable and Brain bundle inputs passed
the retained SHA-256 manifest. The run at 23:02:47–23:02:49 UTC exited 1:
v5 runtime creation and identity checks passed, then real Brain runtime
construction threw an unformatted error before control or a physical root
was started. The stage log and input check are retained. This is a failed
integration diagnostic, not a publication, rollback or force result.

The separately identified typed Brain-construction diagnostic ran at
23:05:22–23:05:23 UTC with executable SHA-256
`ff718ba29c5f39c8b5d9f0b6b9c0607aa95b8fafe3ed7a9dad61ff0e8cc35e2f`.
All 32 immutable inputs passed the retained manifest, SHA-256
`46e2baa07c6cb5373850b5347a6b346522a81fb72169a7f0a56181fe72bac6b5`.
The v5 world was created, and Brain construction reached the fast-tissue
factory before exiting 139. Its crash report records `objc_retain` in the
test-only qualification wrapper; the underlying construction error is not
identified. The stage log, build, input manifest, wrapper, factory diagnostic
diff and crash report are retained. No Brain witness, physical step or v5
publication was reached.
