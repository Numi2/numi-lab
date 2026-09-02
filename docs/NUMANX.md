# NumanX: one embodied transaction

NumanX is the production boundary joining NumiBrain, Numi Human, MyoSim,
NHTENDON, and Matter. It is not a bridge demo and it is not a sequence of
loosely related CPU calls. One control root owns one structurally validated,
timeline-bound motor candidate, one causal physical horizon, one sensor
generation, and one joint Brain/physics decision.

Development is promotion-gated rather than timeline-gated. A smaller result is
never renamed as the final architecture to meet a date. Each boundary becomes
production only after its owning code, failure semantics, replay evidence, and
device evidence all agree.

## Architectural contract

The transaction has one authority chain:

```text
committed brain root
  -> exact NumiBrain motor-buffer leases
  -> Human begin-step excitation
  -> MyoSim activation and source-route J^T
  -> NHTENDON transfer evidence
  -> checkpointed Human/Matter candidate solve
  -> unpublished causal sensor candidate
  -> unpublished NumiBrain fast+cognitive consequence
  -> mutation-free physical proposal
  -> completed Brain preflight and GPU ACK
  -> Matter-then-Human apply or restore
  -> exact joint-publication fence
  -> physical, sensor, and Brain generation release
```

Every device buffer remains owned by its producing runtime. Consumers receive
versioned borrowed views with fixed-width counts, exact strides, exact device
identity, exact GPU addresses, program and transaction fingerprints, and
explicit phase access rights. A callback may encode onto its borrowed command
timeline; it may not commit, wait, retain, replace, or synchronously read a
borrowed resource.

The protocol's FNV-1a fingerprints are deterministic integrity and replay
identities inside a trusted same-process, same-device boundary. They are
unkeyed and collision-weak, so they are not cryptographic authentication,
tamper resistance, or an adversarial security boundary.

Command completion is transport evidence, not publication authority. Human
diagnostics must certify the exact NumanX program and exact command-buffer
identity before a sensor candidate can replace the previously published
generation. Rejection and late GPU failure quarantine the candidate and leave
the last accepted generation byte- and identity-stable.

### Joint Brain/physics publication protocol

The production protocol is a proposal/ACK/apply/publication transaction across
the legacy Metal physics queue and the Metal 4 Brain queue. A queue signal is
only ordering and liveness evidence; it is never itself an acceptance
decision. Authority comes from versioned records plus terminal command
completion.

```text
Physics queue
  checkpoint Human and Matter
  consume exact NumiBrain motor leases
  solve the coupled Human/Matter candidate
  write an unpublished HumanIO sensor candidate
  prepare Matter's success-surviving accepted-state proof
  validate the terminal physical record at command completion
  advance physical-prepared liveness

Brain consequence queue
  wait physical-prepared
  validate the canonical prepared token and sensor leases
  prepare fast and cognitive consequence in private shadow state
  finish every fallible journal/pointer preflight
  write BrainCommitPreflight and advance preflight-ready liveness

Proposal / ACK / apply
  owner emits a mutation-free proposal
  Brain GPU validates proposal + preflight + fast gate + witness
  completion handler validates Brain ACK and advances ACK liveness
  Matter applies or restores first; Human applies or restores second
  owner emits AppliedOutcome only after terminal apply completion

Joint publication
  validate AppliedOutcome and reserve every fallible release
  flip private fast+cognitive pointers while readers are gated
  write the exact COMMITTED joint-publication fence
  publish the reserved HumanIO sensor candidate and physical root
  latch the Brain generation under the same aggregate reader gate
  release the root only if all three domains agree
```

No command-buffer host wait or sensor-payload readback occurs in this
handshake. Every wait and signal value is monotonic and reserved before
submission. Completion handlers may inspect compact owner control records to
decide whether a generation remains releasable or terminally quarantined;
those records are never physical-state substitutes. Resources, checkpoints,
borrowed leases, and both shadow generations remain quarantined through the
final publication decision. If a command fails before checkpoint authority is
known complete, the legal result is terminal no-touch quarantine, not an
unsafe attempted restore from potentially partial checkpoints.

The accepted physics token must contain a GPU-derived fingerprint of the live
accepted Human and Matter state. A hash of transaction metadata, buffer
addresses, or program identity alone is not a physical-state proof.

## Physics ownership

The implicit coupled unknown contains Matter field state and a variable-size
Human state. The current full-body asset has `nq=129` and `nv=128`; the
runtime's q/DoF capacities are 161/160. Coupled operations iterate only the
logical `nv`, never invented padding coordinates. Matter may ask the Human
owner for four operations:

1. candidate kinematics and point Jacobians;
2. source-step effective-tangent action;
3. source-step effective-tangent preconditioning;
4. staged publication of the accepted generalized reaction.

The unconstrained first production operator is the frozen source-step tangent

```text
A0 = M(q0) + armature + h D.
```

Candidate kinematics and attachment Jacobians remain exact and nonlinear,
while Matter's Newton/FGMRES Human block uses this byte-stable quasi-Newton
operator. The stand solve must factor the same `A0`; otherwise a staged
reaction does not reproduce Matter's accepted velocity increment. A proper
constraint projector cannot be inserted as a square full-space inverse:
constrained production requires an explicit nullspace operator `N^T A0 N` or
an exact KKT/Schur solve.

These operations use the same current Human state and the same constraint
tangent. Support contact and anatomical equality constraints therefore cannot
silently fall back to an unconstrained mass operator. Unsupported constrained
mode fails closed.

The production NHCNT support seam lives in Matter's monolithic KKT, not in a
second Stand contact solve after Matter. Matter imports all ten exact
point-plane rows and their candidate Human point Jacobians. Its nonlinear
residual contains `J^T lambda`; its matrix-free FGMRES action contains the
locally condensed `J^T D J` block for unilateral proximal Coulomb contact. The
resulting Human generalized reaction returns through the staged
`A0 * deltaV / h` path. Owner contact remains disabled for that transaction so
the accepted candidate is not double-solved.

Support multiplier history and the 64-byte consequence for every row are
checkpointed, committed, restored, included in the accepted-state proof, and
persisted by snapshot archive v4. The unpublished candidate consequences feed
the ten-by-seven touch tensor: point position, signed separation, normal
force, friction force, and slip speed. They become visible only with the joint
Human/Matter/sensor/Brain root. An unloaded root may correctly carry zero
normal force; the focused downward-contact GPU probe separately establishes a
nonzero multiplier, `J^T lambda`, `J^T D J`, commit, and exact rollback. A
frozen square projector is insufficient here because the unilateral Coulomb
active set is solution-dependent.

Moving FEM attachments obey

```text
v_node = J v_human
f_human = J^T r_node
```

and must pass the discrete virtual-work identity

```text
p_human^T (J^T r_node) == (J p_human)^T r_node.
```

Force ownership is exact:

```text
Human generalized load = MyoSim source-route load
                       + Matter attachment/contact reaction
```

NHTENDON generalized corrections remain diagnostics. They are not added again,
because MyoSim has already contributed the wrench-equivalent source-route
`J^T` load. A future continuum-replacement muscle mode must explicitly remove
the replaced MyoSim rows before returning continuum force.

The coupled method follows the same monolithic principle used by GPU IPC work:
reduced and full coordinates participate in one residual, with reactions
returned through the transpose map. See Huang et al.,
[GPU-Accelerated Simulation of Deformable Objects with Multilevel GPU-IPC](https://kemenghuang.github.io/img/GUIPC.pdf).

## Causal sensing

The current Human sensor tensor is environment-major, then horizon step, then
muscle receptor. Each receptor publishes ten FP32 features plus a UInt32
validity mask:

1. excitation;
2. activation;
3. fibre length;
4. fibre velocity;
5. path length;
6. path velocity;
7. active force;
8. tendon tension;
9. activation derivative;
10. normalized equilibrium residual.

Geometry and tendon quantities are the current pre-dynamics evaluation;
activation/fibre state is the explicit MyoSim update; delivery occurs after the
corresponding stand step. Metadata fingerprints bind allocation, layout,
provenance, and timing. They do not claim to cryptographically authenticate
private GPU payload bytes. Accepted physical diagnostics, immutable
runtime-owned candidate storage, and ordered production on the same device
timeline are therefore mandatory parts of the integrity boundary.

MyoSim is retained as a source-faithful muscle model, not represented as a
generic torque policy. Its role is consistent with the explicit musculoskeletal
state/action boundary described by the
[MyoSuite paper](https://zenodo.org/record/6818245/files/caggiano22a.pdf).

## Joint decision and rollback

The final device record is environment-major and starts `pending`. It records
the NumanX ABI, decision, environment, step, Human status/completion, and Matter
status/completion. Pending fails closed. Human q/v and Matter continuum state
remain staged until both sides accept. On any Human prepare/stage failure,
Matter Newton/FGMRES failure, post-certification failure, command transport
failure, or caller rejection, both checkpoints are restored on the device
timeline and no brain or sensor generation advances.

Metal 4 command buffers, residency sets, and legacy/Metal event interoperation
are the platform direction for eliminating host serialization; see Apple’s
[Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
and [resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization).

## Research bar

NumanX is evaluated against current embodied-musculoskeletal and coupled-GPU
work, but does not inherit a state-of-the-art claim from architecture alone.

- [MuscleMimic](https://arxiv.org/abs/2603.25544) demonstrates scalable
  full-body muscle control across hundreds of motions and reports strong joint
  kinematic agreement, while explicitly identifying the remaining gap between
  motion imitation and physiological muscle activation. NumanX therefore
  treats pose reproduction as one metric, not the biological validation.
- [MS-Emulator](https://arxiv.org/abs/2603.29332) demonstrates the contemporary
  scale of whole-body neuro-muscular learning at roughly 700 muscles. NumanX
  must report exact active-muscle count, environment throughput, sample
  efficiency, and held-out task coverage before making a comparative learning
  claim.
- [SMS-Human](https://arxiv.org/abs/2506.00071) couples proprioceptive, tactile,
  vestibular, and visual inputs with a musculoskeletal controller. NumanX's
  present proprioceptive packet is consequently a first qualified modality,
  not a complete human sensorium; additional modalities require the same
  causal accepted-state provenance and ablation evidence.
- Recent GPU rigid/MPM work reports very large speedups from weak asynchronous
  time splitting, but also reports a smaller stable timestep and weaker
  coupling than fully implicit alternatives; see
  [Wang et al.](https://arxiv.org/abs/2503.05046). NumanX keeps the monolithic
  attachment residual as the correctness baseline. A future asynchronous fast
  path must be compared against it on impulse, energy, stiction, mass-ratio,
  and convergence error rather than promoted from throughput alone.

Apple's supported direction also permits neural inference to remain on the GPU
timeline through Metal 4 machine-learning passes and argument tables; see
[Running a machine learning model on the GPU timeline](https://developer.apple.com/documentation/metal/running-a-machine-learning-model-on-the-gpu-timeline).
NumiBrain may use custom compute or a packaged ML pass, but either path must
retain exact resources through completion and expose an event-ordered device
boundary without a hot-loop host wait.

## Promotion gates

The following are completion requirements, not optional follow-up work:

- Exact ABI size/version/access checks and malformed/partial configuration
  rejection.
- Valid motor accept, invalid/non-finite/out-of-range motor fail-closed, and
  exact motor kind/morphology/species binding.
- Same-device zero-copy sensor and motor handles with stable GPU addresses.
- Wrong transaction, program, command buffer, generation, modality, shape,
  latency, or accepted-physics token rejection.
- Exact full-body `nq=129`/`nv=128` execution within q/DoF capacities 161/160,
  including under-capacity, over-capacity, stride, uint-index, alias, and
  device rejection.
- Candidate kinematics finite differences, mass/inverse consistency, and
  constrained-tangent checks.
- Attachment virtual work, total impulse, and angular-momentum balance.
- Zero-stiffness Matter gives byte-identical standalone Human output.
- Nonzero tendon transfer cannot enter Human generalized force twice.
- Human failure, Matter failure, post-certification failure, and pre-commit
  callback rejection each restore both runtimes byte-for-byte.
- Accepted transactions increment Human, Matter, sensor, physics, and brain
  generations exactly once and replay bitwise.
- No hot-loop command-buffer wait, CPU payload assembly, or sensor readback.
- Current-revision Apple GPU counters, System Trace, resident-memory accounting,
  and scaling measurements before a performance claim.
- Held-out motion/task generalization, perturbation recovery, and sensor
  ablations; training-set imitation is not a general-control result.
- Muscle activation, tendon force, joint reaction, metabolic/effort, and
  kinematic comparisons against independent experimental or model evidence;
  matching pose alone is not physiological validation.
- Monolithic-versus-asynchronous coupling comparisons across timestep, mass
  ratio, stiffness, frictional stiction, impulse, energy drift, and failure
  rate before selecting a performance mode.

## Executable evidence

From a Release build configured with `BUILD_TESTING=ON`:

```sh
ctest --test-dir build -L numanx --output-on-failure -j1
```

The NumanX label is reserved for executable ownership/rollback/replay gates.
Source inspection, a successful shader build, and an offline visual probe are
useful evidence, but none independently proves coupled physics, performance,
or joint publication.

### 2026-08-31 Gate A runtime qualification

The current source was qualified on an Apple M4 MacBook Air with 24 GB memory,
macOS 26.6 build 25G5028f, Swift 6.3, and Metal 32023.883. The production C ABI
loaded the provenance-fixed 157-body `nq=129`/`nv=128`, 416-muscle Human plus an
attached one-tet Matter world. NumiBrain supplied exact motor-header,
excitation, autonomic, active-sensing, and ready-gate leases; HumanIO returned
416x10 proprioceptive and 416x1 interoceptive candidates.

Four accepted roots were published from persistent device-resident Human
q/v/MyoSim state. A valid stale-predecessor request failed before slot or state
mutation. A sticky-timeout attempt was explicitly force-rejected and restored;
its exact retry reproduced the full accepted token and both sensor payloads
byte for byte before publication. The rejected HumanIO generation remained
unpublished and was not reused.

The focused native matrix passed 10/10 twice, covering transaction, HumanIO,
variable-logical-DoF CoupledHuman, ABI4 owner, exact-candidate admission, real
Human/Matter adapter, accepted-state proof/restore, attachment runtime, and the
full-body bridge. The cross-repository E2E passed 1/1 and the NumiBrain package
passed 144 tests with one skip and zero failures.

This is correctness and replay evidence for the one-environment, one-substep
transactional runtime. The 4.96-second E2E test duration includes pipeline and
fixture setup and is not control latency. The Brain topology is synthetic and
the one-tet attachment is an execution fixture; physical fidelity, biological
validity, learned generalization, and performance comparisons remain open.
