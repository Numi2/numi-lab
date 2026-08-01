# Validation and release gates

Tests establish bounded behavior on a recorded revision and platform. They do
not by themselves establish physical fidelity, sim-to-real transfer, or
competitive superiority.

## Evidence model

`schemas/capability_matrix.json` is the claim registry. Its generated output is
[CAPABILITIES.md](CAPABILITIES.md). A qualified capability must name an owning
CTest and have a committed evidence manifest recording that check as passed.

Evidence manifests record:

- source revision, artifact and metallib fingerprints;
- hardware model/chip, OS, and thermal state;
- execution plan, solver settings, topology, contacts, and rows;
- p50/p95 latency, submission time, throughput, and failures;
- memory by session/environment/contact/sensor/tape and high-water mark;
- exact checks, corpus cases, seeds, and acceptance thresholds.

Historical manifests remain immutable. A newer run supersedes evidence only
for the checks and contracts it actually exercised.

The TaskIR owner includes exact-rate and non-divisor SensorIR schedules,
whole-sample latency, mid-rollout reset, actor/critic consumer permissions, and
accepted-state ordering for final-policy observations. This is scheduling and
contract evidence. Its IMU case drives a kinematic rigid parent through a known
velocity change while rotating the sensor frame, and checks specific force,
gravity removal, local angular rate, TaskIR binding, and final compact
publication against the analytic result.

The same owner places an articulated tool against a compliant static obstacle,
samples a six-axis force/torque sensor at the tool COM, and compares all six
native channels with TaskIR's independent reduction of the same committed
NumiSolver constraints. It requires a nonzero finite wrench and verifies that
a mid-rollout reset clears the sensor view. This qualifies resultant contact
wrench sensing; it is not yet joint-transmission load-cell or full tactile-
schedule evidence.

Its contact-state case independently mirrors the accepted Metal constraint
stream on the host and checks active state, block count, summed normal force,
resultant tangential force, and maximum penetration. A semantic obstacle-body
filter must include that contact while a ground-only filter stays exactly
zero. An unresolved filter must reject compilation without replacing the
previous program, and the current WorldPack binary round trip must preserve
and re-resolve the semantic filter rather than persisting a packed body index.

The corruption case runs clean and corrupted copies of the same native world.
All six twist channels and pose translation match a host mirror of the shared
counter RNG; pose orientation matches the same Gaussian tangent perturbation
and remains normalized. Identical seeds replay outputs and metadata exactly,
a changed seed separates the stream, and a reset selects the next episode key.
A 100% dropout sensor publishes an all-zero coherent sample with `fresh` and
`dropped` set, `valid` clear, and the dropped bit observable through TaskIR.
WorldPack round-trip evidence preserves all three authored corruption fields.

The scalar-joint SensorIR case resolves an authored joint name to immutable
joint and q/v indices, runs three contact-free control steps through a
joint-only native graph, and requires its two output channels to equal the
final accepted generalized state exactly. It binds both channels and validity
into TaskIR, rejects an unresolved name without replacing the previously
compiled program, verifies that the execution plan omits kinematics pipelines,
and round-trips the semantic target through the current WorldPack format.
Fixed and multi-DOF joint observations remain rejected rather than being
silently flattened.

The physical/SensorIR transaction case first commits one articulated-tool
contact, its manifold, and a nonzero pose history in a resident session. Reset
then moves a second kinematic obstacle into contact against a one-pair
operational capacity. It requires the exact pair-overflow status, byte-identical
restoration of q/v, scene bodies, manifold headers/points/counts, SensorIR
metadata and compact outputs, a still-valid resident token, and a successful
following submission after the second obstacle moves clear. SensorIR is
installed without TaskIR to prove that its runtime ownership is independent.

The TaskIR transaction case repeats the same failure with native TaskIR and
PolicyIR installed. A reference resident session advances normally while a
candidate session attempts the rejected reset. The rejected result must retain
its typed physics-error transition. The following accepted transition must be
byte-identical between branches across q/v, scene/manifolds, actor and critic
observations, policy samples/log probabilities/values, task transitions,
SensorIR output/metadata, and contact evidence. This qualifies persistent
per-environment TaskIR rollback. The sensor set includes IMU, noisy pose, and
filtered dropped-contact canaries, so their kinematic state, corruption episode
identity, histories, metadata, and compact publications must also return to the
reference branch; global curriculum scheduling remains outside that claim.

The native integration owner runs five PPO updates through a three-slot shared
rollout ring. It therefore covers slot reuse, managed MLX payload release,
monotonic policy revisions, native rollout serialization, learner checkpoints,
deployment-policy publication, immutable policy topology, stale-revision
rejection, transactional private-bank swapping, and direct native command-
buffer publication into opaque rollout leases with no duplicate compact host
result or CPU payload scan. It does not yet qualify a no-copy MLX-to-bank blit.

The TaskIR owner fills a one-slot rollout in two native submissions, checks
every compact float and transition byte against the owning Metal result,
verifies the terminal bootstrap offset, rejects visibility before sealing and
a second simultaneous lease, then releases and reuses the slot. A second
leased-only execution requires zero compact learning elements in the ordinary
result while retaining status records and exact payload parity. This is the
offset, lifetime, and transactional-publication owner; PPO integration is the
cross-language lifetime owner.

The same path runs a generated-ABI Metal reduction across all compact floats
and transitions. The CPU accepts the lease only when its 64-byte status echoes
the current ABI, exact element counts, policy revision, task fingerprint, and
unique submission token with zero failures. An unpublished result cannot leave
a reusable resident session token.

The TaskIR owner also executes a nonzero deterministic dense policy over two
environments and four control steps, then compares all eight native SIMDgroup
outputs against an FP32 host dot-product reference. This qualifies the current
dense operator contract, not convolution, recurrence, attention, reduced
precision, or performance floors.

The same owner uploads policy A, uploads a weight-only revision B with the same
topology fingerprint, then reactivates retained A. It requires exactly two
bank uploads and one reuse and verifies both numerical outputs. An activation
change must produce a distinct topology fingerprint.

The TaskIR and Metal-contact owners also validate compiled pipeline selection.
Every pipeline member and Metal entry point comes from the generated runtime
schema. A task/policy/sensor execution and the comprehensive contact/CCD/rod
execution must each create fewer than the full reachable inventory, later
policy revisions must not recreate pipelines, and three asynchronous state
arenas must share one immutable pipeline cache. This qualifies feature-group
selection and cache ownership, not the unfinished Metal 4 command-allocation
and argument-table migration.

## Owning checks

The intended consolidated suite is:

1. compiler/model/generated-ABI;
2. physics/NumiSolver/rod/constraints;
3. TaskIR;
4. SensorIR and tactile;
5. PolicyIR and learner boundary;
6. visual presentation;
7. native differentiation;
8. end-to-end Swift/Metal integration.

Each subsystem has one focused correctness executable. Examples demonstrate
product flow. One benchmark executable owns performance and long soaks. The
historical executable probe forest has been deleted; new standalone probes are
not accepted in place of extending the relevant owner or benchmark profile.

The compiler/model and TaskIR owners preserve semantic site identity through
the current WorldPack and TaskPack formats, reject unresolved sites
transactionally, and compare compiler-composed task and sensor site frames
against their equivalent authored body-local transforms.

The TaskIR owner also executes episode-sampled translation and tangent
orientation goals plus a ping-pong SE(3) trajectory. It mirrors the stable
counter key and analytic interpolation on the host at every accepted step,
forces one environment into a new episode, checks actor and critic values, and
replays the complete run byte-for-byte. Invalid trajectory timing is rejected
without replacing the last compiled program, and the current TaskPack
round-trips the complete goal contract.

The same owner compiles world-linear and world-angular Jacobian components for
a named articulated frame and generalized-velocity coordinate. It requires
one immutable compiler-owned point-query packet, round-trips the semantic
coordinate through TaskPack, rejects an unresolved coordinate without
replacing the compiled program, and checks `J_linear * v` against the native
frame-origin linear velocity while independently checking the analytic angular
column. The common articulated operator owns both the ordinary contact
Jacobian path and the six-row semantic path; a distinct task-only kinematics
implementation is not allowed.

The TaskIR owner also compiles a scalar SignalIR graph whose truth leaves are
that semantic Jacobian and a named frame-twist SensorIR channel. Metal samples
the sensor after accepted physics, materializes the authorized value before
reward evaluation, then publishes actor/critic sensor histories. Each generic
reward is checked against the same step's post-physics SensorIR value, so a
one-step-stale ordering cannot pass. The bounds termination remains inactive,
TaskPack round-trip preserves the graph, and a forward reference is rejected
without replacing the last valid compiled program.

The same fixture expresses fixed, episode-sampled, and trajectory frame
objectives without constant or frame-specific reward/termination opcodes.
TaskPack 16 round-trips those graphs, preserves their numerical outcomes, and
proves that deleting the duplicated goal fields and shrinking the termination
record does not change dynamic-goal replay.

The G1 branch-pruning fixture now lowers all nineteen rewards into SignalIR,
including heading-frame command tracking, yaw tracking, joint acceleration,
action delta, compiler-resolved soft limits, mechanical power, phase/contact
matching, slip, forbidden contact, and frame-based foot clearance. The native
reward loop has no opcode switch: its record is a signal channel, reporting
channel, and weight. Three recorder bindings and the curriculum success metric
also consume authored SignalIR nodes. The compiled task contains 92 nodes, 167
semantic sources, eleven reductions, three frames, and three compact recorders.

Paired eight-environment, sixteen-step native rollouts against the preceding
revision cover zero and deterministic nonzero host actions. Physical, contact,
failure, termination, reward, and all eight reporting channels are exact. All
zero-action recorder values are exact. With nonzero actions, tracking and root
height are exact and the graph-composed tilt differs by less than `6e-11` from
the removed duplicate inline expression. The compact transition remains 96
bytes. The recorder/curriculum program adds 512 retained bytes: 160 immutable
private, 192 transient private, and 160 shared boundary bytes; persistent
physical state remains byte-for-byte unchanged.

The named-command hard cut is compared independently against `d889456` with
the same eight environments, sixteen control steps, ground scene, seed, fixed
eight-step Swift chunks, and disabled scheduled resets. Both zero actions and
the deterministic host action stream preserve every physical, contact,
failure, termination, reward, recorder, and reporting metric exactly. The G1
TaskPack now resolves `forward_velocity`, `lateral_velocity`, and
`yaw_velocity` to three immutable scalar command records; the owner rejects
duplicate identities, unresolved observation bindings, and invalid ranges
transactionally.

The subsequent topology-sized command cut is compared against `b7f0188` with
the same paired rollout contract. Zero and deterministic host actions again
preserve every physical, contact, failure, termination, reward, recorder, and
reporting metric exactly. The runtime state no longer embeds a three-element
command vector: compiler-sized scalar state stores contact reductions followed
by all named commands, and the complete stride is journaled transactionally.
A five-command fixed-base Metal fixture consumes command five through SignalIR
and recorder output; its rollback variant makes that command episode-random,
forces a reset followed by pair-capacity failure, and requires the next
accepted transition to match an untouched reference branch byte-for-byte.
TaskRuntime's observe, apply, effort, complete, and curriculum bindings now
come from the generated ABI schema; no raw numeric Metal buffer bindings remain
in that shader. G1 retained memory increases by 32 bytes overall while both
persistent and transient state decrease by 32 bytes.

An independent fixed-base fixture records `axis_velocity_squared` and compares
the Metal recorder value with the generic reward-channel contribution. The
compiler also rejects duplicate recorder identities, unresolved recorder
signals, more than three compact recorders, and a multi-level curriculum with
no success SignalIR binding without replacing the last valid task.

The same fixture compiles a two-source SensorIR reduction over the current
pose-height value and validity bit. The compiler assigns two dense semantic
source slots, Metal evaluates the sum of squares, and every transition matches
the independently published accepted post-step critic sensor value. A sensor
without truth permission is rejected transactionally. The scratch cost is two
floats per environment for this fixture; the bundled G1 task has no SensorIR
signal sources and its retained, transient, shared, and persistent allocation
totals are exactly unchanged from the preceding revision.

The owner additionally rejects an invalid reward channel, invalid soft-limit
factor, ignored source parameters, and an empty reduction while preserving the
last compiled task transactionally.

## Numerical corpus

Paired FP64, Metal, and MuJoCo cases cover:

- contact-free motion and every supported joint type;
- lower/upper limits and mixed limit-contact islands;
- impacts, stacks, friction cones, rolling/torsional friction, and CCD;
- equality constraints, tendons, gears, and attachments;
- actuator delay quantized to the control period, deterministic backlash play,
  damping, armature, saturation, dry friction, torque-speed envelopes, reset,
  rejected-step rollback, and fingerprinted presets;
- multiple articulations and dynamic bodies in one island;
- rod stretch, bend, twist, buckling, attachments, tool contact, self-contact,
  friction, and energy behavior.

MuJoCo is an offline comparator. It is never loaded as a production fallback.
Trajectory comparisons use matched model, actuator, timestep, contact, and
observation contracts with explicit tolerances.

## Product qualification

- G1: fingerprinted mechanics, torque/contact trace parity, deterministic
  posture control, 60-second zero-command standing, then `+0.1 m/s` tracking,
  turns, pushes, and terrain across 256 held-out seeds with at least 95%
  completion and no systematic pitch/roll failure.
- Franka: reach, grasp, and multi-body manipulation in matched MetalRobo and
  MuJoCo scenes.
- Wave: synchronized native camera, deformation, fingertip six-axis F/T,
  tactile history, and visual-tactile policy input without host readback or
  Python scheduling.
- Policy: dense, recurrent, and visual inference parity; atomic history/reset;
  stable policy revisions; no full simulator state entering MLX.
- Differentiation: finite-difference tolerances from the numerical contract and
  rejection of topology-changing cases.

Policy transfer is judged by success and outcome distributions under the same
action, observation, actuator, timing, reset, and termination contracts—not
bitwise cross-simulator trajectories.

## Safety and soak

Run on the dedicated Mac mini in this order:

1. 100,000 debug-canary contact-heavy transitions.
2. 1,000,000 randomized/reset stress transitions.
3. 10,000,000 optimized release transitions.

Acceptance requires zero GPU restarts, hangs, invalid dispatches, silent
contact loss, partial publication, or retained-memory growth beyond 2% after
warm-up. Same hardware/OS/build replay is bitwise; other Apple GPU generations
use declared tolerances.

## Performance floors

On an M4-class qualification machine:

- 150k environment control-steps/s for 4,096 contact-free Franka worlds;
- 40k for 1,024 contact-rich Franka worlds;
- 25k for 2,048 contact-rich G1 worlds.

These are release floors, not current claims. Results without hardware, OS,
thermal, topology, latency, memory, and failure context are not publishable.

## Claim discipline

The project does not publish “MuJoCo/Isaac competitor,” “state of the art,”
“TGS,” “sim-to-real,” or equivalent claims until the corresponding registry
entry and evidence gate are qualified. Unsupported capabilities must fail
compilation or be clearly identified as offline/experimental behavior.
