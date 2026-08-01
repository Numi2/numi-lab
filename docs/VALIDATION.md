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
contract evidence; it does not qualify unimplemented sensor modalities.

The same owner places an articulated tool against a compliant static obstacle,
samples a six-axis force/torque sensor at the tool COM, and compares all six
native channels with TaskIR's independent reduction of the same committed
NumiSolver constraints. It requires a nonzero finite wrench and verifies that
a mid-rollout reset clears the sensor view. This qualifies resultant contact
wrench sensing; it is not yet joint-transmission load-cell or full tactile-
schedule evidence.

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
remaining historical probes are temporary migration diagnostics and must be
folded or deleted with their replacement owner.

## Numerical corpus

Paired FP64, Metal, and MuJoCo cases cover:

- contact-free motion and every supported joint type;
- lower/upper limits and mixed limit-contact islands;
- impacts, stacks, friction cones, rolling/torsional friction, and CCD;
- equality constraints, tendons, gears, and attachments;
- actuator delay, damping, armature, saturation, backlash, and presets;
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
