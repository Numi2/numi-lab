# Persistent Metal world graph

`MetalWorldContext` owns the persistent standalone execution graph for one
compiled articulation plus arbitrary dynamic, kinematic, and static scene
bodies. The ABI reserves articulation indices but v1 intentionally accepts one
selected articulation per environment.

## Public ownership

- `CompiledWorld` is an immutable validated snapshot. It exposes scene-body
  indices, canonical eligible pairs, configured capacities, conservative
  minimum capacities, and a content fingerprint.
- `MetalWorldBatch` contains environment-major articulation and scene state,
  control-step-major efforts, optional reset state, and optional kinematic
  targets. `submit()` snapshots every caller-owned span before returning.
- `MetalWorldContext` owns the device, queue, cached compute
  pipelines, immutable uploads, and a typed grow-only arena.
- `MetalWorldSubmission` is a move-only asynchronous ticket. `submit()` commits
  without waiting; `wait()` is the explicit host publication boundary.
- `MetalWorldResult` returns accepted state, fixed-shape observations,
  acceleration, raw per-step GPU status, optional fixed-capacity evidence, and
  one horizon-aggregated `MetalWorldStatus` per environment.

The aggregate status records exact capacity requirements, resident high-water
counts, manifold retention, maximum residuals, solver iterations, and the
first failing stable pair/constraint index. One failed environment rolls back
without preventing healthy environments in the same dispatch from publishing.

## Compiled task and policy execution

A robot model does not select a Metal shader. Every production run is compiled
from independently owned packages:

```text
RobotPack + ScenePack + SensorPack
        + TaskPack + RealityPack
        + TeacherPack + PolicyPack + RunProfile
        |
        v
stable semantic indices, tables, capacities, fingerprints
        |
        v
MetalWorld + LocomotionTask.metal + PolicyInference.metal
```

`TaskPack` owns action bindings, semantic groups, objectives, termination,
commands, terrain semantics and its operational contact-graph capacity.
`SensorPack` owns observation/history execution, `RealityPack` owns atomic
reset variation, and `TeacherPack` owns pre-actuation proposals. `compileRun`
resolves every authored body, joint, sensor and variation once, then fuses the
three executable programs into fixed Metal tables. GPU execution consumes only
those tables, counts, offsets and fingerprints.
Imported WorldPacks retain their exact asset ownership table during this
compilation, so articulated, rigid, kinematic, and static bodies cannot be
silently collapsed into one dynamics class. Persisted reality and external
sensor-program content hashes are part of the immutable run identity.

Imported runs persist those authorities independently as `TaskPack` v16,
`RobotActuatorPack` v2, `SensorProgramPack`, and `RealityProgramPack`. Runtime
construction requires every artifact and compiles it directly; there is no
combined execution wrapper and no authoring-time transfer of observation or
reset fields through `TaskPack`. `RealityProgramPack` v2 is the authority for
both world/physics variation and task-state reset semantics; `MRWorldPack`
supplies scene mechanics and assets but does not override that program.

`PolicyPack` owns dense actor weights, normalization, output transforms, clips,
and a monotonically increasing revision. `compilePolicyProgram` verifies its
observation/action contract against the compiled task and proves finite
accumulator bounds before publishing immutable GPU tables. The generic dense
kernel has no robot, joint, or network-width branches.

Every train/evaluate/deployment handle is now created from `CompiledRun`. The
compiled `RunProfile` is the only source for environment count, timestep,
substeps, solver iterations, streamed inverse-ABA selection, reset-band range,
and seed; constructor configuration cannot silently replace those semantics
after fingerprinting. Layout evidence publishes run, robot, sensor, reality,
teacher, world, task, observation, and action fingerprints. The executor
retains the immutable `CompiledRun`; its model, world and program references
cannot diverge after construction. Public evaluation consumes the
TaskPack-authored outcome schema rather than assuming a humanoid or projectile
task.

The sole public constructor is `mr_create_task_rollout(MRRunManifestC)`, and
Swift uses `MetalRoboRunManifest`. Bundled G1, ARDY-backed G1, Franka, imported
URDF and WorldPack sources all terminate at this boundary. Source selection is
finished before compilation and never enters the Metal hot loop.
The legacy Franka `Runtime` interface and its separate `Physics.metal` shader
no longer exist; benchmark, evaluation and deployment cannot select a second
physics implementation.

Locomotion worlds may also author independent dynamic scene bodies. The
generic sphere helper is a convenience builder for projectile and disturbance
scenes; the resulting bodies use the same broadphase, manifolds, islands, and
temporal-cone solve as every other rigid object. An authored scene-state flag
may preserve initial linear/angular velocity across transactional task reset,
which supports launched bodies without a scripted impulse or robot-specific
kernel. A launch step may also be compiled into that body state: the generic
task kernel holds the body at its authored pose with zero velocity, releases
it deterministically at the requested control step, and rearms it on a
transactional reset. Rollout inspection can compile without task termination
operators when a continuous post-failure trajectory is required; production
training retains the authored termination and reset contract. Final robot
configuration and packed scene-body state are optional
explicit readbacks for inspection and native presentation capture; training
keeps them disabled.

An event-driven projectile sequence may bind a semantic protected-body group.
Native task completion evaluates per-link clearance barriers from accepted
body and scene velocity, recognizes a clean miss, and terminates only when the
active projectile's solved contact pair contains a protected member. Ordinary
support contact therefore cannot become a false dodge failure.

The bundled Unitree G1 factory is mechanics plus independently authored task,
actuator, sensor, and reality programs. Imported floating-base URDF/SRDF and
WorldPack sources provide the same four artifact authorities to
`mr_create_task_rollout`; the public executor always retains one `CompiledRun`.
A new `.metal` extension is appropriate only for a new physics primitive,
sensor modality, task operator, or genuinely new typed actuator executor.

Swift owns rollout length, chunking, submission/wait boundaries, resets,
policy revision, and error reporting. `MetalWorldResidentState` keeps
articulation state, scene state, manifolds, warm starts, actuator history,
sensor history, episode state, and RNG state private across submissions.
Reset is one native transaction. Only actor/critic observations, transition
records, and explicit diagnostics cross the learning boundary.
An optional device-observation program composes rendering or perception into
the same command buffer after SensorPack proprioception and before generic policy
inference. MetalWorld selects the exact accepted-or-reset q and scene state
into its unused ping-pong destination, refreshes articulated kinematics, and
publishes one global body arena. The callback may encode work against borrowed
buffers but cannot commit, wait, or retain them. Visual state therefore stays
synchronized on reset without another capacity-sized simulator copy.
Temporal proprioception and direct device observations have independent
layouts: SensorPack history multiplies only the temporal frame, while a rendered
depth or tactile suffix is appended once with its own native history program.
The final rollout chunk appends a value-only policy evaluation for the
accepted post-step state to the same command buffer. Bootstrap values
therefore do not require a discarded physics step or another submission.
Every timeout transition likewise carries the critic value of its accepted
terminal state, evaluated before the next native reset, so GAE never
bootstraps from the following episode.

`TaskPack` and `PolicyPack` have deterministic, fingerprinted, transactional
binary artifacts. MLX is a batch learning backend in
`python/metalrobo/mlx_policy_learning.py`; it has no simulator or rollout
scheduler dependency and publishes actors through the canonical PolicyPack
writer. Its PPO minibatch compiles forward evaluation, backward evaluation,
global policy gradient clipping, and Adam into one graph with explicit model
and optimizer state capture. Swift reuses one preallocated rollout arena, lends
its arrays directly to the synchronous native artifact writer, and applies a
configurable timeout to the long-lived learner protocol. No per-chunk batch
list, flattened duplicate, or second full tensor allocation remains; the
writer reserves one exact-size serialized payload. Python memory-maps that
payload and asks the native library to verify its canonical content hash.
It neither copies the complete file into a Python `bytes` object nor hashes
each byte in the interpreter.

Dynamics retain body state at each centre of mass, but task semantics are link
semantics. Compilation bakes the COM-to-link reference for the root and every
contact-group reference body into the immutable task tables. Root height,
local velocity, yaw tracking, support slip, foot clearance, and terrain
sampling therefore agree with URDF link frames without a runtime name lookup.
The mechanical-power reward consumes the actual effort after actuator
limiting and the torque-speed envelope; it does not reconstruct a nominal
controller torque from post-step state. Unitree's authored command bands are
sampled independently for every deterministic episode. An exponent biases the
distribution toward easier bands while leaving the entire range observable.
Each transition publishes the sampled band; learner checkpoints contain only
model and optimizer state.

## Encoded graphs

Free motion remains available as a compatibility and dynamics oracle:

```text
prepare/reset/checkpoint
  -> [ABA -> transactional commit] x physicsSubsteps
  -> observation/status capture
```

`temporalCone` is the single production contact graph:

```text
actions/reset
  -> current-state implicit-drive refresh
  -> ABA candidate
  -> articulation forward kinematics
  -> articulation/free/static body projection
  -> free-body prediction
  -> parallel collider transform/AABB projection
  -> parallel precompiled-pair overlap flags
  -> analytic narrowphase
  -> persistent manifold refresh/merge/reduction
  -> canonical ConstraintIR emission
  -> current-microstep IR/Jacobian/operator evaluation
  -> mixed articulation/free-body islands
  -> coupled circular/elliptic-cone solve
  -> constrained integration
  -> per-environment failure latch
  -> transactional q/v/body/manifold commit
```

The complete horizon is encoded before one command-buffer commit. Normal
stepping performs no CPU count read, collision/solver fallback, resource
synchronization, or command-buffer wait between stages or control steps.
Contact counts remain in device status buffers until the explicit ticket wait.
The overlap queue is environment-major and has no former 65,536-pair limit;
the probe executes 66,049 eligible pairs in one environment and detects the
only overlap in the final pair slot. Accepted pairs are consumed in compiled
order, so manifold identity and capacity-failure precedence remain
deterministic without global append atomics.

The temporal cone solver rebuilds body transforms, contacts, manifolds, point
Jacobians, and response columns on every microstep. Eligible one-articulation
native task worlds stream active response columns directly from inverse ABA
into the contact solve, so they materialize neither a dense mass factor nor a
response arena. Capacity rows stay explicitly zero and the inverse operator
visits only the device-published active constraint count. The streamed path is
the production default after the scale-conditioned coupled point solve passed
a deterministic 1,024-environment qualification spanning 16 policy updates,
256 steps per update, and 4,194,304 transitions with no failed environment
steps. Multi-articulation,
rod, authored-generalized-constraint, quality-solver, and explicit
qualification runs retain the dense factor path. Normal and both tangential
directions are solved as one coupled 3D block, with the tangential pair
projected onto the exact circular or elliptical Coulomb cone.
The held control target is converted to effort from the current q/v before
every microstep; reusing a control-step-start PD effort across later microsteps
is not dynamically equivalent. Each microstep integrates immediately before
contact is refreshed. Sequential cone projection is an inner numerical
operation, not a second selectable world solver. Parallelism is across
environments, islands, and bounded work cohorts.

## Persistence and transactionality

Published q/v, scene bodies, and manifold caches are immutable inputs to a
control step. Candidate state uses independent ping-pong storage. The first
capacity, finite-value, factorization, or solver failure latches its typed
status. The final commit then:

- publishes every candidate for a healthy environment;
- restores that environment's control-step q/v, scene-body, and manifold
  checkpoint after a failure;
- writes zero acceleration for the failed control step;
- continues all already-encoded work for unrelated environments.

Persistent manifold identity includes the environment, compiled collider pair,
collider generations, and patch slot. Old local anchors are refreshed, broken
anchors are rejected, new witnesses are matched first by feature and then by
anchor proximity, and candidates reduce deterministically to four points.
The old tangent is transported onto the new plane with a least-aligned-axis
fallback. Warm-start impulses remain a coupled normal/tangent triplet.

## Current capacity and execution boundary

The production-shaped contact graph now includes:

- stable count/scan/scatter pair and island queues;
- Wave32 exact-cone 8/16/32-contact cohorts;
- deterministic 32-contact tiles, single-group spill through 256 contacts,
  and stable distributed reductions beyond 256;
- private immutable, persistent-state, and transient placement heaps in the
  standalone three-slot submission ring;
- exact capacity reporting and per-environment transactional rollback;
- native Wave32 workers that pull compact packets from an invocation-local
  cursor through indirect dispatch. Their fixed worker grid comes from the
  compiled Apple-device profile rather than a per-step scheduling decision.

The bundled G1 TaskPack uses a measured operational profile rather than the
all-self-pair topology envelope: 128 candidate/raw pairs, 32 manifolds, 64
constraint blocks, 192 rows, 1,024 mesh candidates, and corresponding
fixed-capacity endpoint/query/island tables. Endpoint-runtime and articulation
point-query counts are canonical, not independent tuning knobs: each must be
exactly twice the constraint-block capacity, and the host rejects a mismatched
pack before allocating or dispatching. A 1,024-environment,
204,800-step randomized run reached 88 candidate pairs, 28 raw contacts, 11
manifolds, 25 blocks, 75 rows, and 248 mesh candidates. Training has so far
raised the candidate and mesh high-water marks to 100 and 272. Physical
outputs were identical to the previous oversized profile, retained device
storage fell to 2,314,985,006 bytes, and any future overrun remains a typed
transactional failure. This is not an arbitrary one-gigabyte cap.

On the 24 GiB M4 Pro qualification host, the active-row inverse-ABA path
reached 10,063 environment control-steps/s at 12,288 environments and 10,265
at 16,384 in a 32/24-step physics sweep, with no failed GPU steps. The larger
batch retained 14.1 GB before visual and learner state, so 12,288 is the
production training point rather than the rollout-only maximum. A resumed
visual ball-dodge update at 12,288 x 24 samples completed 294,912 transitions,
32 PPO minibatches, and one motion-prior update at 5,532 end-to-end environment
steps/s with zero GPU failures and KL 0.00204. The prior 11,264-environment
qualification ran at 3,713 end-to-end environment steps/s.

The post-transition critic used for timeout bootstrapping stays in the same
command buffer but masks non-timeout environments before the first dense
multiply. On the matched 12,288 x 24 visual ball-dodge rollout this raised the
two-run mean from 5,692 to 5,801 environment control-steps/s and reduced GPU
time by 1.96%, with identical physical outcomes and zero GPU failures. A
1,200-step qualification produced finite critic values for all eight actual
timeouts and left all 9,592 non-timeout bootstrap fields zero.

The full-capacity ABA kernel stores each symmetric 6 x 6 articulated inertia
as its canonical 21-value lower triangle and reuses dead kinematic position
and angular-velocity scratch for the later acceleration sweep. This reduces
static threadgroup memory from 13,120 to 10,160 bytes on M4 Pro, crossing the
three-group 32 KiB residency boundary without reducing topology capacity or
precision. In a same-binary, metallib-only 12,288 x 24 visual ball-dodge A/B,
end-to-end throughput rose from 5,546 to 5,865 environment control-steps/s
(5.74 percent) and GPU time fell from 153,316 to 144,805 ms (5.55 percent),
with zero failed environment steps. The focused FP64 parity, deterministic
replay, transactional failure, body-wrench, damping, armature, and contact
checks all pass. Selecting one canonical FP32 triangle intentionally removes
the separately rounded upper-triangle evaluation, so long chaotic rollout
fingerprints are not claimed bit-identical to the prior redundant storage.

Forward ABA also has a schedule-driven SIMD32 implementation for branching
trees. `CompiledWorld` cooks and fingerprints the immutable level/parent/child
schedule once; submissions only upload it when the compiled world or immutable
arena changes. One lane owns each body in a frontier. Reverse body lanes emit
disjoint projected inertia and bias contributions, and parent lanes add
siblings in stable cooked order without floating-point atomics. The runtime
selects the ordered kernel for width-one chains and SIMD32 only when a body
frontier has useful parallel work. `MetalWorldConfig::preferParallelABA=false`
retains the serial kernel as a paired qualification oracle.

On an Apple M4, a same-binary 4,096-environment x 8-control-step G1 free-motion
pair measured 291.8 ms median GPU time for SIMD32 versus 398.4 ms for the
serial kernel, a 1.365x speedup. The width-4 SIMD32 result differed from the
serial FP32 oracle by at most `2.33e-10` in q, `2.98e-8` in v, and `5.06e-6`
in scaled acceleration; multi-step FP64 gates, bitwise same-path replay,
rollback, native-task parameterization, and dual-PSM multi-articulation probes
also pass. These are same-device internal measurements, not an external
simulator or hardware-dynamics accuracy claim.

Analytic/SAT paths cover the inexpensive primitive pairs. Exact cylinder
support, robust GJK with MPR/EPA fallback, cooked convex patches, static or
kinematic mesh BVH4 traversal, and direct cell-indexed static heightfields
cover remaining convex and surface pairs. Dynamic concave shapes remain
unsupported.

Hybrid CCD computes deterministic, capacity-bounded event intervals for
analytic, support-mapped, and convex-mesh paths. ABI v4 carries event cursors,
simultaneous-impact clusters, split budgets, zero-time replay limits, consumed
time, and first failing event keys. The current step clusters certified events
and uses speculative temporal cone solves to consume the complete microstep; literal repeated
TOI advance/solve/continue publication remains the next collision milestone.

## Native sensors and specialized physics

Physics, rendering, tactile, and geometric-sensor stages execute on native
Metal command queues. Their state and scratch buffers remain private to their
owning contexts. A stage may lend a compact output buffer to a downstream
native stage, but it does not borrow MLX's command encoder or allocate
simulator state as learning tensors. Artifact fingerprints are checked when a
WorldPack, TaskPack, PolicyPack, or sensor program is installed.

High-poly authored geometry remains compute-native. MetalRobo builds tight
fixed-size cluster bounds once from streamed geometry, culls clusters in
parallel per environment and active shutter band, and lets the flat triangle
kernel reject from compact triangle-to-cluster and cached visibility tables
before vertex fetch. Visible microtriangles use packed atomic depth/identity;
larger triangles enter the cooperative tile resolver.

### GPU-native geometric sensors

The native body-state stage derives one environment-major geometric body arena
from articulated `q`/`v` and standalone scene-body state. Sensor graphs reuse
the accepted world arena and skip duplicate materialization.
Articulated twists use forward tree recursion and are then published across
one fixed SIMD body cohort. This does not expand a body-by-DoF Jacobian merely
to recover rigid-body velocity.

`scene_raycast` casts shared or per-environment ray batches against the
compiled physics scene without host publication. The kernel intersects
spheres, boxes, capsules, cylinders, planes, convex faces, and cooked
quantized BVH4 meshes. Every valid hit returns metric distance, world-space
point and normal, plus stable shape, body, material, and geometric-feature
identities. Two-way collision masks, excluded bodies, one- or two-sided mesh
queries, and face-forward normals are array inputs or graph-fixed options.

`scene_raycast_pattern` is the mounted-sensor form. Its ray geometry is cooked
once in parent-body coordinates and shared across environments. One Metal
kernel reads each environment's current body pose, transforms the local ray,
self-filters the mounting body by default, and traverses the complete scene.
No environment-major world-ray tensor is constructed. Deterministic grid and
LiDAR pattern builders cover terrain scanners and range sensors; arbitrary
patterns can bind different rays to different bodies or to the world.

Large graph-static ray batches use a projected-shape specialization. One
linear Metal pass composes each shape transform and its verified conservative
radius once per environment; all rays then reuse that compact state and reject
distant shapes before exact intersection. Small batches retain direct
traversal because measurement shows that projection setup is not free. The
specialization is fixed by the compiled Apple-device profile, never by a
per-frame sensor or solver flag.

This is the common geometry primitive for range cameras, LiDAR, terrain
height scanners, visibility tests, occupancy observations, and planning
queries. It stays on the existing compute timeline and reuses the immutable
world buffers; it does not build a per-step acceleration structure.
Standalone `sensor_reference` presentation
uses compacted Metal BLASes and grouped motion-instance TLASes when authored
high-poly visibility warrants their build and encoder-transition cost.

An authored `MRWorldPack` can be compiled directly into the native executor.
Its `EngineModel` and `CookedTactileSystem` are the only physics/tactile
sources; there is no collision-derived authored scene or alternate pack path.
For tactile packs, native kernels pack final solver contacts and publish
metric depth, tangent motion, named summaries, and object-local labels. All
temporal tactile state is persistent simulator state. Packs without authored
tactile sensors allocate and dispatch no tactile work.

Optional per-DoF actuator profiles carry joint-side torque constant, current
limit, no-load speed, efficiency, backlash, and delay. Cooking derives stall
torque. Metal consumes immutable authored profiles, while native task reset
owns per-environment profile values, backlash target, and delay history. One
correlated profile may be selected at reset, but no per-frame calibration
branch exists in the solver loop. Missing measured profiles use neutral
execution records and do not imply calibration.

Focused native operators expose free-motion and contact-capable Franka and PSM
scenes. A fixed-capacity generalized-constraint operator consumes the cooked
multi-articulation program; dual PSM and heterogeneous dual-PSM-plus-G1 graphs
are executable without a CPU fallback. Its inverse-ABA path
executes forward/reverse body frontiers across SIMD32 and uses deterministic
parent-owned sibling reductions, sharing one factorization across each RHS
packet. The same primitive can select a GPU semismooth-Newton quality path
for scalar bounded ConstraintIR rows; it uses diagonally scaled natural maps,
normal-equation CG, safeguarded line search, and physical residual
certification. Exact contact-cone unification and direct ABA Hessian-vector
products remain open. The force/actuation ABA step still needs the same
frontier conversion; full multi-articulation collision/island composition
remains the next shared-world boundary.

The canonical fixed-root PSM is not an independent-jaw approximation. It owns
one unbounded generalized gear row enforcing equal-and-opposite jaw velocity,
and its hardware-facing command map evaluates the pinned JHU Large Needle
Driver 400006 `ActuatorToJointPosition` matrix before publishing collision-jaw
targets. Arm/tool position bounds and effort limits are likewise the pinned
JHU values. The dual-PSM composer replaces each local jaw row with its rebased
equivalent alongside the floating-base locks. Cable elasticity, backlash,
sterile-adapter compliance, and unit-specific calibration are not published by
that source and remain outside the qualified fidelity boundary.

The surgical DER path now uses a separate shared-ABI graph in the same
metallib. It resolves homogeneous rigid bindings into environment-major
attachment targets, runs the SIMD32 rod projection, records the accumulated
equal-and-opposite attachment impulse plus average force, and applies that
impulse at the dynamic needle anchor without floating-point atomics. The
public dual-PSM needle/thread factory derives its binding from the curved
needle's rear swage geometry and initializes the thread in world coordinates.
The standalone host and persistent `MetalWorld` both publish rod and rigid
output transactionally. Non-adjacent edges are radius-correct capsules:
closest witnesses, coincident normals, four-node inverse-mass response, and
contact refresh run inside every DER sweep on FP64 and Metal. After the
position solve, SIMD32 rebuilds the final contact shell and canonical pair
order compacts at most 64 load-bearing contacts. Eight alternating projected
Gauss-Seidel sweeps retain each tangent multiplier and project its accumulated
impulse onto the Coulomb disk capped by the normal impulse inferred from the
constraint-induced normal velocity change. Capacity overflow rejects the
transaction instead of dropping a contact. The surgical PDO preset uses its
conservative dynamic calibration for this single-coefficient self-friction
path. The versioned heterogeneous rod program validates and fingerprints the
coefficient. Resolved surface-spin torque, distinct static and dynamic
self-friction, and stronger coupled rod/rigid iterations remain open fidelity
boundaries.

`HeterogeneousWorld` is the owned compilation boundary above those executors.
It composes `EngineModel` instances transactionally, records the exact global
body indices represented by the environment-major scene-state tensor, owns
DER reset/binding sidecars, validates anchor agreement at the reset pose, and
fingerprints every topology, state and sidecar byte. The canonical surgical
factory produces two PSM articulations plus one dynamic compound needle and
one swage-bound rod. Static generalized rows can already run through the
multi-articulation Metal operator and the rod/needle graph consumes the
same bundle. Dynamic contact rows spanning more than one articulation are not
yet admitted into the shared `MetalWorld` contact graph. The CPU FP64
`MultiArticulatedContactProblem` now closes the reference semantics first:
analytic point Jacobians from any articulation are packed under global
velocity offsets, retained articulation-local factors apply every
`M^-1 J'` column, and the exact-cone quality solver handles self,
articulation-articulation and articulation-static contacts without a dense
global inverse. The Metal articulated operator now has a dedicated
kinematics-plus-point-Jacobian mode which skips mass assembly, factorization
and impulse response; it emits deterministic zero generalized payloads while
preserving point results bitwise against the full operator. The CPU operator
also appends one 6D maximal-coordinate block per dynamic scene body, applies
world-frame inverse mass/inertia directly, and treats static/kinematic point
velocity as prescribed. The dual-PSM/needle `HeterogeneousWorld` now enters
one exact-cone island through that path.
Model-owned unbounded generalized equality and gear rows are also reduced
exactly in the FP64 oracle: articulation-local inverse ABA constructs
`G M^-1 G'` and `G M^-1 J'`, a deterministic dense Cholesky factors only the
small equality Schur complement, and the projected contact operator solves in
the equality null space. Final equality impulses are reconstructed after the
cone solve and expose their achieved residual. The dual PSM probe therefore
keeps all twelve floating-base locks and both jaw gears active during the
needle contact solve.

The standalone Metal frontend now assembles those identical global rows on
device. In one command buffer it emits articulation-local point Jacobians,
packs dynamic scene-body 6D blocks, applies every articulation-local inverse
ABA response packet, appends generalized equality rows to the same RHS stream,
constructs their small Schur complement, projects contact response into the
equality null space, constructs the projected `J M^-1 J'`, solves the exact
circular cones, reconstructs equality impulses, and transactionally publishes
candidate velocities per environment. It owns
no CPU solver fallback and exposes failed point, inverse-mass, quality and
physical-operator stages separately. This first host boundary intentionally
uses owned shared buffers and one terminal wait for evidence extraction; it
is not yet the persistent private-buffer runtime. The remaining shared-world
work is manifold-endpoint scatter plus promotion of this encoder sequence
into persistent `MetalWorld`.

The inverse-ABA primitive also accepts the compiled task's environment-major
body and controller parameter streams. It scales each body's mass and inertia
by the same randomized physical parameter used by ABA and can factor the
implicit-drive operator `M + hD + h^2K` selected by the world pass. Identity
parameters preserve the original response bitwise; the focused GPU probe also
compares randomized effective responses against a materialized model and
requires deterministic agreement. This closes the physical-operator contract
needed before the matrix-free response path replaces production dense factors;
the production TemporalCone kernels still consume their existing Cholesky
factor until response construction and every contact/rod/typed-row consumer
are promoted together.

Three-axis translational loop/fixture blocks now enter this graph dynamically.
Each authored body-local point reuses the articulation point-Jacobian stream;
the three arbitrary world-space frame axes are dotted with those analytic
Jacobians on device and appended after immutable model-owned equality rows.
Semantic ConstraintIR records are environment-major, while topology and query
slots remain immutable across cloned environments. Final residual and
null-space certification read the same combined operator used by elimination,
so dynamic loop rows cannot be certified against stale static-row storage.
This tranche supports articulation-articulation and articulation-static
fixtures as well as articulation/free-body and free-body boundary loops.
Dynamic scene bodies contribute linear inverse mass and world-frame inverse
inertia to every equality RHS; static or kinematic scene endpoints own no
columns and instead shift the target by their prescribed point velocity.

Three-axis angular frame rows use the same query stream. For each articulated
endpoint, the cooker emits one base point plus three unit body-basis offsets.
The Metal row kernel reconstructs the angular Jacobian exactly from
`0.5 * sum(r_i x (J_i - J_base))`, then projects it onto the authored
world-space frame. Dynamic scene bodies map directly to their packed angular
velocity block; kinematic angular velocity shifts the row target. Point and
angular rows can be combined into a six-axis weld without finite
configuration differencing, dense body Jacobians, or another solver. This
activates multi-contact ABI v3 while leaving ConstraintIR ABI v2 unchanged.

The public angular-equality authoring path accepts a desired B-in-A relative
quaternion, forms the desired B orientation from the current A endpoint, and
maps the shortest relative rotation into the authored world-space frame with
the SO(3) logarithm. It supports articulated bodies, dynamic or kinematic
scene bodies, and static world endpoints. Antipodal quaternion
representations are identical, the exact-pi axis tie is canonical, and
invalid input leaves every previously accepted equality unchanged. The
resulting three scalar coordinates feed the same CPU/Metal spatial-row solve;
future fully device-resident constraint compilation can reuse this convention
without changing ConstraintIR v2.

`CompiledMetalMultiArticulatedContactProgram` snapshots the immutable model
and deterministic parallel-ABA frontier schedule once, so repeated submissions
do not recook tree topology. The dual-PSM/needle heterogeneous bundle uses this
path as a 34-DoF island and matches the FP64 contact solution within `7.1e-5`.
The Metal quality solver now certifies both its scaled natural map and a
normalized primal-cone, dual-cone and complementarity KKT residual. This
prevents a stiff operator's small projected-gradient step from making a zero
impulse look converged. A cold, ill-conditioned solve that cannot meet the
FP32 certificate fails transactionally; persistent manifold warm starts
converge and expose the achieved KKT value. On the constrained dual-PSM
needle scene, the Metal result matches the FP64 state, contact and reconstructed
equality impulse payload within `6.9e-8`; all fourteen base-lock and jaw-gear
rows close at `7.5e-9`.

The former Python/MLX physics extension and MLX-owned task frontends have been
removed. Simulator state, reset, reward, termination, observation construction,
and rollout scheduling have one owner: the native compiled-task executor.

### Apple-silicon training scale

The production G1 visual-dodge path has been qualified on an Apple M4 Pro with
Swift scheduling, native Metal physics/rendering/inference, and MLX used only
for batch learning. Environment count is not inferred from a fixed product
preset. Compilation accounts for the selected world, task, visual profile, and
the Metal device's recommended working set.

The canonical rollout artifact retains its 32-bit payload boundary. Darwin
file transfers are issued in bounded chunks, so a valid payload larger than
2 GiB is not passed to one oversized `write(2)` call. The learner memory-maps
the artifact and retains rollout-sized tables as NumPy views. Only the active
PPO minibatch is transferred into MLX; a 16,384-sample visual minibatch is about
69 MiB instead of a second device-resident copy of the complete actor table.
The masked-depth presentation preset also carries an explicit 4 GiB renderer
preflight budget rather than inheriting the conservative standalone-renderer
default.

The measured qualification ladder on the 24 GiB M4 Pro is:

| Environments | Steps | Samples | End-to-end rate | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,024 | 256 | 262,144 | 3,596 env-steps/s | pass |
| 2,048 | 64 | 131,072 | 3,685 env-steps/s | pass |
| 4,096 | 64 | 262,144 | 3,756 env-steps/s | pass |
| 6,144 | 43 | 264,192 | 3,610 env-steps/s | pass |
| 8,192 | 32 | 262,144 | 3,784 env-steps/s | pass after active-pair staging compaction |
| 9,408 | 28 | 263,424 | 3,759 env-steps/s | pass after topology cache compaction |
| 11,264 | 23 | 259,072 | 3,683 env-steps/s | pass |

The 4,096 qualification reported zero failed environment-steps, 10.43 GB of
retained buffer capacity, and 9.49 GB of transient private capacity. Those
figures describe allocator classes and must not be added as if every byte were
simultaneously resident. A longer 96-step rollout also passes, but its larger
publication artifact creates materially more memory pressure than the 64-step
qualification. Long campaigns therefore retain measured horizon and swap
behavior alongside raw environment count.

Before contact staging compaction, the 8,192 profile required 20.79 GB against
the device's 19.07 GB recommended working set. Diagnostics identified 7.06 GB
of pair raw-contact staging as the largest allocation. Narrowphase already
consumes a stable compact work queue bounded by the TaskPack's operational
pair capacity, but it wrote each result back into a scratch span sized for all
1,684 eligible G1 pairs. The compact path now stores one raw-contact span per
accepted work slot and retains the original compiled-pair index in the work
and count records. Canonical manifold and ConstraintIR ordering remains owned
by the later eligible-pair segmented scan.

For the bundled G1 profile this changes raw-contact scratch from 1,684 to 128
pair slots per environment. Finalized manifold header and point scratch use
the same compact active-pair slot while the eligible-pair scan continues to
own canonical output order. At 8,192 environments this second compaction
reduced retained capacity from 14.91 GB to 11.17 GB and transient private
capacity from 13.05 GB to 9.30 GB. The revision 116 to 117 qualification
published zero failed environment-steps without swap growth, throttling, or a
GPU restart.

Convex query caches are persistent semantic state, so they are not compacted
by transient overlap activity. The cooker instead assigns one stable cache
slot only to topology pairs that can execute convex or mesh queries. Analytic
pairs carry the invalid slot. Reset, narrowphase diagnostics, and transactional
publication consume that same compiled mapping. This removes unused cache
entries without changing warm-start identity or adding a hot-loop lookup.

The fully compacted path qualified 9,408 environments at 12.77 GB retained and
10.65 GB transient private capacity, then 11,264 environments at 15.28 GB and
12.76 GB respectively. The latter is a 19.7% environment increase over 9,408;
it completed revision 119 to 120 with zero failed environment-steps, no swap
growth, no throttling, and no GPU restart. A single-update qualification is a
capacity and execution gate, not proof of sustained memory stability, so long
campaigns still require checkpointed multi-update monitoring. The 11,264
profile crossed that boundary when the second update raised swap use from
0.37 GB to 2.14 GB. The run was stopped after preserving revision 121; 9,408
is the current sustained-training profile and 11,264 remains a qualified
single-update capacity ceiling rather than a production claim.

The G1 operational candidate-pair arena retains six Wave32 cohorts (192
pairs). A fresh-seed ball-dodge continuation measured an exact requirement of
145 pairs, beyond the former 144-pair arena. Rounding through the next Wave32
boundary and retaining one additional cohort avoids a one-slot empirical bump
without allocating the complete 1,684-pair topology envelope per environment.
Typed overflow and transactional rollback remain the authority if a future
trajectory exceeds this operational contract.

Ball-dodge learning uses four authored projectile-speed bands. Every band is
sampled from the first update, with a mild easy-band bias and no advancement
decision. Masked-depth corruption follows the sampled band. Every
1,000-control-step window reports projectile contacts, clean misses,
height/tilt failures, and mean tracking as exposure-normalized evidence. The
device-resident reducer cannot modify sampling or reject learning, and no
simulator progress state is stored in MLX checkpoints.

Visual policy rollouts keep physics and inference at the authored control
rate, but sample the camera at its independent sensor cadence. The generic
device tracker retains depth, identity, and validity in private buffers and
reuses them between exposures; no readback or copy is introduced. On the M4
Pro qualification host, a 4,096-environment, 16-step ball-dodge rollout rose
from 2,685 environment-steps/s at 50 Hz camera sampling to 4,005 at 15 Hz
with zero failed steps. This is a 49 percent end-to-end throughput increase,
not a physics-rate reduction, and cadence remains independent of submission
chunk size through a persistent observation sequence.

Canonical contact indexing still reserves two articulated point-query slots
per constraint, but fixed-scene endpoints and per-environment batch tails are
now marked inactive instead of masquerading as root-body queries. The generic
articulated operator zero-publishes those Jacobians without traversing the
joint tree. At 11,264 environments this raised the matched four-step rollout
from 4,551 to 4,634 environment-steps/s (1.8 percent); the rollout fingerprint
and reward stream were bit-identical.

### Balance recovery and get-up training

Bundled G1 recovery uses the generic compiled TaskPack path. Select
`--task disturbance-recovery` to fine-tune a standing actor through sampled
horizontal impulses, reset noise, mass/controller variation,
and action delay. `--initialize-actor-policy-pack` copies an existing
deployment actor exactly, creates a fresh critic and exploration head, and
therefore initializes a new task without pretending to resume missing
optimizer state. Ordinary dynamic balls provide additional physical evidence.

Select `--task supine-get-up` for Stage-I get-up discovery. This task uses a
fixed supine reset, ten-frame meaningful proprioceptive history, an asymmetric
critic with root/support state, full-body collision, no fall termination, and
generic pelvis/torso height, body-up, low-foot, bilateral-support, and standing
completion rewards. It intentionally starts a separate policy: standing
recovery and fallen get-up have different observation meanings and termination
contracts. Trajectory refinement begins only after Stage I records an actual
standing transition; a higher fallen-pose reward is not a trajectory. Learner
checkpoints must use the `.safetensors` suffix because MLX selects its
metadata-capable loader from that extension.

Select `--task developmental-recovery` to retain supine discovery unchanged
while inserting a distinct, physically executed tuck/brace region between the
floor and squat resets. This task has its own fingerprint and policy contract;
it cannot silently replace or resume a `supine-get-up` brain. The intermediate
pose changes reset state only. NumiSolver still applies gravity, collision,
contact and friction on every microstep, and task-authored typed outcomes expose
the individual recovery contributions used for progress analysis.

Select `--task adult-locomotion` for the next developmental transition. The
task preserves the 610-value developmental actor history and 29-action ABI,
converts the actor's three support-sense slots into velocity commands, and
retains support patches for contact-rich balance. Its standing-only reset is
followed by command-band expansion, deterministic actuator/reset variation,
action delay, native horizontal impulse, and clock stress. The adult clock
keeps band zero at the authored 10-second command horizon and 2-4-second
disturbance interval, then progressively compresses those windows to 45% and
50% respectively by the highest band. Impulse magnitude is unchanged, so the
policy must survive with less reaction time. The critic receives the command
and privileged support fields; deployment still requires matched held-out
physical progress and no regression in termination, tilt, height, or tracking
outcomes.

Select `--task ball-recovery` to train and evaluate against four ordinary
dynamic rigid spheres. The bundled app supplies their authored mechanics while
the run's RealityPack randomizes position, velocity, height, direction, and
launch step per environment and episode. They use the same broadphase,
manifolds, islands, and temporal-cone solve as every other scene body. This is
the physical disturbance distribution; native root-velocity impulses remain a
high-throughput complementary workload.

## Decisive probes

```sh
./build/bin/metalrobo_task_program_check
./build/bin/metalrobo_task_rollout \
  --metallib build/shaders/MetalRobo.metallib \
  --envs 32 --steps 48 --repeats 20 --chunk 8 \
  --scene terrain --native-policy
./build/bin/metalrobo_task_train \
  --metallib build/shaders/MetalRobo.metallib \
  --native-library build/lib/libmetalrobo.dylib \
  --mlx-python python/.venv/bin/python --python-root python \
  --initialize-policy unitree_g1_native_locomotion \
  --policy-pack /tmp/g1-initial.policypack \
  --updated-policy-pack /tmp/g1-policy.policypack \
  --deployment-policy-pack /tmp/g1-deployment.policypack \
  --rollout-pack /tmp/g1.rolloutpack \
  --learner-state /tmp/g1-learner.safetensors \
  --envs 1024 --steps 24 --chunk 8 --updates 100
./build/bin/metalrobo_metal_world_contact_probe
./build/bin/metalrobo_metal_world_probe

cd python
python3 probes/mlx_policy_learning_check.py \
  --library ../build/lib/libmetalrobo.dylib
```

The task-program check covers deterministic semantic compilation, imported
URDF mechanics, capacity contracts, transactionality, PolicyPack compatibility,
and TaskPack/PolicyPack artifact round trips. The Swift rollout executable
covers resident state, bounded native submissions, randomized actions,
compiled policy inference, scheduled resets, compact publication, memory
accounting, and both throughput solvers.

The contact probe covers a resting sphere/plane cache, greater than 99 percent
unchanged-frame retention, deterministic replay, a mixed Franka/1 kg cube
contact, exact isolated capacity rollback, and a 66,049-eligible-pair stream
beyond the former scan ceiling. The free-world probe covers Franka/G1 FP64
parity, paired SIMD32/serial G1 parity and throughput, topology-aware kernel
selection, asynchronous ownership, bitwise same-path replay, failure rollback,
grow-only reuse, and the 4,096-environment throughput gate. The
heterogeneous-world probe additionally executes the multi-articulation SIMD32
path.

The MLX learner check performs a real PPO update without importing simulator
state or scheduling a transition, then publishes a PolicyPack accepted by the
Swift/Metal executor.

The persistent learner releases each completed rollout batch and clears MLX's
inactive allocation cache at the update boundary after policy, optimizer, and
motion-prior state are evaluated and published. Active learner arrays remain
resident. This prevents capacity-sized cached Metal allocations from competing
with MetalWorld's private heaps in unified memory: on the M4 Pro, five resumed
8,192 x 24 visual ball-dodge updates held used swap at 529--561 MB, whereas the
uncleared worker grew from 0.50 GB to 6.69 GB over five updates. Update records
publish active, cached, released-cache, and peak MLX byte counts. Revision
checkpoints retain the stochastic PolicyPack and matching learner state for
resume, plus a separately named deterministic PolicyPack for physical
evaluation and explicit deployment selection.
