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
- `MetalWorldContext` owns the device, queue, twenty-one cached compute
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

## Encoded graphs

Free motion remains available as a compatibility and dynamics oracle:

```text
prepare/reset/checkpoint
  -> [ABA -> transactional commit] x physicsSubsteps
  -> observation/status capture
```

`throughputTGS` and `throughputPGS` execute the contact graph:

```text
actions/reset
  -> ABA candidate
  -> articulation forward kinematics
  -> articulation/free/static body projection
  -> free-body prediction
  -> parallel collider transform/AABB projection
  -> parallel precompiled-pair overlap flags
  -> analytic narrowphase
  -> persistent manifold refresh/merge/reduction
  -> canonical ConstraintIR emission
  -> current-microstep IR/Jacobian/factor evaluation
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

Temporal TGS rebuilds body transforms, contacts, manifolds, point Jacobians,
and factor-backed response columns on every microstep. Normal and both
tangential directions are solved as one 3D block, with the tangential pair
projected onto the exact circular or elliptical Coulomb cone. PGS consumes the
same manifold, ConstraintIR, material, restitution, compliance, sign, and
response records.

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
- persistent MLX Wave32 workers that pull compact packets from an
  invocation-local cursor while standalone Metal consumes the same queue
  through indirect dispatch. The MLX resource bundle selects its fixed worker
  grid from a registry-ID/Apple-GPU-family tuning profile before lazy
  execution; Apple9/10 devices currently use 64 worker groups.

Analytic/SAT paths cover the inexpensive primitive pairs. Exact cylinder
support, robust GJK with MPR/EPA fallback, cooked convex patches, and static or
kinematic mesh BVH4 traversal cover remaining convex and mesh pairs.
Heightfields and dynamic concave shapes remain unsupported.

Hybrid CCD computes deterministic, capacity-bounded event intervals for
analytic, support-mapped, and convex-mesh paths. ABI v3 carries event cursors,
simultaneous-impact clusters, split budgets, zero-time replay limits, consumed
time, and first failing event keys. The current step clusters certified events
and uses speculative TGS to consume the complete microstep; literal repeated
TOI advance/solve/continue publication remains the next collision milestone.

## MLX active-encoder adapter

The Python package pins `mlx>=0.32,<0.33` and builds a nanobind custom
extension. The primitive:

- allocates outputs and temporaries through MLX;
- encodes ABA and transactional publication into MLX's active Metal command
  encoder;
- can append `sensor_fast` authored-mesh visibility and all synchronized visual
  modalities through `visual_observation`, writing directly into MLX-owned
  arrays on that same encoder;
- creates, commits, and waits on no command buffer;
- has no CPU fallback;
- rejects JVP, VJP, and `vmap` explicitly;
- carries environment batching as the leading array dimension.

The visual primitive receives current and previous environment-major
`MRBodyStateGPU` records from a physics stage. Tactile authored worlds publish
the final body arena already materialized for tactile sampling as
`StepOutput.body_states`; this adds neither a second kinematics pass nor a
second body-state allocation. The visual primitive accepts only an explicit
`PackedWorldFamily`, and its MRWorldPack hash must exactly match the
`MLXCompiledWorld` hash before graph construction. Matching tensor dimensions
cannot hide a mismatched authored world.

MLX renderers are compiled with `graph_only=True`. They retain visual scratch
needed by rasterization but no capacity-sized RGB, depth, identity, normal,
motion, or validity planes. `visual_observation` always publishes RGB, metric
depth, and validity. Segmentation, identities, normals, and motion are a
static named selection; omitted fields have zero spatial extent and allocate
no dense MLX storage. This selection is fixed when the graph is built and
does not add a per-frame solver flag.
The primitive retains the native renderer and sampled world-family resources
through lazy evaluation and Metal command completion, registers both body
arrays as MLX inputs, and publishes the arrays without routing through
renderer-owned images. Metal resources referenced indirectly by the scene
argument buffer live in a one-heap residency set committed once with the
renderer and attached to the current MLX command buffer. The compute-only tile
path is deliberate: MLX owns an active compute encoder, whereas Apple hardware
rasterization and mesh shaders require a render pass and therefore cannot be
inserted by ending or replacing MLX's encoder.

High-poly authored geometry remains compute-native on this path. MetalRobo
builds tight fixed-size cluster bounds once from streamed geometry, culls
clusters in parallel per environment and active shutter band, and lets the
flat triangle kernel reject from compact triangle-to-cluster and cached
visibility tables before vertex fetch. Visible microtriangles use packed
atomic depth/identity; larger triangles enter the cooperative tile resolver.
This retains one active MLX compute encoder while avoiding all-triangle
transform work for off-camera links, objects, and scanlines.

### GPU-native geometric sensors

`materialize_body_states` derives one environment-major geometric body arena
from articulated `q`/`v` and standalone scene-body state on MLX's active
encoder. When a world step already publishes `StepOutput.body_states`, sensor
graphs reuse that array and skip materialization.
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
world buffers; it does not build a per-step acceleration structure or cross
an MLX command-encoder boundary. Standalone `sensor_reference` presentation
uses compacted Metal BLASes and grouped motion-instance TLASes when authored
high-poly visibility warrants their build and encoder-transition cost.

`WorldState`, `SolverCache`, and `StepOutput` are explicit MLX PyTrees. The
pure `step()` API supports explicit MLX reset masks/state. `MLXRolloutCollector`
compiles policy inference, effort mapping, physics, reward, termination, and
reset into one lazy graph; rollout storage and GAE are MLX arrays.
`mx.async_eval` bounds rollout chunks, while blocking evaluation is restricted
to declared rollout/logging and optimizer/checkpoint boundaries.

An authored `MRWorldPack` can be compiled directly into the same primitive.
Its `EngineModel` and `CookedTactileSystem` are the only physics/tactile
sources; there is no collision-derived authored scene or alternate pack path.
For tactile packs, the primitive packs final solver contacts and publishes
metric depth, tangent motion, named summaries, and object-local labels on the
active encoder. All temporal tactile state is explicit. Packs without authored
tactile sensors allocate and dispatch no tactile work.

Optional per-DoF actuator profiles carry joint-side torque constant, current
limit, no-load speed, efficiency, backlash, and delay. Cooking derives stall
torque. Standalone Metal consumes immutable authored profiles; MLX carries
fixed per-environment profile values, backlash target, and delay history as
explicit state. One correlated profile may be selected at reset, but no
per-frame calibration branch exists in the solver loop. Missing measured
profiles use neutral execution records and do not imply calibration.

The active-encoder world primitive exposes free-motion and contact-capable
Franka, G1, and PSM scenes, including persistent Wave32 work pulling and
literal event-time CCD. A separate fixed-capacity generalized-constraint
primitive consumes the cooked multi-articulation program on the same active
encoder; dual PSM and heterogeneous dual-PSM-plus-G1 graphs are executable
without a secondary command buffer or CPU fallback. Its inverse-ABA path
executes forward/reverse body frontiers across SIMD32 and uses deterministic
parent-owned sibling reductions, sharing one factorization across each RHS
packet. The same primitive can select a GPU semismooth-Newton quality path
for scalar bounded ConstraintIR rows; it uses diagonally scaled natural maps,
normal-equation CG, safeguarded line search, and physical residual
certification. Exact contact-cone unification and direct ABA Hessian-vector
products remain open. The force/actuation ABA step still needs the same
frontier conversion; full multi-articulation collision/island composition
remains the next shared-world boundary.

The surgical DER path now uses a separate shared-ABI graph in the same
metallib. It resolves homogeneous rigid bindings into environment-major
attachment targets, runs the SIMD32 rod projection, records the accumulated
equal-and-opposite attachment impulse plus average force, and applies that
impulse at the dynamic needle anchor without floating-point atomics. The
public dual-PSM needle/thread factory derives its binding from the curved
needle's rear swage geometry and initializes the thread in world coordinates.
The standalone host currently submits this three-kernel graph and publishes
rod and rigid output together. Non-adjacent edges are now radius-correct
capsules: closest witnesses, coincident normals, four-node inverse-mass
response, and contact refresh run inside every DER sweep on FP64 and Metal.
The versioned heterogeneous rod program owns and fingerprints this policy.
Promotion into the MLX `WorldStepPrimitive`, thread-tool witness generation,
and strong coupled rod/rigid iterations remain open.

`HeterogeneousWorld` is the owned compilation boundary above those executors.
It composes `EngineModel` instances transactionally, records the exact global
body indices represented by the environment-major scene-state tensor, owns
DER reset/binding sidecars, validates anchor agreement at the reset pose, and
fingerprints every topology, state and sidecar byte. The canonical surgical
factory produces two PSM articulations plus one dynamic compound needle and
one swage-bound rod. Static generalized rows can already run through the
multi-articulation Metal/MLX primitive and the rod/needle graph consumes the
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
into the persistent `MetalWorld` and MLX active-encoder contexts.

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

The NumPy/ctypes Franka task remains available only as
`--backend ctypes-debug` for compatibility and oracle work. The CLI training
default is the MLX-native Franka joint-stabilization task.

## Decisive probes

```sh
./build/bin/metalrobo_metal_world_contact_probe
./build/bin/metalrobo_metal_world_probe

cd python
python3 setup.py build_ext --inplace
python3 probes/mlx_world_probe.py
python3 probes/mlx_multi_articulated_probe.py
```

The contact probe covers a resting sphere/plane cache, greater than 99 percent
unchanged-frame retention, deterministic replay, a mixed Franka/1 kg cube
contact, exact isolated capacity rollback, and a 66,049-eligible-pair stream
beyond the former scan ceiling. The free-world probe covers
Franka/G1 FP64 parity, asynchronous ownership, bitwise replay, failure
rollback, grow-only reuse, and the 4,096-environment throughput gate.

The MLX probe covers `mx.compile` for Franka and G1, 100 exact replays,
FP64 parity, single-environment NaN failure isolation, explicit autodiff
rejection, a bounded native rollout, and a real PPO optimizer update with zero
NumPy conversions in the step loop.
