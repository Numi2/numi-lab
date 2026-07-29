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
- creates, commits, and waits on no command buffer;
- has no CPU fallback;
- rejects JVP, VJP, and `vmap` explicitly;
- carries environment batching as the leading array dimension.

`WorldState`, `SolverCache`, and `StepOutput` are explicit MLX PyTrees. The
pure `step()` API supports explicit MLX reset masks/state. `MLXRolloutCollector`
compiles policy inference, effort mapping, physics, reward, termination, and
reset into one lazy graph; rollout storage and GAE are MLX arrays.
`mx.async_eval` bounds rollout chunks, while blocking evaluation is restricted
to declared rollout/logging and optimizer/checkpoint boundaries.

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
rod and rigid output together. Promotion into the MLX `WorldStepPrimitive`,
thread self/tool contact, and strong coupled rod/rigid iterations remain open.

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
Angular weld/orientation rows remain explicit follow-on work.

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
