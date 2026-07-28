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

The current production-shaped vertical slice removes the legacy 128-contact
island ceiling, but it is not the final throughput architecture:

- the configured compact bucket accepts up to 512 manifold-point constraints
  because the current articulated point-operator ABI exposes 1,024 query
  slots;
- oversized requirements report exact counts and roll back the affected
  environment; tiled spill storage is represented in the public capacity/status
  ABI but is not yet executed;
- `minimumCapacities()` reports conservative compiled worst cases, while an
  explicitly smaller runtime profile is legal and produces exact
  transactional overflow evidence rather than failing compilation;
- collider projection and eligible-pair overlap tests use flattened parallel
  queues. Narrowphase, manifold construction, island union, and solve
  currently use one deterministic thread per environment. This proves the
  device-resident transaction and mixed response semantics, not the final
  compacted SIMD32 throughput design;
- immutable and scratch buffers still use the checked shared arena. Private
  heaps, staged uploads, residency sets, indirect queues, and counter heaps are
  the next optimization after the numerical graph is stable.

Implemented analytic pair classes are sphere/sphere, sphere/plane,
capsule/plane, box/plane, cylinder/plane, sphere/capsule, capsule/capsule,
sphere/box, capsule/box, and SAT box/box. Box/box currently uses deterministic
SAT plus inside-vertex/support witnesses rather than full face clipping.
Non-plane cylinder pairs and general convex GJK/MPR/EPA are not implemented.

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

The active-encoder primitive currently exposes **free-motion ABA only** for
Franka and G1. It rejects non-empty scene/contact state instead of silently
falling back to ctypes. The standalone contact graph must be moved behind the
same encoder adapter before contact training can be called MLX-native.

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
