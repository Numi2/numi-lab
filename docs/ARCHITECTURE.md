# Architecture

This document defines the one allowed MetalRobo product architecture. The
generated [capability matrix](CAPABILITIES.md) records which parts are
qualified today. Code that bypasses this architecture is migration material,
not an alternative product path.

## Compiled model, mutable session

```text
WorldSource + TaskPack + PolicyPack + settings
                         |
                         v
                 SimulationCompiler
                         |
                         v
                 CompiledSimulation
  ModelIR | ActuatorIR | ConstraintIR | RodOperatorIR
  TaskIR  | SensorIR   | PolicyIR     | ExecutionPlan
                         |
                         v
               MetalSimulationSession
  queues | heaps | physical state | contacts | histories | RNG
                         |
              +----------+-----------+
              |                      |
              v                      v
        RolloutSession       DifferentiableSession
              |                      |
              +----------+-----------+
                         v
                 bounded buffer views
```

`CompiledSimulation` is immutable. It owns stable indices, topology-derived
capacities, layouts, schedules, pipelines, and fingerprints. A
`MetalSimulationSession` owns all mutable state and synchronization. No caller
may construct packed GPU offsets or mutate compiled topology.

## Ownership

| Layer | Owns | Must not own |
|---|---|---|
| Import/compiler | Parsing, defaults, semantic resolution, validation, capacities, fingerprints | Runtime state or command scheduling |
| Swift | Sessions, tickets, rollout chunks, timeouts, policy revisions, deployment, checkpoints | Physics equations or per-step tensor graphs |
| Objective-C++ bridge | Private translation between Swift contracts and Metal objects | Public robot/task APIs |
| Metal runtime | Physics, contacts, actuators, RNG, tasks, sensors, reset, inference, tapes | String lookup or robot-specific dispatch |
| MLX Swift | Networks, losses, optimizers, minibatches, learned checkpoints | Simulator state, physics encoder, reset, rollout scheduling |
| Python tools | Offline conversion, datasets, code generation, external comparison | Production execution or learning orchestration |

## Internal IRs

- `ModelIR` contains validated bodies, joints, inertials, collision geometry,
  materials, sites, and semantic groups.
- `ActuatorIR` is the canonical actuator graph used by both the FP64 oracle and
  Metal execution.
- `ConstraintIR` represents contacts, joint limits, equality constraints,
  tendons, gears, attachments, and friction blocks with variable row and
  endpoint ranges.
- `RodOperatorIR` contains topology, tangent coordinates, block bandwidth,
  factor storage, and material parameters for a connected rod component.
- `TaskIR` is a typed, phase-separated, fixed-shape operator graph.
- `SensorIR` defines native scheduling, latency, history, noise, reset, and
  observation/recorder bindings for every sensor.
- `PolicyIR` defines inference topology, normalization, action transforms,
  recurrent state, and weight layouts. Its topology fingerprint is immutable
  for a session; learned revisions alternate between two private Metal banks.
- `ExecutionPlan` is the generated resource and pass graph consumed by Metal.

IR types are private. Public authoring uses importers and validated packs.

## One native transaction

Every accepted control transition follows one transaction:

1. Snapshot or identify the last committed environment state.
2. Apply scheduled reset, command, randomization, and episode state.
3. Seed reset-only kinematics and SensorIR history, refresh TaskIR actor/critic
   views, then execute policy and actions.
4. Generate collision and constraint candidates.
5. Validate all counts, scans, offsets, and capacities.
6. Solve into uncommitted state with NumiSolver.
7. Integrate, complete tasks, sample the accepted state, update SensorIR-bound
   histories and terminal bootstrap views, and advance counters.
8. Validate status, atomically publish state, and blit compact learning
   outputs into the pending native rollout lease before command-buffer commit.

Overflow, invalid dispatch, factorization failure, or nonfinite output must keep
the last committed state. Core prepare passes checkpoint q/v, scene bodies,
manifolds, generalized warm starts, rod state and witnesses before applying a
reset to working state. Convex-query caches are journaled only for reset
environments. SensorIR likewise journals schedule state, history, compact
output, and metadata only for reset environments. Rejected transitions restore
those bytes; successful and ordinary non-reset steps pay no checkpoint-copy
bandwidth beyond the core state already required for rollback. Whole-session
atomicity for TaskIR, actuator/backlash, tactile, presentation, curriculum, and
future recurrent-policy state is still incomplete and must not be inferred
from the qualified physics/SensorIR boundary.

## Generated ABI

`schemas/runtime_abi.json` is the source of truth for the world resource table,
resource lifetimes, persistent input ownership, debug names, shared kernel
bindings, cross-language record fields/offsets/sizes/alignments, and the
reachable Metal pipeline inventory. Every pipeline entry declares its feature
group, host member, Metal function, minimum threadgroup geometry, and required
SIMD width. Code generation emits the C++/Metal declarations and assertions,
Swift layout metadata, and the pipeline table consumed by the executor.

`ExecutionPlan` selects pipeline groups from compiled world topology and the
task, sensor, policy, contact, CCD, quality, and rod features actually enabled
for a session. Initialization may add groups to the immutable cache, but it may
not compile an entry point absent from the generated table. All mutable arena
slots share the resulting device, library, command queue, and pipeline objects.

Numeric buffer slots, duplicated host/shader enums, and handwritten lifetime
switches are forbidden. Persisted ABI changes increment the ABI version;
ordinary internal refactors do not create version-suffixed types.

## Extension boundary

A native extension is justified only for a new physics primitive, actuator,
sensor modality, task operator, or policy operator. It declares:

- immutable parameters and persistent state;
- capacity and alignment rules;
- reset, serialization, and fingerprint behavior;
- forward execution and optional backward execution;
- validity and failure publication.

A new robot layout, semantic group, goal, reward, or observation is data and
does not justify a new shader.

## Explicit boundaries

- Visual Presentation V3 is the only visual authoring path.
- MuJoCo is an offline numerical comparator, never a runtime fallback.
- Rigid bodies, articulated bodies, and first-party rods are in scope.
- Dynamic concave geometry requires convex decomposition.
- Cloth, fluids, general deformables, MJCF plugins, and SDF dynamics are
  unsupported until explicitly added to the capability registry.
