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
  recurrent state, and weight layouts.
- `ExecutionPlan` is the generated resource and pass graph consumed by Metal.

IR types are private. Public authoring uses importers and validated packs.

## One native transaction

Every accepted control transition follows one transaction:

1. Snapshot or identify the last committed environment state.
2. Apply scheduled reset, command, randomization, and actuator inputs.
3. Generate collision and constraint candidates.
4. Validate all counts, scans, offsets, and capacities.
5. Solve into uncommitted state with NumiSolver.
6. Integrate and update native sensors, tasks, histories, and counters.
7. Validate status and finite results.
8. Atomically publish the new environment state and compact outputs.

Overflow, invalid dispatch, factorization failure, or nonfinite output keeps the
last committed state. Partially scattered constraints or half-reset sensor
histories are never observable.

## Generated ABI

`schemas/runtime_abi.json` is the source of truth for the world resource table,
resource lifetimes, persistent input ownership, debug names, shared kernel
bindings, and cross-language record fields/offsets/sizes/alignments. Code
generation emits the C++/Metal declarations and assertions plus Swift layout
metadata.

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
