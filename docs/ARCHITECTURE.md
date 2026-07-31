# MetalRobo architecture

MetalRobo has one production execution path. Robot mechanics and task policy
contracts are compiled on the host; persistent simulation, task evaluation,
and policy inference execute in native Metal; Swift owns rollout scheduling;
MLX owns only differentiable learning.

```text
WorldPack or imported URDF
        +
TaskPack
        +
PolicyPack
        |
        v
compiled stable indices, tables, capacities, and fingerprints
        |
        v
MetalWorldContext
  physics + contact + terrain + task + policy inference
        |
        v
compact rollout artifact
        |
        v
MLX learner -> next PolicyPack revision
```

No Python object owns simulator state, schedules a transition, or borrows an
MLX command encoder for physics. No external physics engine is linked or
called by the runtime. MuJoCo is used only by an explicit deployment
comparator against pinned upstream Unitree assets.

## Authored artifacts

### WorldPack

A WorldPack contains pointer-free mechanics: bodies, joints, inertias,
actuators, shapes, materials, sensors, collision filters, and capacity
requirements. The native URDF/SRDF importer compiles into the same model. A
bundled factory such as Unitree G1 is a preset, not a separate execution
architecture.

The canonical body translation and linear velocity are measured at center of
mass. Joint anchors are stored relative to body COMs. Floating roots use world
COM `xyz` plus quaternion `xyzw` in configuration and world COM linear plus
world angular velocity in generalized velocity.

### TaskPack

A TaskPack declares action-to-DoF bindings, observations and history,
semantic contact groups, reward operators, termination rules, reset and
randomization distributions, and terrain samples. Compilation resolves names
such as `left_ankle_pitch` or `foot_contact` once. The GPU consumes fixed
indices, ranges, and counts; it performs no string lookup and has no
per-robot shader branch.

`TaskRuntime.metal` executes compiled task operators. The bundled G1
task is data compiled through the same TaskProgram as an imported robot. A new
robot therefore requires mechanics and task data, not a new `.metal` file.
Custom Metal is reserved for a new physics primitive, sensor modality, or
task operator.

### PolicyPack

A PolicyPack seals actor and critic topology, normalization, stochastic action
state, action scaling, task compatibility, content hash, identity, and
revision. The training pack retains the Gaussian behavior actor and critic;
the deployment pack strips exploration and critic state while preserving the
same actor revision. Native inference rejects incompatible dimensions and
fingerprints before a rollout.

## Native execution

`MetalWorldContext` owns the command queue, pipeline cache, persistent private
buffers, scratch arena, and asynchronous submission tickets. Its resident
state includes articulation and scene state, contact manifolds and warm
starts, actuator delay/backlash, tactile history, observation history,
episode counters, and counter-based RNG streams.

One native transaction performs action application, randomized physical
inputs, physics substeps, collision and contact, terrain interaction,
observation construction, reward, termination, and reset publication. Reset
atomically replaces all episode-owned state. A failed environment restores
its prior state and reports a typed stage/status record without publishing a
partial transition.

Collision streams use deterministic count, scan, and canonical scatter.
Compiled capacities are derived from the model and task rather than guessed
from environment count. Every hot stage reports a retained high-water value
so a real overflow is distinguishable from a driver failure.

## Swift rollout ownership

The Swift executables own rollout length, submission chunking, completion and
error handling, policy revision consistency, rollout artifact publication,
and in-process MLX Swift learning. Chunking bounds command-buffer work while
the native context and its private state remain resident. An empty reset stream is
passed as a null pointer and zero count; an authored reset stream must contain
exactly `control_steps * environments` entries.

`metalrobo_train` keeps one simulator context and one in-process MLX Swift
learner. It appends native chunks directly into a preallocated rollout arena,
writes one fingerprinted rollout artifact, performs PPO at the declared batch
boundary, and installs the resulting weights directly into the resident Metal
world. Python is absent from the runtime. The safetensors sidecar atomically
checkpoints model and bias-corrected Adam state with the native task-wide
curriculum level; resume restores that compact state before the first Metal
submission and begins a new synchronized evaluation window.

## MLX learning boundary

MLX receives compact actor observations, critic observations, sampled latent
actions, log probabilities, values, transition records, and bootstrap values.
It computes GAE and compiled PPO forward/backward/gradient-clip/Adam updates.
It does not receive manifold caches, scene state, actuator state, tactile
history, or mutable simulator buffers.

Offline visual-tactile representation and imitation learning also remain in
MLX because they operate on datasets and model parameters. Any temporal state
used by those models is learner state, not simulator state.

## Sensors and presentation

Visual Presentation V3 consumes authoritative native pose buffers and authored
V2 asset/environment packs. It never reconstructs visuals from collision
geometry. Tactile simulation consumes geometry plus exact solver contact
bases and impulses, producing metric deformation, motion, wrench, center of
pressure, and validity evidence. TaskPack contact-wrench observations use the
same retained contact basis in a named body frame.

## Memory and synchronization

Persistent simulator internals use Metal private buffers. Only compact
rollout and artifact boundaries become host-visible. Allocation is preflighted
against `MTLDevice.maxBufferLength` and the recommended working set while
reserving unified-memory headroom for MLX parameters, rollout publication,
macOS, and optional presentation.

Fingerprints occur at WorldPack, TaskPack, PolicyPack, rollout, replay, cache,
and deployment boundaries. They are not recomputed per frame.

## Evidence boundary

Focused correctness executables own their subsystem invariants. Long native
rollouts prove runtime stability and high-water behavior. A policy is not a
deployment result until it independently beats the default-pose comparator in
the pinned official Unitree MuJoCo model. Build success, learner loss, and
in-simulator reward do not substitute for that evidence.

Detailed subsystem contracts live in [WORLD_ENGINE](WORLD_ENGINE.md),
[METAL_WORLD](METAL_WORLD.md),
[TACTILE_GEOMETRY_BRIDGE](TACTILE_GEOMETRY_BRIDGE.md), and
[VISUAL_PLATFORM](VISUAL_PLATFORM.md).
