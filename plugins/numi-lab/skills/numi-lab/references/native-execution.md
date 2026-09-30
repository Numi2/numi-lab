# Apple-native execution and source ownership

For the Lab robotics/embodiment path, preserve this Apple-native ownership
model. Sibling runtimes retain their own contracts: NumiVivo precision-sensitive
chemistry can legitimately run native FP64 on CPU; reference solvers and Python
authoring tools must be identified as such. Do not force every suite tool into
CompiledRun, MLX, or GPU execution:

- The native `CompiledRun` boundary composes `RobotPack`, `ScenePack`,
  `SensorPack`, `TaskPack`, `RealityPack`, optional `TeacherPack`, exact
  `PolicyPack`, and `RunProfile` into stable indices, fixed-capacity tables,
  and fingerprints. A new robot is authored mechanics, semantic roles and
  capability data, not a new shader or host execution mode.
- A robot brain is a `PolicyPack` permanently bound to exact world, task,
  observation, and action fingerprints. Legacy v3 packs are migration inputs;
  newly published v4 packs must not be dimension-only or silently rebound.
- Teachers such as ARDY, GR00T, or demonstrations propose learning evidence.
  They never bypass Metal gravity, collision, contact, sensing, resets, or
  time and never become an alternate physics path.
- Metal owns persistent physics, contact, terrain, task operators, sensing,
  rendering, policy inference, simulator state, and counter-based randomness.
  Keep the control loop device-resident and free of per-environment host loops,
  per-step string lookup, and unnecessary readback.
- Swift owns rollout length, chunking, the asynchronous submission/wait
  boundary, timeouts, atomic resets, policy revision, and error publication. It
  reuses bounded rollout storage rather than accumulating per-chunk copies.
- MLX owns batch learning only. It consumes compact memory-mapped rollout
  artifacts and publishes the next fingerprinted `PolicyPack`; it does not own
  physics, simulator state, or rollout scheduling.
- Apple unified memory is shared capacity, not permission to duplicate data.
  Account for retained native heaps, transient private arenas, MLX active and
  cached allocations, publication buffers, and the device's recommended
  working set. Prefer borrowed buffers, fused encoders, and explicit lifetime
  boundaries.
- One environment control step is a transaction. Accepted state publishes;
  failed state rolls back with typed status while healthy environments
  continue. Chunk size and scheduling must not change random streams or replay.
- GPU submission is asynchronous. A ticket wait is the explicit host
  publication boundary; rendering or sensing callbacks may encode against
  borrowed buffers but must not independently commit, wait, or retain them.

Do not infer hardware execution from an Apple Silicon build, a CPU probe, or a
simulator result. Report the actual device, runtime path, memory behavior, GPU
status, and physical or replay evidence produced by the run.

## Infrastructure routing

Load only the owner documentation needed for the request from the runtime root,
then trace its live code path:

- CLI discovery, overlays, generated evidence, and installation:
  `docs/NUMI_CLI.md`.
- Robot, task, policy, artifact, or compiler architecture: `docs/WORLD_ENGINE.md`.
- Metal execution, submissions, private heaps, unified-memory scale, and native
  training: `docs/METAL_WORLD.md`.
- Rendering, RGB-D, device observations, and zero-readback perception:
  `docs/VISUAL_PLATFORM.md`.
- FP32/FP64 parity, contact correctness, transactionality, and solver evidence:
  `docs/NUMERICS.md`.
- Solver discovery, profiles, compatibility, and external overlays:
  `docs/NUMI_SOLVERS.md`.
- Tactile geometry, contact fields, and sensor bridge work:
  `docs/TACTILE_GEOMETRY_BRIDGE.md`.
- Foundation action proposers: `docs/FOUNDATION_POLICIES.md`.
- Motion-imagination providers and physical realization:
  `docs/MOTION_PROVIDERS.md`.
- PX4 X500 source, flight control, and evidence boundaries: `docs/PX4_X500.md`.

For cross-layer changes, preserve the ownership boundary: C++ compiles and
validates the world, Metal executes the hot loop, Swift schedules bounded
rollouts, and MLX learns from published batches. Change the lowest owning
layer that can express the requested capability.

## Freedom model

Use the smallest sufficient level, without asking the user to translate intent
into implementation details:

- Configure user or workspace preferences and profiles.
- Extend the lab with executable commands under `.numi/commands`.
- Modify the Numi source when physics, sensing, learning, or task behavior must
  change. New robots are mechanics plus authored packs plus a policy contract,
  not hard-coded CLI branches.

Workspace commands and instructions belong to the user. Preserve them during
runtime updates. Prefer transparent files and executable capabilities that a
future Codex model can inspect and improve.
