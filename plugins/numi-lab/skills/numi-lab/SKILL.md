---
name: numi-lab
description: Use when the user wants Codex to configure, operate, profile, train, evaluate, simulate, or extend Apple-native robotics and coupled-physics workflows through the local Numi Lab runtime.
---

# Numi Lab

Treat Codex as the roboticist and Numi Lab as the user-owned local laboratory.
Do not force requests into a fixed robotics schema or invent a second planner.
Use this skill for work through Numi Lab or its `numi` CLI. Do not activate it
for generic robotics explanations or unrelated runtimes unless the user asks to
integrate or migrate them with Numi Lab.

## Start from live truth

1. Run `numi doctor` when machine or installation readiness matters.
2. Run `numi context` before choosing a workflow. It is the current source for
   installed capabilities, overlays, paths, revision, and extension points.
   Resolve Numi source and owner documentation relative to its reported
   `Runtime root`; resolve user overlays relative to its reported `Workspace`.
3. Run `numi robots list` or `numi robots inspect ROBOT_ID` before configuring
   a robot. Use its authored capabilities and semantic roles rather than
   assuming G1 joints, humanoid sensors, or locomotion outcomes.
4. Run `numi <capability> --help` before operating that capability.
5. For solver choice, run `numi solvers list` (filter with `--target` for an
   intended runtime), inspect the selected descriptor, and create or resolve a
   fingerprinted solver profile. Respect its domain, role, exact selector,
   targets, and evidence boundary; never treat every solver as an
   interchangeable `train` or `evaluate` backend.
6. Inspect the owning repository code when the request needs behavior that the
   installed commands do not already provide.

Discover missing inputs from context, catalogs, capability help, and existing
artifacts. Ask only when live discovery cannot resolve a required robot, task,
artifact, outcome, or hardware-arming choice; never guess paths, fingerprints,
physical results, or approval.

Keep the user-facing flow concise: lead with the outcome or blocker, summarize
large catalog/help/JSON output, and give the next safe recovery command. On a
failed run, inspect its typed failure and retained artifacts before retrying;
never duplicate an expensive workload merely because it stopped.

## Apple Silicon execution model

Use Numi Lab as one Apple-native system, not as a Python simulator wrapped by
Codex:

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

## Completion contract

For discovery commands, return the relevant version, resolved path, status, and
output or typed failure. For commands that execute or produce artifacts, return
the exact runtime revision and worktree state, arguments, artifact directory,
relevant runtime and artifact hashes, stdout/stderr or typed failure, and the
actual device/runtime used. For simulation, training, evaluation, and profiling,
also return failed environment steps, throughput, retained and peak memory,
replay/fingerprint evidence, available traces or counters, and task-specific
physical outcomes. State when a requested profiler gate or physical outcome was
unavailable. A build, test, reward, liveness check, or timeline-only trace is not
physical or detailed GPU performance proof.

Before a long Metal, training, evaluation, or profiling run, inspect active
workloads and existing artifacts. Do not duplicate a live run or contend for a
dedicated GPU; use isolated build/worktree paths and checkpointed execution
when the workload warrants them.

Retain every physically valid candidate and its measured outcome. Changing the
configured production policy is an explicit evidence-backed selection, not a
binary verdict that erases partial progress.
Simulator evidence is not hardware evidence. Simulation, authoring, and local
training may be autonomous. Before real hardware can move, stop and obtain the
owner approval required by the configured arming policy, and verify limits and
an emergency stop; never infer that authority from approval to simulate.
