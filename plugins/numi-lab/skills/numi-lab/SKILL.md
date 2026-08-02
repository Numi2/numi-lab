---
name: numi-lab
description: Use when the user wants Codex to configure, operate, train, evaluate, simulate, or extend robots through the local Apple-native Numi Lab runtime.
---

# Numi Lab

Treat Codex as the roboticist and Numi Lab as the user-owned local laboratory.
Do not force requests into a fixed robotics schema or invent a second planner.

## Start from live truth

1. Run `numi doctor` when machine or installation readiness matters.
2. Run `numi context` before choosing a workflow. It is the current source for
   installed capabilities, overlays, paths, revision, and extension points.
3. Run `numi <capability> --help` before operating that capability.
4. Inspect the owning repository code when the request needs behavior that the
   installed commands do not already provide.

## Apple Silicon execution model

Use Numi Lab as one Apple-native system, not as a Python simulator wrapped by
Codex:

- The native compiler resolves robot mechanics, `WorldPack`, `TaskPack`, and
  `PolicyPack` artifacts into stable indices, fixed-capacity tables, and
  fingerprints. A new robot is authored data and mechanics, not a new shader
  or host execution mode.
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

Load only the owner documentation needed for the request, then trace its live
code path:

- Robot, task, policy, artifact, or compiler architecture: `docs/WORLD_ENGINE.md`.
- Metal execution, submissions, private heaps, unified-memory scale, and native
  training: `docs/METAL_WORLD.md`.
- Rendering, RGB-D, device observations, and zero-readback perception:
  `docs/VISUAL_PLATFORM.md`.
- FP32/FP64 parity, contact correctness, transactionality, and solver evidence:
  `docs/NUMERICS.md`.
- Tactile geometry, contact fields, and sensor bridge work:
  `docs/TACTILE_GEOMETRY_BRIDGE.md`.

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

For training or evaluation, return the exact revision, arguments, artifact
directory, policy/checkpoint paths, failed environment steps, throughput,
memory, replay/fingerprint evidence, and task-specific physical outcomes that
the run actually produced. Reward or test success alone is not physical proof.

Never replace an incumbent policy unless the relevant evaluation gate passes.
Simulator evidence is not hardware evidence. Simulation, authoring, and local
training may be autonomous; real hardware execution must obey the owner's
configured arming, limits, emergency stop, and approval policy.
