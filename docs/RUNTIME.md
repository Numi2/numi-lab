# Native runtime

Swift is the public scheduler. Metal owns execution and mutable simulation
state. Objective-C++ is a private bridge between them.

## Session ownership

`MetalSimulationSession` owns:

- the Metal device, queues, pipeline cache, command allocators, and events;
- immutable, persistent, and transient private heaps;
- articulation, rigid-body, and rod state;
- contact manifolds, active sets, and warm starts;
- actuator delay/backlash and controller state;
- sensor, tactile, recurrent-policy, and observation histories;
- episode counters and counter-based RNG streams;
- transaction checkpoints and structured failure status.

Simulator internals are not ordinary MLX tensors and are never reconstructed
by selecting or zeroing a world-shaped tensor tree.

## Submission ring

The production session uses three submission slots. Each slot owns all writable
resources that can overlap another in-flight submission, including transient
arenas, status, compact outputs, and command allocation. Immutable resources
and explicitly synchronized persistent state may be shared.

The execution plan uses generated argument tables, queue residency sets,
topology-derived private heaps, indirect dispatch for validated GPU counts,
and explicit barriers. Same-device ordering uses ordinary Metal events;
shared events are reserved for CPU or process boundaries.

The chunk length is part of the recorded rollout configuration. Scheduling may
change submission boundaries but cannot change physics timestep, RNG keys,
policy revision, or reset semantics.

## Tickets and views

Submitting work returns a move-only `SimulationTicket`. Waiting is explicit and
may occur once per rollout chunk rather than once per control transition.

Compact output is exposed through lifetime-scoped borrowed views:

- actor and critic observations;
- actions and behavior-policy metadata;
- rewards, termination, and truncation;
- reset masks and recurrent initial state;
- policy revision, validity, and selected metrics;
- requested sensor/recorder streams;
- selected gradient buffers for differentiable sessions.

A view retains its ring slot. The slot cannot be reused until all Swift and MLX
consumers release it.

## Current SensorIR execution boundary

The canonical MetalWorld session now owns persistent schedule, history, output,
and metadata buffers for pre-control parent-frame pose sensors. One thread owns
one environment/sensor history ring, so a reset clears its schedule and history
without atomics or host reconstruction. Rates use integer nanosecond phase
accumulators; whole-sample latency is selected from the retained ring. Compact
latest values and timestamp/age/validity metadata cross the inspection boundary
only when explicitly requested.

Presentation and tactile passes have not yet been folded into this scheduler.
They remain native but are rejected by this execution gate instead of being
silently skipped or routed through Python.

## MLX boundary

Native Metal writes compact rollouts into preallocated shared buffers. Swift
wraps those buffers using MLX Swift's managed raw-pointer initializer and
retains the underlying `MTLBuffer` through the finalizer. There is no
intermediate `[Float]` concatenation.

For weight installation, the learner evaluates one contiguous MLX array,
obtains its no-copy Metal view, and blits into an inactive private policy bank.
The array remains alive until blit completion, then the session atomically
publishes the new revision between chunks.

MLX lazy evaluation occurs at deliberate minibatch, update, and checkpoint
boundaries. MLX does not borrow the physics encoder or decide per-step
evaluation boundaries.

## Reset and failure

Reset is a native transaction covering:

- articulation, scene, and rod state;
- manifolds, active constraints, and warm starts;
- actuator delay/backlash and controller state;
- sensor, tactile, observation, and recurrent histories;
- episode counters, curriculum state, and RNG stream identity.

If any generated count, offset, capacity, factor, finite check, or extension
status fails, the environment retains its previous committed state and emits a
typed status containing the stage, stable key, required capacity, and first
failing element. Healthy environments in the same batch may still commit.

## Memory accounting

There is no arbitrary global memory ceiling. Compilation and session creation
report:

- fixed bytes per compiled simulation and session;
- bytes per environment;
- bytes per candidate/contact/constraint row;
- bytes per sensor and retained history sample;
- bytes per policy state and rollout slot;
- bytes per differentiation tape checkpoint;
- transient high-water and peak aliased bytes.

The device's recommended working-set size remains a hardware failure boundary,
not a product-level 1 GB policy.

## Determinism

Same hardware, OS, build, execution plan, chunking, and policy revision target
bitwise replay. Cross-generation Apple GPU replay uses declared tolerances.
Benchmark and replay manifests record every relevant execution choice.
