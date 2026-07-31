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
- sensor, tactile, and observation histories; recurrent-policy state joins
  this ownership boundary when recurrent PolicyIR operators land;
- episode counters and counter-based RNG streams;
- transaction checkpoints and structured failure status.

Simulator internals are not ordinary MLX tensors and are never reconstructed
by selecting or zeroing a world-shaped tensor tree.

## Submission and rollout rings

The core Metal context supports a configurable submission ring. The current
resident C API session deliberately serializes one in-flight physics submission
because mutable state continuation has not yet been split from its arena slot.
This is a remaining executor migration item, not a claim of three overlapping
physics command buffers.

The Swift learner boundary does use a three-slot rollout ring. Each slot owns
separate page-aligned `storageModeShared` Metal buffers for actor and critic
observations, latents, behavior metadata, values, bootstrap values, and native
transitions. Fixed capacities are derived from the compiled layout and update
horizon. A generation-checked lease prevents reuse while Swift or MLX retains
the slot.

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

The canonical MetalWorld session owns persistent schedule, history, output, and
metadata buffers for parent-frame pose sensors. One thread owns one
environment/sensor history ring, so a reset clears and seeds it without atomics
or host reconstruction. A second boundary advances the schedule after physics
accepts the next state; rollout-chunk boundaries therefore do not duplicate a
sample. Rates use integer nanosecond phase accumulators and whole-sample latency
is selected from the retained ring.

Compiled TaskIR operators consume named SensorIR values and validity bits
directly on-device. Reset refresh fills every actor/critic history slot before
policy inference. Accepted-state refresh writes the shifted history tail and
republishes final actor and critic views before terminal policy/value inference.
Compact latest values and timestamp/age/validity metadata cross the inspection
boundary only when explicitly requested.

Presentation and tactile passes have not yet been folded into this scheduler.
They remain native but are rejected by this execution gate instead of being
silently skipped or routed through Python.

## MLX boundary

Native task kernels produce compact rollout streams. The current publication
path copies each completed compact stream once from the native result boundary
into its preallocated shared Metal ring slot. Swift does not materialize or
concatenate `[Float]` batches. MLX arrays are constructed with the underlying
managed C API, retain the slot lease through an owning finalizer payload, and
therefore consume the shared buffers without another input copy. Publishing
directly into the leased ring from the native command buffer remains an
explicit optimization gate.

Policy installation currently serializes and uploads the compiled PolicyPack
between update chunks. Contiguous MLX weight flattening, inactive private
weight banks, a no-copy Metal view, and an atomic blit-driven bank swap remain
to be implemented.

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
