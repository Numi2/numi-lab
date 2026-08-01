# Native runtime

Swift is the public scheduler. Metal owns execution and mutable simulation
state. Objective-C++ is a private bridge between them.

## Session ownership

`MetalSimulationSession` owns:

- the Metal device, persistent command queue, immutable pipeline cache, and
  synchronization objects;
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

The runtime schema declares every reachable Metal entry point and groups it by
feature. A compiled world and step configuration select the required groups;
initialization creates only missing pipelines and retains them in one cache
shared by every in-flight arena slot. Adding a task, sensor, policy, contact,
CCD, quality, or rod feature upgrades that cache monotonically. Arena slots
keep disjoint heaps and mutable buffers but do not duplicate the device,
library, command queue, or immutable pipeline objects.

The Swift learner boundary uses a three-slot rollout ring owned and allocated
by the native runtime. Each slot contains separate `storageModeShared` Metal
buffers for actor and critic observations, latents, behavior metadata, values,
bootstrap values, and native transitions. Fixed capacities are derived from
the compiled layout and update horizon. Swift receives an opaque lease rather
than an `MTLBuffer`; its generation prevents reuse while Swift or MLX retains
the slot.

Each physics command buffer blits its compact streams directly into the
pending lease before commit. The native submission owns the append token, and
advances the lease cursor only after world-state validation and publication
succeed. Failed validation leaves the cursor unchanged and permanently
invalidates that lease. The last chunk alone may publish terminal bootstrap
values and make a full lease sealable.

The current executor uses topology-derived private heaps and indirect dispatch
for validated GPU-owned counts. Resource slots and the pipeline inventory are
generated, but many pass encoders still bind their arguments individually on
the classic `MTLCommandBuffer` path. Metal 4 command allocators, generated
argument tables, queue residency sets, and the final explicit-barrier graph are
release gates, not current capability claims. Shared events remain reserved
for CPU or process boundaries.

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
metadata buffers for parent-frame pose, world-twist, and six-axis contact-
wrench sensors. One thread owns one environment/sensor history ring, so a reset
clears and seeds it without atomics or host reconstruction. A second boundary
advances the schedule after physics accepts the next state; rollout-chunk
boundaries therefore do not duplicate a sample. Rates use integer nanosecond
phase accumulators and whole-sample latency is selected from the retained ring.

Force/torque reads the same final-microstep solved contact constraints used by
TaskIR and the tactile force authority. It sums the force and moment applied to
the parent body about the authored sensor origin and rotates the result into
sensor-local axes. Before a requested reset, the SensorIR pass copies that
environment's schedule state, complete latency ring, compact output, and
metadata into topology-sized private checkpoint buffers. A rejected world or
contact transaction restores the journal byte-for-byte; successful and
ordinary non-reset transitions perform no checkpoint copy. The same schedule
also runs directly against a compiled world when no TaskIR program is present.

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

Native task kernels produce compact rollout streams and the same Metal command
buffer publishes them into the leased shared slot. Swift neither allocates the
Metal streams nor materializes, copies, or concatenates `[Float]` batches. In
leased mode, `MetalWorldResult` publishes status and diagnostics but allocates
zero duplicate actor, critic, latent, value, or transition elements. Ordinary
inspection calls retain those vectors explicitly. MLX arrays are constructed
with the underlying managed C API, retain the slot lease through an owning
finalizer payload, and consume those shared buffers without an input copy.

Before the lease blit, one cooperative Metal threadgroup validates every
compact float and transition. It emits a 64-byte generated-ABI status with the
first invalid stream/index, checked counts, runtime ABI, policy revision, task
fingerprint, and a unique submission token. After command completion the CPU
reads only this status before advancing the lease cursor; it does not scan or
materialize the payload. A malformed whole-session publication invalidates the
resident token rather than silently continuing from an unreported step.

Policy installation has two private native banks. The first installed policy
locks a topology fingerprint containing its identity, task contract, operator
shapes, activations, offsets, and arena size. A changed policy must carry a
strictly newer revision. Its complete header and arena are staged into the
inactive bank, and that bank becomes active only while encoding the next
submission; an already encoded command buffer retains the old Metal resources.
Returning to a retained fingerprint reuses its bank without another upload.

The learner still converts evaluated MLX parameters into host PolicyPack
arrays before native staging. Contiguous MLX weight flattening, a no-copy Metal
view, and a direct blit into the inactive bank remain optimization gates.

Dense native inference assigns one output neuron to one cooperative
SIMDgroup. Lanes accumulate strided features in FP32, use a deterministic SIMD
reduction, and lane zero applies bias and activation. Up to eight output
SIMDgroups share one threadgroup. This removes the former serial per-neuron
feature loop, but it is not yet the final shape-specialized matrix-tile,
fp16/bfloat16, convolutional, or recurrent PolicyIR backend.

MLX lazy evaluation occurs at deliberate minibatch, update, and checkpoint
boundaries. MLX does not borrow the physics encoder or decide per-step
evaluation boundaries.

## Reset and failure

The complete reset contract covers:

- articulation, scene, and rod state;
- manifolds, active constraints, and warm starts;
- actuator delay/backlash and controller state;
- sensor, tactile, observation, and recurrent histories;
- episode counters, curriculum state, and RNG stream identity.

The qualified native boundary currently checkpoints pre-reset articulation
q/v, scene bodies, manifolds, and SensorIR state. The same core ordering is
implemented for generalized warm starts, rod nodes/edges/witnesses, and convex
query caches, but those reset-failure cases still need focused owner evidence.
TaskIR now checkpoints per-environment episode/RNG identity, delay state,
contact metrics, histories, bias, randomization, and controller parameters.
The owner proves that a branch containing a rejected reset produces a
byte-identical next accepted transition to a branch that never attempted the
failure. Independent actuator/backlash state, tactile history, presentation
history, global curriculum scheduling, and future recurrent-policy state are
not yet one atomic publication. Whole-session reset therefore remains
unqualified.

If any generated count, offset, capacity, factor, finite check, or extension
status fails, the environment retains its previous committed state and emits a
typed status containing the stage, stable key, required capacity, and first
failing element for every state category that has joined the transaction.
Healthy environments in the same batch may still commit.

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
