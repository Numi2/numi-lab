# Persistent Metal world graph

`MetalWorldContext` is the first canonical environment-major execution graph.
It compiles one immutable `EngineModel` snapshot and advances many control
steps in one Metal command buffer. ABI v1 is intentionally named
`freeMotionABA`; collision/contact solver modes exist in the public enum but
fail closed until their graph stages are executable.

## Public ownership model

- `CompiledWorld` is a validated, immutable model snapshot with a stable
  64-bit content fingerprint. Failed compilation leaves the previous snapshot
  unchanged.
- `MetalWorldBatch` describes initial q/v, a control-step-major effort
  trajectory, and optional per-step reset masks plus one reset state per
  environment. `submit()` copies every span before it returns.
- `MetalWorldContext` owns the Metal device, queue, metallib, five persistent
  pipelines, immutable model cache, and one grow-only shared-memory arena.
- `MetalWorldSubmission` owns an already committed command buffer. It can
  outlive its originating context, is move-only, and drains safely when
  discarded.
- `MetalWorldResult` publishes final q/v, every q+v observation, every final
  substep acceleration, and one typed status per environment/control step.

One context deliberately admits one in-flight submission because its arena is
reused. Independent contexts provide queue overlap without aliasing state.

## Encoded graph

For each control step, the host encodes the following passes without waiting
or reading an intermediate counter:

```text
prepare/reset + checkpoint + effort slice
  -> [ABA -> transactional commit] x physicsSubsteps
  -> observation/acceleration/status capture
```

The source and destination q/v buffers alternate after every substep. ABA
writes only candidate q/v/acceleration. Commit is the sole publisher:

- on success, candidate q/v becomes the next accepted ping-pong state;
- on the first failure, its typed ABA error, substep, and failing index latch;
- that failure and every remaining encoded substep restore the immutable
  control-step checkpoint;
- capture writes zero acceleration for the failed control step;
- the next control step starts normally from the rolled-back state.

A reset is applied before the checkpoint is taken. It is therefore retained
if dynamics later fails in that control step. Failures are isolated per
environment; healthy environments in the same dispatch continue.

## Persistent arena

ABI v1 uses 27 buffers:

| Class | Buffers |
| --- | --- |
| Immutable/runtime model | world, articulations, joints, DoFs, bodies |
| Dispatch | ABA dispatch, Metal-world dispatch |
| Accepted state | q/v A, q/v B |
| Transaction state | checkpoint q/v, candidate q/v/acceleration |
| Controls | complete effort trajectory, one working effort slice |
| Optional reset | masks, reset q/v |
| Status | ABA, current environment, public step stream |
| Output | q+v observations, acceleration trajectory |
| Typed empty binding | body-wrench placeholder |

Every logical byte count is checked for `size_t`, `NSUInteger`, shader
32-bit element addressing, `MTLDevice.maxBufferLength`, and the recommended
working set. Each buffer grows geometrically and never shrinks. Immutable
topology streams upload only when the compiled fingerprint changes; timestep
is a per-submission runtime-world field.

## Determinism and evidence

The two ABA pipelines are numerical-identical capacity buckets. Franka and
PSM use a 12-body/16-v/17-q bucket to avoid reserving G1-sized threadgroup
scratch; G1 uses the 32-body/40-v/41-q bucket. Capacity selection is derived
from the immutable articulation and never changes model semantics.

`metalrobo_metal_world_probe` exercises:

- multi-step Franka FP64/Metal parity with resets and three physics substeps;
- bitwise replay on the same device/build;
- asynchronous input snapshot ownership and ticket moves;
- the one-in-flight busy gate and context/ticket lifetime;
- transactional host rejection;
- typed GPU factorization failure and whole-control-step rollback;
- grow-only arena reuse and immutable-model upload caching;
- the 4,096-environment free-space throughput gate.

The local Apple M4 result is recorded in [VALIDATION](VALIDATION.md).

## Next graph insertion

The next production tranche is not another wrapper. It extends this same
submission with:

```text
accepted q/v
  -> articulated body/collider projection
  -> segmented deterministic broadphase
  -> analytic narrowphase
  -> persistent manifold refresh/reduction
  -> evaluated ConstraintIR
  -> articulated/free-body island solve
  -> transactional constrained integration
```

The existing `mr_collide_baseline` is a one-environment correctness kernel,
and `mr_solve_contact_constraints` currently mutates maximal-coordinate rigid
bodies. Neither is silently called by `MetalWorldContext`; doing so would
misrepresent articulated contact semantics. The composed graph must consume
the same evaluated material/timestep rules and articulated inverse-mass
operator as the CPU quality path.

Rewards/resets generated from device observations and MLX-native input/output
arrays follow that contact composition. Until then, the current ticket reads
the final rollout into owned host vectors at `wait()`, so “device resident”
means no host synchronization inside the submitted horizon, not zero-copy
learner interoperability.
