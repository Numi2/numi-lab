# Geometry-consistent tactile sensing

MetalRobo's canonical tactile signal is a dense, metric normal-penetration
map. Simulation computes it directly from known geometry. A physical
vision-based tactile sensor may learn to recover the same map from camera
frames. Policies consume the common representation, not simulated tactile
RGB.

This is the central representation proposed by
[Tacmap](https://arxiv.org/abs/2602.21625). The implementation is a native
MetalRobo subsystem rather than a port of the paper's Isaac Lab or MuJoCo
code.

## What is implemented

- Flat, spherical, and authored custom-atlas sensing surfaces.
- Cooked rest positions, local normals, tangent frames, atlas coordinates,
  validity, represented physical area, and per-sample maximum depth.
- A deterministic FP64 CPU reference.
- FP32 Metal queries for spheres, boxes, cylinders, capsules, closed convex
  geometry, and closed triangle meshes.
- The engine's shared, stackless BVH4 for mesh traversal.
- Persistent fixed-capacity Metal buffers and encoding into a caller-owned
  compute encoder.
- Solver-derived net force, torque, force-weighted center of pressure, and
  contact count.
- Area-weighted geometric centroid, contact area, maximum and mean depth,
  per-sample object identity, and temporal depth change.
- World compiler and `MRWorldPack` persistence.
- C++, C, Swift, and Python interfaces.
- An opt-in metric-map and 3D query debugger.
- A paired-data MLX camera-to-depth translator trainer for the physical side.
- A Franka two-fingertip contact and closed-loop stabilization evaluation.

It is not FEM, a soft-body elastomer, a tactile RGB renderer, or a complete
pressure/shear model. No real-hardware transfer is claimed by this repository.

## Contact model

The physics and tactile geometry share one deliberate compliant-shell model:

```text
object
  |
  v
S_s  undeformed outer sensing boundary and solver rest-offset surface
 |   bounded virtual elastomer volume
S_u  authored rigid backing collision surface
```

The backing collider remains an ordinary rigid shape. Its positive
`restOffset` is the shell thickness and its `contactOffset` is at least that
large. The tactile cooker rejects a sensor whose configured shell and backing
offset disagree. The material's bounded compliance allows a small, stable
compression of the solver's rest-offset surface while the rigid backing
prevents unbounded passage.

For sample rest position `p_s`, outward unit normal `n`, and maximum measurable
depth `d_max`, the runtime queries

```text
r(s) = p_s - s n,  0 <= s <= d_max
```

against only the sensor's cooked target shapes. If `p_s` is occupied by a
target, the first exit distance along the inward normal is the penetration
value. Otherwise the value is zero. Values are clamped to `d_max`; saturation
is explicit in the validity bits. Ties between simultaneous targets use the
stable lower shape index.

This differs from pretending that post-solve rigid overlap is an elastomer.
The measured depth is the compression of an authored virtual shell, and the
same rest-offset shell participates in contact dynamics. It also differs from
inferring force from the depth image: force and torque come only from solver
impulses divided by the explicitly supplied impulse interval.

## Architecture mapping

| Platform layer | Tactile integration |
| --- | --- |
| Episode authoring | `EpisodeTwin::tactileSensors` owns the physical sensor definition and target filters. |
| World compiler | Validates backing/contact semantics, cooks samples, and fingerprints the complete observation definition. |
| World assets | `WorldTemplate::tactileSystem` contains pointer-free GPU records; runtime atlas processing is forbidden. |
| World packs | The current pack serializes tactile data and authored visual data as one contract. |
| Collision/contact | Target geometry reuses cooked engine shapes and BVH4; wrench data is adapted from actual solved manifolds. |
| Physics stepping | The tactile encoder accepts body states and solver contacts from the same step and distinguishes observation `dt` from impulse `dt`. |
| Metal resources | Static geometry is shared; dense outputs/history are fixed-capacity environment-major private buffers. |
| RL observations | Depth, velocity, validity, identity, summaries, and status are directly borrowable `MTLBuffer` objects. |
| Checkpoints | One exact tactile JSON contract is stored beside the policy. |
| Replay | Frame index, reset mask, timestamp, and the existing world identity make the temporal contract explicit. |
| Debugging | Optional PGM, CSV, OBJ, and JSON export; no renderer or readback is required in headless training. |
| Real sensors | Paired calibration JSONL and an MLX translator produce the same metric map; simulation never calls the network. |

Tactile does not introduce a visual compatibility route. Authored visual
presentation remains V2-only. A missing V2 visual pack still fails through the
visual compiler; collision geometry is never promoted into a V1 presentation
scene and no fallback visual hash is synthesized.

## Cooked sensor definition

`TactileSensorAuthoring` identifies:

- the owning body and rigid backing shape;
- a local sensor pose;
- atlas dimensions and surface kind;
- maximum depth, active threshold, shell thickness, and query epsilon;
- update period;
- collision-filtered target shape IDs;
- samples for a custom atlas, or parameters for a flat/spherical factory.

Each `MRTactileSampleGPU` stores:

- rest position in the sensor frame;
- outward local normal;
- orthonormal local tangents;
- represented physical area in square metres;
- per-sample maximum depth;
- atlas coordinate, sensor ordinal, and validity.

Flat maps use cell-centred samples and exact cell area. Spherical maps sample
the authored angular patch, use a local normal at every point, and include the
spherical area Jacobian. Custom atlases permit irregular fingertips and
invalid image regions without flattening the physical query domain.

## Query backends

The selected production backend is
`MR_TACTILE_QUERY_METAL_ANALYTIC_BVH4`:

- analytical exits for sphere, box, and cylinder;
- a fixed 28-iteration signed-distance bisection for capsules;
- half-space clipping for closed convex geometry;
- parity containment plus stackless BVH4 traversal for closed meshes.

The capsule solve is the only bounded numerical approximation in the current
shape set. CPU and Metal execute the same decision rules. Meshes must be
closed because penetration depth requires an inside/outside decision.

Metal ray-query support is feature-detected and reported, but it is not the
default. The existing BVH4 shares geometry across environments, has bounded
storage and deterministic traversal, and covers hardware without Metal ray
queries. A hardware intersection-function backend remains a distinct enum so
it can be added and benchmarked without redefining normal penetration.

## Observation contract

The authoritative machine-readable contract is
`schemas/tactile_observation.schema.json`. Dense arrays are
environment-major; each sensor owns a contiguous row-major atlas range.
Depth remains FP32 and in metres through the native publication boundary.

| Channel | Meaning |
| --- | --- |
| `penetration_depth_m` | Normal compression in `[0, sampleMaximumDepth]`. |
| `depth_velocity_m_per_s` | Difference from the previous updated map divided by the elapsed observation interval. |
| `validity_bits` | Physical sample, active contact, saturation, and target-filter flags. |
| `object_shape_id` | Stable selected target shape, or `MR_INVALID_INDEX`. |
| `sensor_pose` / `timestamp_s` | Sensor-to-world pose and sampled time. |
| `net_force_n` | Sum of solver impulses on the backing shape divided by the solver impulse interval. |
| `net_torque_nm` | Moment of those forces about the sensor origin. |
| `center_of_pressure_sensor_m` | Solver-force-magnitude-weighted contact position in sensor coordinates. |
| `geometric_contact_centroid_m` | Area-weighted centroid of active depth samples. |
| `contact_area_m2` | Sum of represented physical sample areas. |
| `maximum_depth_m` / `mean_depth_m` | Saturation-aware local depth reductions. |

The center of pressure is a compact resultant-contact descriptor inspired by
[Beyond Binary](https://arxiv.org/abs/2605.28812). It is not a reconstruction
of arbitrary multi-contact pressure, and it is intentionally kept separate
from the geometric centroid.

## Device-resident execution

`MetalTactileContext::encode` borrows:

- environment-major body-state buffers;
- optional fixed-stride solver-contact and count buffers;
- an optional reset mask;
- a live caller-owned `MTLComputeCommandEncoder`.

It encodes sample, reduction, and history kernels without allocating,
committing, waiting, or reading back. `nativeBuffer` publishes depth,
velocity, validity, object IDs, summaries, and statuses directly to the native
tensor/MLX boundary. Host `observe` and `readback` are convenience and
diagnostic paths.

Debug hits are disabled by default. The choice is specialized when the Metal
pipelines are created, not carried as a per-frame flag. In headless mode the
dense hit, hit-history, and hit-readback allocations collapse to fixed dummy
bindings and the specialized kernels do not access them.

## Temporal and replay semantics

- `frameIndex` determines each sensor's update schedule.
- A skipped update retains depth, validity, and identity while reporting zero
  depth velocity.
- A reset mask clears temporal velocity for the reset environment.
- `observationTimestepSeconds` is the base simulation/control-step interval.
  For a sensor updated every `N` frames, depth velocity uses
  `(currentDepth - previousUpdatedDepth) / (N * observationTimestepSeconds)`.
- `contactImpulseTimestepSeconds` measures the interval represented by solver
  impulses. This is often one physics substep and must not be replaced with a
  decimated sensor interval.
- The complete cooked definition has a deterministic 64-bit fingerprint.
- Policy checkpoint loading may reject any fingerprint mismatch.

## Real-camera translation

`python/metalrobo/tactile_translator.py` implements a small MLX residual
encoder-decoder:

```text
registered tactile camera frame
              |
              v
sensor-specific MLX translator
              |
              v
metric normal-penetration atlas
```

Calibration rows conform to
`schemas/tactile_calibration_record.schema.json`. Every row names the sensor
and indenter asset, then records the sensor-relative pose,
indentation depth, timestamp, raw frame, target map, and optional validity and
wrench data. Relative paths resolve against the manifest.

The target map should be generated from the known indenter geometry and
measured rig pose using the same atlas query definition as simulation.
Training minimizes masked metric pixel MSE, with an optional spatial-gradient
term. Saved weights retain the one observation fingerprint needed to prevent
a policy or translator from consuming the wrong atlas.

The model uses operators selected to be Core ML friendly, but this milestone
does not claim a validated Core ML conversion. The saved translator contract
records `conversion_validated: false` until an exported model is tested.

## Extension boundary

The reviewed follow-on work changes what should sit above or beside the raw
map, not what the raw map means:

- [SimShear](https://arxiv.org/abs/2508.20561) motivates a future
  tangential-state channel and learned shear decoder.
- [TactSpace](https://arxiv.org/abs/2606.18959) motivates modality-specific
  encoders into a shared latent above canonical physical channels.
- Object-centric contact fields and predictive tactile models belong above
  the sensor-local contract.

These ideas remain roadmap layers, not fields in the raw tactile contract.
The next physical channel should be surface-relative tangential displacement
plus contact velocity and friction state, validated independently before any
learned dense shear reconstruction.

## Acceptance

There is one geometry check, one executable example, and one optional
performance tool:

- `metalrobo_tactile_check` compares the clear FP64 reference with Metal for
  flat and curved contacts, analytical and cooked geometry, transforms,
  saturation, temporal updates, solver wrench, and center of pressure.
- `metalrobo_franka_tactile_example` exercises the normal authored world-pack
  flow, actual contact evidence, native buffers, deterministic feedback, and
  optional visualization.
- `metalrobo_tactile_benchmark` is not a regression suite; it exists only for
  explicit performance measurements.

Run:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target \
  metalrobo_tactile_check \
  metalrobo_franka_tactile_example \
  metalrobo_tactile_benchmark
./build/bin/metalrobo_tactile_check
./build/bin/metalrobo_franka_tactile_example
./build/bin/metalrobo_franka_tactile_example \
  --debug-dir /tmp/metalrobo-tactile-debug
```

The standalone benchmark reports compile time, median/p95/min/max observation
time, tactile frames/s, samples/s, retained bytes, per-environment bytes, and
explicit diagnostic-readback time as one JSON object. It does not invent
stage timings that Metal has not counter-sampled.
The measured Apple M4 matrix is recorded in `docs/TACTILE_PERFORMANCE.md`.

```sh
./build/bin/metalrobo_tactile_benchmark \
  --environments 256 --sensors 2 --width 32 --height 32 \
  --warmup 3 --iterations 9
```

## Current limits

- Normal penetration only: no distributed shear, micro-slip, viscoelastic
  memory, elastic waves, optical appearance, or complete force field.
- The compliant shell is a calibrated rigid-contact engineering model, not a
  deformable continuum.
- Capsule exit uses bounded bisection.
- Closed triangle meshes only; mesh parity near non-manifold geometry is
  rejected at cook time rather than guessed.
- Current profiling exposes aggregate tactile geometry/reduction/history
  latency. Per-kernel GPU counter samples and complete physics-step
  enabled/disabled deltas remain a profiling milestone.
- Only the local Apple GPU used by the recorded run has been measured. Results
  must not be generalized to other Apple GPU generations without rerunning the
  benchmark.
- The Franka example evaluates a deterministic tactile feedback law; it is
  not evidence of a trained or physically transferred policy.
