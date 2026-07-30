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
- Instantaneous surface-relative tangential velocity and bounded
  contact-relative target-anchor displacement in every cooked sample frame.
- Solver-derived normal/tangential impulse and friction evidence, reduced to
  RMS tangential speed, maximum motion, and friction-utilization summaries.
- World compiler and `MRWorldPack` persistence.
- C++, C, Swift, and Python interfaces.
- Direct authored-pack execution in `WorldStepPrimitive`, with explicit MLX
  tactile history and no intermediate command buffer or host bridge.
- Fixed-capacity masked object-local contact labels for estimator training.
- An opt-in metric-map and 3D query debugger.
- A paired-data MLX camera-to-depth translator trainer for the physical side.
- Optional MLX shared-latent, reconstruction, object-field, and tactile
  dynamics models above the canonical observation.
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
| RL observations | Depth, depth velocity, tangent motion, validity, identity, named summaries, and status are directly borrowable `MTLBuffer` objects or named MLX arrays. |
| Checkpoints | One exact tactile JSON contract is stored beside the policy. |
| Replay | Frame index, reset mask, timestamp, and the existing world identity make the temporal contract explicit. |
| Debugging | Optional PGM, CSV, OBJ, and JSON export; no renderer or readback is required in headless training. |
| Real sensors | Paired calibration JSONL and an MLX translator produce the same metric map; simulation never calls the network. |
| Learned interface | Modality-specific MLX stems publish a 64-value latent with explicit presence, confidence, and recurrent state. |

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
| `tangential_displacement_u_m` / `v_m` | Bounded target-anchor motion in the cooked sample tangent frame. This is a kinematic proxy, not elastomer strain. |
| `surface_velocity_u_m_per_s` / `v_m_per_s` | Instantaneous target-minus-sensor point velocity projected onto the cooked tangents. |
| `sensor_pose` / `timestamp_s` | Sensor-to-world pose and sampled time. |
| `net_force_n` | Sum of solver impulses on the backing shape divided by the solver impulse interval. |
| `net_torque_nm` | Moment of those forces about the sensor origin. |
| `center_of_pressure_sensor_m` | Solver-force-magnitude-weighted contact position in sensor coordinates. |
| `geometric_contact_centroid_m` | Area-weighted centroid of active depth samples. |
| `contact_area_m2` | Sum of represented physical sample areas. |
| `maximum_depth_m` / `mean_depth_m` | Saturation-aware local depth reductions. |
| `tangential_speed_rms_m_per_s` | Physical-area-weighted RMS of active sample tangent speed. |
| `maximum_tangential_displacement_m` | Maximum bounded anchor motion over the active atlas. |
| `friction_utilization_force_weighted` | Normal-force-weighted `|J_t| / (mu_static J_n)` from solver evidence. |
| `friction_utilization_maximum` | Maximum finite utilization over matching solver contacts. |

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

It encodes sample, motion, reduction, and history kernels without allocating,
committing, waiting, or reading back. `nativeBuffer` publishes depth,
velocity, tangent motion, validity, object IDs, summaries, and statuses
directly to the native tensor boundary. Host `observe` and `readback` are
convenience and diagnostic paths.

The authored-pack MLX path owns the same cooked sensor arena. After the final
solver microstep, `WorldStepPrimitive` materializes committed body velocity,
packs final solver contacts, samples tactile geometry, reduces summaries, and
publishes named arrays on MLX's active command encoder. Previous depth,
validity, object identity, tangent motion, anchors, frame index, time, and
reset are ordinary `WorldState` arrays. A pack with no tactile sensors has
zero-sized tactile state, creates no tactile pipelines or resources, and
encodes no tactile dispatch.

Debug hits are disabled by default. The choice is specialized when the Metal
pipelines are created, not carried as a per-frame flag. In headless mode the
dense hit, hit-history, and hit-readback allocations collapse to fixed dummy
bindings and the specialized kernels do not access them.

## Temporal and replay semantics

- `frameIndex` determines each sensor's update schedule.
- A skipped update retains depth, validity, and identity while reporting zero
  depth velocity and zero instantaneous tangent velocity.
- A target-local anchor is established at contact onset. The same rigid target
  point is reprojected into the current sensor frame on later observations.
- Contact loss, target-identity change, or environment reset clears the anchor
  and displacement. Authored maximum tangent motion clamps the proxy.
- A reset mask clears temporal depth and tangent history for the reset
  environment.
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
and indenter asset, sequence and frame identity, then records the
sensor-relative pose, indentation depth, timestamp, raw frame, target map, and
optional validity, tangent-motion, and measured-wrench data. Relative paths
resolve against the manifest.

The target map should be generated from the known indenter geometry and
measured rig pose using the same atlas query definition as simulation.
Training minimizes masked metric pixel MSE, with an optional spatial-gradient
term. Saved weights retain the one observation fingerprint needed to prevent
a policy or translator from consuming the wrong atlas.

The model uses operators selected to be Core ML friendly, but this milestone
does not claim a validated Core ML conversion. The saved translator contract
records `conversion_validated: false` until an exported model is tested.

## Learned hierarchy

`python/metalrobo/tactile_latent.py` implements optional MLX layers without
changing the raw metric definition:

1. `SharedTactileEncoder` uses a small modality-specific residual CNN, masked
   pooling, and an explicit 64-state GRU. Canonical simulation, RGB camera,
   capacitance, magnetic, marker-motion, and force-array stems share the same
   `[environment, sensor, 64]` policy interface.
2. `TactileReconstructionDecoder` supplies training-only native,
   depth/tangent, wrench, center-of-pressure, and friction heads.
   Paired-stimulus contrastive and masked metric losses support
   self-reconstruction and cross-reconstruction training.
3. `ObjectContactFieldEstimator` maintains a 128-state temporal estimate and
   evaluates contact at caller-provided object-local points. Semantic features
   are accepted only by a model explicitly configured for authored semantics.
4. `TactileDynamicsModel` predicts the next latent, contact transition,
   wrench, friction utilization, contact loss, and uncertainty from current
   tactile latent, state, and action.
5. Missing-sensor imputation is opt-in. Policies always receive measured versus
   predicted masks and confidence; a missing real encoder asset fails rather
   than substituting simulated or imagined touch.

The simulator additionally publishes a fixed-capacity masked point set with
object shape identity, object-local point/depth, and object-local sensor
normal. It does not assign force to depth samples. Exact solver contact points
and impulses remain a separate evidence set.

`TactileEncoderAsset` binds a learned encoder to one observation fingerprint.
`check_stateful_encoder_parity` validates sequence outputs and recurrent state
through caller-supplied MLX and Core ML runners. Core ML tooling remains an
optional hardware-deployment dependency; simulation never invokes Core ML.

These model definitions and failure boundaries are implemented, but no
cross-sensor weights, Core ML artifact, object field, or predictive model is
claimed trained by the repository. Reduced-order MPM, sensor-specific shear
rendering, and force-adapter pretraining remain offline calibration tools, not
live physics modes.

## Actuator transfer boundary

Accurate touch does not compensate for an idealized actuator. An optional
`ActuatorProfile` therefore records joint-side torque constant, current limit,
no-load speed, efficiency, backlash, and command delay per actuated DoF.
Cooking derives stall torque; an explicit calibrated bit distinguishes
measured records from engineering placeholders. Existing armature, dry
friction, effort limits, and controller gains remain authoritative.

Native Metal applies the current/stall limit, linear torque-speed envelope,
and efficiency in the actuation kernel. MLX carries deterministic backlash
play and delay history explicitly. Training can select one correlated
six-parameter multiplier profile per environment at reset; the resulting
fixed per-environment records are branch-free solver inputs. Robots without
identified data retain neutral execution records and cannot be presented as
actuator-calibrated.

## Acceptance

There is one geometry check, one executable example, and one optional
performance tool:

- `metalrobo_tactile_check` compares the clear FP64 reference with Metal for
  flat and curved contacts, analytical and cooked geometry, transforms,
  saturation, temporal updates, solver wrench, center of pressure, tangent
  projection, anchor resets, identity changes, friction utilization, and
  batched determinism.
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

- Tangent displacement is a bounded rigid-anchor motion proxy. It is not
  distributed elastomer shear, micro-slip, viscoelastic memory, elastic waves,
  optical appearance, or a complete force field.
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
- Franka actuator values are not fabricated. Hardware transfer evaluation
  remains blocked on measured profiles and physical sensor/encoder artifacts.
