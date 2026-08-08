# MetalRobo visual simulation and perception platform

MetalRobo now turns one authoritative live physics state into synchronized
camera observations, dense supervisory truth, replaceable perception inputs,
policy observations, and deterministic visual episode streams.

The platform defines the data plane. A policy or perception model remains an
interchangeable consumer.

## Implemented architecture

```mermaid
flowchart LR
    Physics["Live Metal physics state"] --> Scene["Physics-bound visual scene"]
    World["WorldFamily scenario and appearance"] --> Scene
    Scene --> Sensor["Metal visual sensor runtime"]
    Sensor --> Frames["VisualFrameBatchV1"]
    Sensor --> Truth["VisualTruthBatchV1"]
    Frames --> Provider["PerceptionProviderV1"]
    Provider --> Results["Typed perception results"]
    Frames --> Assembler["PolicyObservationAssemblerV1"]
    Results --> Assembler
    Proprio["Proprioception, prior action, task command"] --> Assembler
    Assembler --> Actor["Deployable observation group"]
    Truth --> Supervisor["Supervisory or critic group"]
    Frames --> Episode["VisualEpisodeStreamV1"]
    Truth --> Episode
    Actor --> Episode
```

### Physics-bound scene

`VisualSceneManifestV3` binds immutable `VisualAssetPackV2` files to the
authoritative `WorldTemplate`. A live scene contains only pack references,
body/link instances, camera bindings, the selected light rig, and an optional
`VisualEnvironmentPackV2`; it never flattens authored geometry into a host
scene arena.

`metalrobo_visual_cook` accepts GLB, glTF, STL, USD, USDA, USDC, and USDZ. STL
is imported through Model I/O without a coordinate or unit conversion so its
authored URDF visual transform remains the single source of scale. The USD
path uses Model I/O with `MTKMeshBufferAllocator`, applies stage units and
up-axis conversion, preserves material subsets and symbolic link bindings,
and writes a sectioned, content-addressed `.mrvpack`. Geometry, material
records, texture descriptors, texture payloads, and symbolic bindings occupy
independently hashed sections. USDZ package resolution remains inside Model
I/O; a reusable `MTKTextureLoader` converts one material texture at a time
without color-transforming linear normal, roughness, metallic, occlusion, or
mask data. Separate Model I/O roughness, metallic, opacity, and clearcoat-gloss
maps retain scalar-channel semantics instead of being interpreted as glTF's
packed material channels. Vertices, indices, and texture subresources are
aligned for direct Metal I/O loading. Authored minification, magnification,
mipmap, and U/V wrap modes are encoded independently; renderer compilation
deduplicates only the sampler states actually used by the scene.

`metalrobo_environment_cook` converts HDR or EXR equirectangular images into a
diffuse irradiance cube, a full prefiltered GGX mip chain, and the shared DFG
BRDF LUT. The resulting `.mrenv` is immutable and records color conversion,
sample schedules, exact IBL-kernel hash, and source provenance. GPU
convolution runs in one ordered command stream; derived texture readback uses
an actual two-slot pipeline with exact final-capacity reservation, avoiding
per-subresource synchronization and vector-growth memory spikes.

Captured Gaussians remain an optional pre-lit appearance layer in
`VisualRenderSceneV3`. They share metric depth, identity, normal, and motion
outputs with authored PBR geometry.

`writeVisualSceneManifestV3` publishes immutable pack hashes and lightweight
runtime bindings against `visual_scene_manifest_v3.schema.json`.

`composeVisualBodyStates` evaluates articulated bodies with MetalRobo's
authoritative FP64 kinematics and combines them with sampled scene bodies in
global `EngineModel` body order. Articulated body poses are COM-centred, so
URDF visual cooking applies each link's authored origin-from-COM offset before
binding its mesh. The renderer therefore consumes the same poses used by
collision and contact without disassembling link-origin geometry.

### Visual sensor runtime

`MetalHybridRenderer` streams cooked sections into private placement heaps
with Metal I/O, binds native textures and samplers once through a tier-2
argument buffer, and consumes live body/link buffers directly on Metal. It
produces:

- Linear RGBA.
- Metric depth and depth validity.
- Semantic, instance, link, and primitive identities.
- Camera-space surface normals.
- Previous-to-current pixel motion.
- Frame validity and versioned frame metadata.

Both renderer profiles use the same metallic-roughness PBR, visible HDR
environment, and image-based lighting implementation. A compact
environment-major state pass resolves current and previous camera plus authored
instance transforms once, removing repeated body binding and quaternion work
from triangle visibility, shadows, and composition. `sensor_fast` packs depth
and triangle identity into one Apple9 64-bit atomic visibility key. Streamed
indexed geometry is partitioned into 64-triangle clusters, and tight cluster
bounds are built once on Metal after the Metal I/O load. Each camera pass
culls environment-major clusters in parallel against the calibrated frustum
and active rolling-shutter band. The flat triangle pass then performs one
coalesced triangle-to-cluster and cached-visibility lookup before touching
vertices, so high-poly work scales with visible geometry without a
compact-list scan or an encoder transition. Shadow clusters use a cooperative
light-frustum path.

Within visible clusters, projected triangles with a bounded microtriangle box
use the direct atomic lane, while larger triangles append their
already-projected screen-space record to 16×16 tiles. One 256-thread group
then resolves each tile in 128-record shared-memory batches. This keeps
subpixel robot meshes inexpensive, eliminates a second world/camera transform
for tiled geometry, and prevents large triangles from serially walking their
complete pixel bounding box. Tile-list order is irrelevant because the packed
winner is an exact minimum; capacity overflow takes the bounded atomic lane
without dropping geometry. Near-plane work launches indirectly only when
clipped geometry exists; winner clearing and sensor response are fused into
passes that already touch the pixels. World buffers are resolved once per
physical frame, resource residency is declared once per active encoder, and
fixed buffer tables are bound as ranges rather than as individual Objective-C
calls.

`sensor_fast` uses environment-major shadow atlases and stratified space-time
samples during physical exposure. Near-plane scratch storage scales with the
compiled triangle count up to a fixed GPU-local ceiling rather than reserving
the ceiling for every scene. `sensor_reference` uses
one compacted BLAS per unique authored mesh and GPU-authored component-motion
instances. TLASes cover groups of at most 32 environments; a one-hot Metal
instance mask isolates every environment without shifting world coordinates
or sacrificing metric precision. Group builds and refits share one encoder
with disjoint scratch ranges, and periodic rebuilds restore traversal quality
after accumulated motion. Primary visibility, all-scene shadows, and
dynamic-only Gaussian receiver shadows use the same structures with explicit
role filtering. The renderer then applies stratified subpixel and shutter-time
samples, direct shadow rays, and exact per-row exposure timing. Deployable RGB
is integrated over those samples while depth, identities, normals, and truth
remain exact center-exposure observations. Sensor effects are keyed by
scenario, camera, sensor sequence, frame identity, pixel, and sample, so
rerenders remain deterministic. Sensor depth validity is separate from
geometric validity:
range failure or dropout masks deployable depth without erasing
simulation-only identities, normals, motion, or visibility.
Disabled color noise, depth noise, and dropout execute no random hash work.
Reference subpixel rotation is generated once per pixel and reused across its
exposure samples.

`renderLive` accepts host-visible state for inspection and export.
`encode` accepts borrowed Metal body-state buffers and a caller-owned active
compute encoder. It commits no command buffer and performs no readback. RGB,
depth, identity, normal, motion, and validity buffers stay device-resident and
are exposed through `nativeBuffer` for MLX, Core ML, or another Metal stage.
`MetalHybridObjectTracker` is the native closed-loop adapter: it renders on a
borrowed `MetalWorld` command buffer, reduces metric depth and instance
identity into compact root-local object position and velocity tracks, and
overwrites compiled SensorPack observation slots before policy inference. Live
camera and root poses come from device buffers rather than a static camera
approximation. One cooperative threadgroup reduces each environment/object
track and temporal track state remains on the GPU.
The same adapter can publish a SensorPack-authored ball-only masked-depth
contract. Exact authored instance identities select pixels, invalid and
non-selected pixels become calibrated far depth, and a device-resident ring
publishes sparse temporal offsets directly into actor slots. The bundled G1
dodge task uses a 16x9 plane normalized over 0.1--5.0 m at offsets 0, 3, 8,
and 18 control steps. It retains all 576 pixels and appends 24 compact values
reduced from those same corrupted planes: visibility confidence, image-plane
bearing and elevation, nearness, and apparent area for each frame, followed by
their newest-to-oldest temporal changes. Thus approach and expansion are easy
for a small policy to consume without introducing scene-state position or
velocity. Object tracks remain critic-only for this task. Reset fills the
complete depth ring from the first accepted post-reset frame, so no visual
belief crosses episode boundaries and the four temporal changes begin at zero.
The bundled head sensor authors a 54-degree vertical field of view and a
50 Hz cadence. Its torso-local mount is 0.08 m forward and 0.45 m upward with
the optical axis pitched 20 degrees above torso-forward. At the task's 50 Hz
control rate, every sparse-history slot therefore represents a distinct
physical exposure; a slower sensor intentionally repeats held frames. The C
rollout boundary accepts an explicit vertical field of view in degrees; zero
retains the legacy focal-length rule for existing callers.
`MetalRoboRunManifest.visualSensor` authors pack references, a body-bound
camera, the matching `WorldFamily`, and this tracker as part of SensorPack
compilation. There is no post-construction sensor attachment API. Explicit
visual/environment content hashes and the complete sensor configuration enter
the SensorPack fingerprint before the executor is created. The immutable
`CompiledRun` retains the complete executable visual program, and executor
construction reads only that retained program. Device materialization reloads
the content-addressed packs and rejects any hash change instead of executing a
different sensor under the compiled identity. Explicit
rigid-body pack bindings keep moving scene objects on the accepted physics
timeline; articulated bindings keep robot presentation on link states. Reset
clears temporal tracks atomically with simulator reset, and rollout chunk size
does not alter the published observation artifact.
`encodeGraph` additionally accepts an active-compute callback surface and a
complete set of caller-owned observation buffers. The Python
`visual_observation` custom primitive uses that surface to write linear RGB,
metric depth, segmentation, identities, normals, motion, and validity directly
into MLX allocations. Its `graph_only=True` renderer keeps no duplicate
capacity-sized observation planes, while retaining the bounded raster,
visibility, and temporal scratch required to produce those outputs. The
policy default is RGB, metric depth, and validity; dense segmentation,
identities, normals, and motion are explicitly selected static truth channels
and otherwise have zero spatial extent. The
primitive rejects a sampled world whose authored pack hash differs from the
compiled physics world. It registers its array dependencies with MLX,
adds no command buffer, and attaches the renderer's private heap through a
Metal residency set compiled and committed once with the immutable scene, then
reused on MLX's current command buffer. This follows MLX's
[custom-extension contract](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
while retaining native heap and argument-buffer residency.
Explicit inspection/export readback coalesces every plane into one transient
aligned shared buffer and reuses caller-owned host capacity. This avoids
per-modality Metal allocations without retaining a resolution-sized staging
buffer after the readback completes.

### Live run inspector

`metalrobo_task_rollout` and `metalrobo_task_train` can open a small native
macOS `MTKView` beside a run. For ordinary use, save the authored visual
observation file as `.numi/window.visual-observation.json` and run:

```sh
numi window
```

The workspace command builds its own isolated runtime, discovers that saved
scene, starts a one-environment zero-action preview, and stops it when the
window closes. The lower-level executable remains available for composed
training/evaluation flows:

```sh
metalrobo_task_rollout ... \
    --inspect-scene /path/to/visual-observation.json \
    --inspect-width 960 --inspect-height 540
```

`--inspect-scene` takes the same portable `numi.visual-observation.v1`
artifact used for authored cameras, but it compiles an independent
presentation-only renderer. It neither changes `CompiledRun`, policy
observations, policy fingerprints, nor the task SensorPack. The render is
encoded after the final accepted control state of each existing rollout
submission, into the submission's command buffer; it never introduces a
command-buffer wait, a CPU state copy, or a pixel readback.

The window consumes device-private linear RGB buffers through a three-slot
ring. A slot is released only after its display command buffer completes; if
the window or compositor falls behind, the producer drops the newest preview
instead of blocking physics, control, or learning. The window currently
shows representative environment zero. It is an inspection aid, not evidence
of real-hardware behavior or a media-capture path. Use ordinary visual export
or state-trace facilities when durable frames or artifacts are required.

Fixed and wrist cameras in `FrankaPickPlaceWorldFamily` are the reference
integration. The fixed camera is calibrated toward the manipulation
workspace; the wrist camera is bound to the final articulated link.

### Deployable observations and supervisory truth

`VisualFrameBatchV1` is accepted for simulation, physical capture, and replay.
It contains:

- Linear RGB, metric depth, and depth validity.
- Intrinsics, distortion, and camera-to-base transforms.
- Capture time, frame age, exposure, shutter readout, frame index, sensor
  sequence, sensor identity, and source.
- Immutable fingerprints for the episode twin, scenario, renderer, sensor
  profile, and calibration.
- Host arrays or typed device-buffer views.

Host-visible invalid depth is canonicalized to zero and interpreted only
through the depth-validity mask.

`VisualTruthBatchV1` is separate. It contains dense semantic/instance/link
masks, normals, motion, visibility, occlusion, object and link poses,
keypoints, and contact annotations in a declared coordinate frame. Dense IDs
remain typed `uint32`; sparse pose/keypoint/contact records use exact
`float64` packing so `uint32` identities and invalid sentinels are not rounded.
Dense visibility is surface coverage in `[0, 1]`; occlusion is its quantized
inverse, with `255` representing an absent or fully occluded truth surface.

`assembleVisualBatches` converts synchronized renderer readbacks into both
contracts and rejects stale or mismatched view metadata. A physical RGB-D
adapter constructs the same `VisualFrameBatchV1` and simply has no
simulation-only truth. Camera timestamps remain per-view so independently
clocked sensors can be aligned under one batch frame index.

### Replaceable perception providers

`PerceptionProviderV1` advertises a content hash, required modalities,
temporal window, device-buffer support, and typed capabilities:

- Dense depth or semantic output.
- Instance masks and tracking.
- Object hypotheses and 6D poses.
- Keypoints.
- Dense features or compact embeddings.

The provider interface is the correct integration point for an Ultralytics
YOLO/Core ML package, an MLX encoder, or a future foundation model. Model
choice does not alter the simulator, episode format, or policy contract.

The Python `CallablePerceptionProviderV1` adapter wraps any callable that
honors this provenance and result contract. Provider results and every tensor
are bound to the originating frame index and timestamp before they can enter a
policy observation; zero detections are represented by valid zero-length
tensors rather than a malformed result.

### Foundation-policy action chunks

`numi foundation` is the provider-neutral boundary for a vision-language-action
model that proposes a finite action chunk. Its first adapter consumes NVIDIA's
staged GR00T N1.7 G1 ONNX export: calibrated ego RGB and named G1 joint state
enter the provider, and named 16-step joint, effort, navigation, and base-height
arrays leave it. The artifact hash, observation fingerprint, stochastic-noise
fingerprint, execution providers, tensor shapes, and per-stage timings accompany
every chunk.

This is intentionally outside the simulator hot loop. A foundation model owns
neither contact nor dynamics and its output is not physical-success evidence.
Numi's native controller may track, reject, blend, or distill a proposed chunk;
Metal remains authoritative for physics, sensing, constraints, rollout, and
measured outcome. This separation also permits future GR00T, MLX, Core ML, or
other providers without changing the episode or simulator contracts.

The ONNX adapter executes preprocessing, visual-language backbone, action head,
and decode as sequential stages. On Apple, ONNX Runtime prefers Core ML and
retains CPU as an explicit unsupported-operator fallback. Only compact stage
outputs survive between large stages. That makes correctness possible on
today's unified-memory Macs while retaining a direct path to future graph
fusion, MLX conversion, persistent sessions, and faster Apple hardware.

### Authored visual-observation program

`numi.visual-observation.v1` removes robot and task identity from the sensor
authoring surface. The JSON artifact lists VisualPacks with their authored
asset, semantic, and instance identities; an optional environment pack; and a
camera parent, local pose, calibration, cadence, visibility threshold, and
retained-memory budget. Relative paths resolve beside the artifact, so the
configuration is portable and fingerprintable.

The Swift manifest loads this artifact before construction, while the native
compiler resolves body indices, tracked task entities, masked-depth slots,
history, and observation layout from the mechanics and SensorPack. Metal still
performs rendering and sensing. The schema is
`schemas/visual_observation.schema.json`.

```sh
numi evaluate --task ball-dodge --scene ground \
    --visual-observation-config /path/to/visual-observation.json \
    --envs 1 --steps 32 --chunk 1 --native-policy
```

### Policy observations

`PolicyObservationAssemblerV1` supports:

- Raw RGB-D.
- RGB with metric base-frame XYZ.
- Object-centric perception results.
- Dense feature maps.
- Compact latent representations.

Proprioception, prior actions, and task commands join the deployable actor
observation. Privileged physics state remains a separate supervisory or
critic group. RGB-XYZ backprojection inverts the same radial/tangential lens
model used by rendering, so base-frame points remain calibrated away from the
optical center.

### Visual episode stream

The Python `VisualEpisodeWriterV1` writes deterministic, content-addressed
NPZ chunks plus a canonical manifest. Each step can carry:

- Multi-view frames and camera calibration.
- Proprioception, actions, task commands, rewards, and events.
- Privileged state and dense visual truth.
- Compact named physics state such as `q`, `v`, and scene-body state for
  deterministic rerendering.
- Scenario identity and immutable renderer, asset, calibration, sensor, and
  physics provenance.

Camera exposure, shutter timing, sensor sequence, and the scenario fingerprint
are retained in every chunk. Independently clocked views may carry different
capture timestamps while sharing one policy-step frame index. Sensor identities
remain stable and each camera timeline remains monotonic.
`VisualEpisodeReaderV1` verifies the manifest, contiguous chunk layout, hashes,
array dtypes/shapes, calibration, frame ordering, depth validity, and scenario
provenance before yielding data.
Simulation and captured episodes share this format. The optional
`export_lerobot_v3` adapter uses LeRobot's native writer when LeRobot is
installed, mapping each parallel environment to an episode with streaming
multi-camera RGB, validity-masked metric depth, state, and actions.

## State-of-the-art alignment

| Signal | MetalRobo implementation |
| --- | --- |
| [ManiSkill3](https://arxiv.org/abs/2410.00425), [Isaac Sim Replicator](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/replicator_tutorials/index.html), and [Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/rendering/index.html) place batched sensors beside GPU physics. | Live transforms, rendering, dense outputs, and the active-encoder handoff remain on Metal. |
| [SplatSim](https://arxiv.org/abs/2409.10161), [RoboGSim](https://arxiv.org/abs/2411.11839), and [GSWorld](https://arxiv.org/abs/2510.20813) combine captured appearance with physics-backed geometry. | Captured Gaussian layers and body-bound meshes share the same renderer and identity/depth buffers. |
| [DP3](https://arxiv.org/abs/2403.03954) and [AnchorDP3](https://arxiv.org/abs/2506.19269) use calibrated 3D geometry. | Metric depth, calibration, base-frame XYZ, object/link poses, keypoints, visibility, and identities are first-class contracts. |
| [ACGD](https://research.nvidia.com/publication/2025-10_acgd-visual-multitask-policy-learning-asymmetric-critic-guided-distillation) and [Isaac Lab observation groups](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.envs.html) separate actor input from privileged information. | Deployable frames and policy observations remain distinct from truth and critic/supervisory groups. |
| [DINOv3](https://ai.meta.com/research/dinov3/) and [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) keep changing the useful representation layer. | Providers advertise capabilities and provenance while the simulator stays encoder-agnostic. |
| [SIMPLER](https://arxiv.org/abs/2405.05941) and [RialTo](https://arxiv.org/abs/2403.03949) depend on matched scenes and calibration. | Simulation, capture, and replay use one frame contract and one immutable episode provenance model. |

## Public surfaces

- C++ contracts: `include/metalrobo/VisualPlatform.hpp`
- Shared Metal ABI: `include/metalrobo/visual_platform_types.h`
- Metal runtime: `include/metalrobo/MetalHybridRenderer.hpp`
- Python contracts and episode stream: `python/metalrobo/visual.py`
- JSON schemas: `schemas/visual_*`, `schemas/visual_observation.schema.json`,
  `schemas/foundation_adapter.schema.json`, and
  `schemas/perception_provider.schema.json`
- Reference integration:
  `apps/visual_platform_probe.cpp`
- Foundation action-chunk adapter:
  `python/metalrobo/foundation_policy.py` and `numi foundation`
