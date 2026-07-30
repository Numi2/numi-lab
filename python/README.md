# MetalRobo Python and MLX

The default training path is a nanobind/MLX custom primitive pinned to
`mlx>=0.32,<0.33`. It allocates arrays through MLX and encodes physics into
MLX's active Metal command encoder. It does not create, commit, or wait on a
second command buffer and has no CPU fallback.

Build the standalone C++/Metal engine first, then build/install the extension:

```sh
cmake -S .. -B ../build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ../build
python3 -m pip install -e .
```

For an in-place development build:

```sh
python3 setup.py build_ext --inplace
python3 probes/mlx_world_probe.py
```

The extension finds `../build/shaders/MetalRobo.metallib` by default.
Relocated builds can pass `metallib_path=` to `compile_world()` or use the
CLI `--metallib` option.

## Pure MLX interface

```python
import mlx.core as mx
from metalrobo import (
    MetalWorldCapacityProfile,
    compile_world,
    initial_state,
    step,
)

profile = MetalWorldCapacityProfile()
profile.constraint_blocks = 32
profile.constraint_rows = 96
profile.solver_tiles = 1
world = compile_world(
    "franka",
    physics_substeps=4,
    capacity_profile=profile,
    stream=mx.default_stream(mx.gpu),
)
state = initial_state(world, 1024)
actions = mx.zeros((1024, world.nv), dtype=mx.float32)

compiled_step = mx.compile(
    lambda state, action: step(world, state, action)
)
output = compiled_step(state, actions)
mx.async_eval(output.next_state, output.observations)
```

`compile_world()` eagerly prepares its immutable pipelines/resources on the
selected Metal stream before lazy compilation. Call
`world.prepare_stream(stream)` explicitly before using another stream. A
capacity overflow never reallocates inside the primitive: the raw fixed-shape
status reports the exact required stage counts and the affected environment
keeps its input state.

### Geometric sensors and scene queries

The same compiled world supports arbitrary environment-major rays without a
CPU copy or a second command buffer:

```python
from metalrobo import materialize_body_states, scene_raycast

body_states = materialize_body_states(world, state)
hits = scene_raycast(
    world,
    body_states,
    origins=mx.array([[0.0, 0.0, 1.5]], dtype=mx.float32),
    directions=mx.array([[0.0, 0.0, -1.0]], dtype=mx.float32),
    maximum_distance_m=4.0,
)
```

Origins and directions may be shared as `(ray, 3)` arrays or supplied per
environment as `(environment, ray, 3)`. Results contain metric distance,
world-space point and normal, shape/body/material/feature identities, and
validity. Analytic shapes, convex geometry, and authored triangle meshes use
the same interface. If a tactile step already returned
`output.body_states`, pass that array directly and avoid another kinematics
dispatch.

Robot-mounted range sensing uses a graph-static pattern instead:

```python
from metalrobo import make_grid_ray_pattern, scene_raycast_pattern

terrain_grid = make_grid_ray_pattern(
    parent_body=base_body,
    size_m=(1.2, 0.8),
    resolution=(24, 16),
    origin_m=(0.0, 0.0, 0.5),
)
hits = scene_raycast_pattern(world, body_states, terrain_grid)
```

The pattern is shared across environments. Metal composes each ray with its
current parent-body pose and traverses the scene in the same dispatch, so no
dense world-space origin/direction arrays are produced per step. A LiDAR
pattern builder and arbitrary per-ray body bindings use the same result type.

Wave32 worker occupancy is fixed when the world is compiled. The device
profile supplies the default; `wave_worker_groups=32|64|96|128` is an explicit
override. Measure candidates outside rollout execution:

```sh
metalrobo tune-workers \
  --model franka \
  --scene cube \
  --envs 1024 \
  --steps 32
```

The command creates one fixed world per candidate, reports wall throughput,
and never autotunes inside `mx.compile`.

`WorldState` is an explicit PyTree containing q/v, scene-body state,
fixed-shape `RodState`, and solver cache. Rigid-only worlds carry zero-sized
rod arrays. Contact worlds are compiled explicitly:

```python
world = compile_world(
    "franka",
    scene="cube",
    environment_capacity=1024,
    solver_mode="throughput_tgs",
    ccd_mode="hybrid",
    max_ccd_advance_solve_passes=4,
    max_ccd_zero_time_replays=2,
    ccd_simultaneous_tolerance=1.0e-5,
)
state = initial_state(world, 1024)
output = step(
    world,
    state,
    mx.zeros((1024, world.nv), dtype=mx.float32),
)
```

The same primitive can select the certificate-producing quality path:

```python
quality_world = compile_world(
    "franka",
    scene="cube",
    solver_mode="quality_newton",
    ccd_mode="disabled",
)
```

`quality_newton` encodes manifold preparation, persistent-impulse velocity
reconstruction, direct or matrix-free Newton, and transactional application
through MLX's active encoder. A scan-ordered active-problem queue feeds the
same fixed occupancy-sized persistent SIMD32 worker pattern as throughput
contact; worker claim order cannot alter problem slots or row reductions.
Hybrid event-time quality re-solves are rejected until they preserve the same
semantics; there is no TGS fallback.

`StepOutput` contains next state, observations, typed status,
`physics_error`, acceleration, and fixed-capacity contact evidence with
stable IDs, counts, and masks. Every contact world also returns the common
eight-channel sensor summary: floating-root linear/angular acceleration (zero
for fixed-base robots), aggregate normal load, and active-contact count.
Reset masks and randomized reset state are explicit MLX inputs.

Authored tactile worlds enter through the same primitive:

```python
from metalrobo import (
    SharedTactileEncoder,
    TactileAtlasRange,
    canonical_metric_tactile_policy_observation,
    canonical_tactile_encoder_input,
    compile_world_pack,
)

world = compile_world_pack(
    "franka-tactile.mrworld",
    environment_capacity=1024,
    control_timestep=0.02,
)
state = initial_state(world, 1024)
output = step(
    world,
    state,
    state.actuators.effective_position_target,
)
encoder_input = canonical_tactile_encoder_input(
    output.tactile,
    (
        TactileAtlasRange(0, 32, 32),
        TactileAtlasRange(1024, 32, 32),
    ),
)
policy_touch = canonical_metric_tactile_policy_observation(
    output.tactile,
    (
        TactileAtlasRange(0, 32, 32),
        TactileAtlasRange(1024, 32, 32),
    ),
)
encoder = SharedTactileEncoder(
    modality="canonical",
    input_channels=7,
    summary_channels=16,
)
latent = encoder(
    *encoder_input[:3],
    encoder.initial_state(1024, 2),
    *encoder_input[3:],
)
```

`output.tactile` exposes named metric depth, depth velocity, bounded tangent
motion, physical summaries, and a masked object-local point set. Tactile
history and per-environment actuator profiles are explicit `WorldState`
arrays. A pack without authored tactile sensors has zero-sized tactile state
and encodes no tactile dispatch. Learned real-sensor encoders are registered
as exact artifacts; a missing encoder does not fall back to simulated or
imagined touch.

The built-in tactile PPO tasks use `policy_touch`: one deterministic
64-value metric stem per sensor plus presence and confidence. The learned
encoder occupies the same policy boundary only after matching weights are
available; random weights are never presented as a cross-sensor latent.

### Pinned physical tactile imitation

`metalrobo-tactile` ingests pinned LeRobot 3 seasons directly through Arrow
and PyAV, then trains a vision/state/ten-fingertip-wrench action-tube policy
with MLX on the Apple GPU. Splits are whole-season, normalization is
training-only, and EMA inference weights plus optimizer/training weights are
SHA-256 sealed at the checkpoint boundary.

```sh
python -m pip install -e 'python[tactile-dataset]'
metalrobo-tactile prepare \
  --dataset-root /path/to/origami \
  --output /path/to/contracts
metalrobo-tactile train \
  --dataset-root /path/to/origami \
  --stream-contract /path/to/contracts/tactile-stream.json \
  --manifest /path/to/contracts/dataset-manifest.json \
  --video-key observation.images.head_left \
  --video-key observation.images.wrist_left \
  --steps 100000 --output runs/origami
```

The initial public Origami contract deliberately blocks hardware execution:
the public card does not verify action semantics or wrench units/frames.
See
[`docs/TACTILE_GEOMETRY_BRIDGE.md`](../docs/TACTILE_GEOMETRY_BRIDGE.md)
for promotion, Wave asset cooking, replay alignment, and evaluation details.

The active-encoder primitive supports Franka, G1, and PSM through the same
device graph. Available contact scenes include dynamic cube/ground,
authoritative cooked-BVH4 rough terrain, and a dynamic curved needle for PSM.
Hybrid mode performs literal multi-event TOI advance/solve/continue inside
the active encoder. No NumPy or ctypes fallback is reachable. JVP, VJP, and
`vmap` are deliberately unsupported in this tranche.

The PyTree already carries the versioned rod-state contract. Nonzero rod
execution is currently rejected explicitly because the two-way procedural
capsule solver is still owned by the standalone Metal rod graph; MLX never
ignores or freezes a supplied thread while reporting a successful world step.

Multi-articulation generalized constraints use a second pure-array primitive
over the same active Metal encoder:

```python
from metalrobo import (
    compile_multi_articulated_program,
    step_multi_articulated,
)

program = compile_multi_articulated_program(
    "dual_psm_g1",
    environment_capacity=256,
    solver_mode="quality_semismooth_newton",
)
q = mx.broadcast_to(
    mx.array(program.default_q, dtype=mx.float32),
    (256, program.nq),
)
v = mx.zeros((256, program.nv), dtype=mx.float32)
next_v, impulses, status = mx.compile(
    lambda q, v: step_multi_articulated(program, q, v)
)(q, v)
```

The cooked ABA topology, generalized Jacobian, row chunks, inverse-mass
packets, and repeated right-hand sides are immutable device resources.
Inverse-ABA factorization and application use SIMD32 level frontiers with
stable parent-owned sibling reductions; no dense generalized inverse is
formed. Failure is isolated per environment: velocity rolls back to the
explicit input, impulses are zero, and the typed status remains visible.
`solver_mode` accepts `throughput_pgs` or
`quality_semismooth_newton`; the latter uses GPU natural-map Newton/CG with
line-search safeguards for scalar bounded rows. Supported compositions are
`dual_psm` and `dual_psm_g1`.

## MLX-native PPO

```sh
./build/bin/metalrobo_tactile_example franka-grasp \
  --write-world-pack /tmp/franka-tactile.mrworld
./build/bin/metalrobo_tactile_example g1-balance \
  --write-world-pack /tmp/g1-tactile.mrworld
./build/bin/metalrobo_tactile_example psm-needle \
  --write-world-pack /tmp/psm-tactile.mrworld

metalrobo train \
  --backend mlx \
  --task franka-grasp \
  --world-pack /tmp/franka-tactile.mrworld \
  --envs 1024 \
  --rollout-steps 64 \
  --iterations 1000 \
  --checkpoint-dir runs/franka-grasp

metalrobo train \
  --backend mlx \
  --envs 1024 \
  --rollout-steps 64 \
  --rollout-chunk-size 16 \
  --iterations 1000 \
  --checkpoint-dir runs/franka-stabilization

metalrobo train \
  --backend mlx \
  --task g1-standing \
  --world-pack /tmp/g1-tactile.mrworld \
  --envs 2048 \
  --rollout-steps 64 \
  --iterations 1000

metalrobo train \
  --backend mlx \
  --task g1-command \
  --world-pack /tmp/g1-tactile.mrworld \
  --envs 2048 \
  --rollout-steps 64 \
  --iterations 1000

metalrobo train \
  --backend mlx \
  --task g1-terrain \
  --world-pack /path/to/explicit-authored-g1-terrain.mrworld \
  --envs 1024 \
  --rollout-steps 64 \
  --iterations 1000

metalrobo train \
  --backend mlx \
  --task psm-needle \
  --world-pack /tmp/psm-tactile.mrworld \
  --envs 1024 \
  --rollout-steps 64 \
  --iterations 1000
```

Policy inference, effort mapping, physics, rewards, termination/reset, GAE,
rollout storage, and PPO updates remain MLX arrays. Policy/physics/reward is
inside `mx.compile`; bounded lazy chunks use `mx.async_eval`. Blocking
evaluation occurs only at declared rollout/logging, optimizer, and checkpoint
boundaries.

The CLI exposes the authored tactile tasks `franka-grasp`, `g1-standing`,
`g1-command`, `g1-terrain`, and `psm-needle`, plus the separate contact-free
Franka joint-stabilization diagnostic. Tactile tasks require `--world-pack`;
the terrain command does not synthesize terrain or adapt a flat pack. G1
command tasks carry episodic planar/yaw
commands inside the compiled observation and resample them on transactional
reset. Reset root position/yaw, joint pose, and generalized velocity are also
randomized as MLX arrays without changing immutable model parameters. The G1
and PSM collectors provide contact-capable task paths;
the policy, implicit targets, physics, reward, termination, reset, GAE, and
updates remain MLX arrays. The PSM task scores the measured dynamic needle
pose and contact evidence and never creates a weld or hidden grasp state.
Contact policies can also wrap the pure `step()` transition with task-specific
MLX reward and curriculum logic inside `mx.compile`.

## Debug compatibility path

The old NumPy/ctypes Franka task is retained as an explicit oracle:

```sh
metalrobo train --backend ctypes-debug --envs 1024
metalrobo benchmark --envs 4096 --steps 2000
```

`NativeRuntime` exposes read-only zero-copy NumPy views over simulator-owned
shared memory. They are invalid after `close()`. This path is not used
silently by the MLX primitive.
