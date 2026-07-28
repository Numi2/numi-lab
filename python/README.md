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

`WorldState` is an explicit PyTree containing q/v, scene-body state, and solver
cache. Contact worlds are compiled explicitly:

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

`StepOutput` contains next state, observations, typed status,
`physics_error`, acceleration, and fixed-capacity contact evidence with
stable IDs, counts, and masks.
Reset masks and randomized reset state are explicit MLX inputs.

The active-encoder primitive supports contact-free Franka/G1 ABA plus the
Franka dynamic-cube contact scene through the same device graph. No NumPy or
ctypes fallback is reachable. JVP, VJP, and `vmap` are deliberately
unsupported in this tranche.

## MLX-native PPO

```sh
metalrobo train \
  --backend mlx \
  --envs 1024 \
  --rollout-steps 64 \
  --rollout-chunk-size 16 \
  --iterations 1000 \
  --checkpoint-dir runs/franka-stabilization
```

Policy inference, effort mapping, physics, rewards, termination/reset, GAE,
rollout storage, and PPO updates remain MLX arrays. Policy/physics/reward is
inside `mx.compile`; bounded lazy chunks use `mx.async_eval`. Blocking
evaluation occurs only at declared rollout/logging, optimizer, and checkpoint
boundaries.

The packaged trainer remains the `franka_joint_stabilization_v1` baseline.
Contact policies can wrap the pure `step()` transition with MLX reward,
termination, and reset logic inside `mx.compile`.

## Debug compatibility path

The old NumPy/ctypes Franka task is retained as an explicit oracle:

```sh
metalrobo train --backend ctypes-debug --envs 1024
metalrobo benchmark --envs 4096 --steps 2000
```

`NativeRuntime` exposes read-only zero-copy NumPy views over simulator-owned
shared memory. They are invalid after `close()`. This path is not used
silently by the MLX primitive.
