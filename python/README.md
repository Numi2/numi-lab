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
from metalrobo import compile_world, initial_state, step

world = compile_world("franka", physics_substeps=4)
state = initial_state(world, 1024)
actions = mx.zeros((1024, world.nv), dtype=mx.float32)

compiled_step = mx.compile(
    lambda state, action: step(world, state, action)
)
output = compiled_step(state, actions)
mx.async_eval(output.next_state, output.observations)
```

`WorldState` is an explicit PyTree containing q/v, scene-body state, and solver
cache. `StepOutput` contains next state, observations, fixed-shape
contact/sensor arrays, typed status, `physics_error`, and acceleration.
Reset masks and randomized reset state are explicit MLX inputs.

The current active-encoder primitive supports contact-free Franka and G1 ABA.
It rejects non-empty scene/contact state instead of silently using ctypes.
JVP, VJP, and `vmap` are deliberately unsupported in this tranche.

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

This first task is `franka_joint_stabilization_v1`, not contact manipulation.
Contact PPO becomes valid only when the standalone contact graph is promoted
to the active-encoder adapter.

## Debug compatibility path

The old NumPy/ctypes Franka task is retained as an explicit oracle:

```sh
metalrobo train --backend ctypes-debug --envs 1024
metalrobo benchmark --envs 4096 --steps 2000
```

`NativeRuntime` exposes read-only zero-copy NumPy views over simulator-owned
shared memory. They are invalid after `close()`. This path is not used
silently by the MLX primitive.
