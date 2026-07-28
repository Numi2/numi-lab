# MetalRobo Python

This package exposes the batched native Franka runtime as NumPy views and
trains continuous-control policies with a native MLX PPO implementation.

Build the C++/Metal runtime first, then install the Python package:

```sh
cmake -S .. -B ../build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ../build
python3 -m pip install -e .
```

By default the package loads `../build/lib/libmetalrobo.dylib` and
`../build/shaders/MetalRobo.metallib` relative to this `python/` directory.
Installed or relocated builds can set:

```sh
export METALROBO_LIBRARY=/absolute/path/to/libmetalrobo.dylib
export METALROBO_METALLIB=/absolute/path/to/MetalRobo.metallib
```

Train PPO and save resumable checkpoints:

```sh
metalrobo train --envs 1024 --iterations 1000 --checkpoint-dir runs/franka
metalrobo train --envs 1024 --iterations 1000 \
  --resume runs/franka/checkpoint-000100
```

Measure native simulation throughput or policy rollout throughput:

```sh
metalrobo benchmark --envs 4096 --steps 2000
metalrobo benchmark --envs 1024 --steps 1000 \
  --checkpoint runs/franka/checkpoint-000100
metalrobo rollout --envs 1024 --steps 2000 \
  --checkpoint runs/franka/checkpoint-000100
```

`NativeRuntime.observations`, `rewards`, `terminated`, `body_positions`, and
`body_rotations` are read-only, zero-copy NumPy views over simulator-owned
shared memory. They are updated in place by `reset()` and `step()` and become
invalid after `close()`. Copy a view when it must outlive the current step.
