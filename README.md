# MetalRobo

MetalRobo is a Metal-native robotics simulation and reinforcement-learning
runtime for Apple silicon. The first vertical slice targets thousands of
parallel Franka Panda environments on one Mac GPU; Unitree G1 is the next
articulation target.

The engine is being built from a native C++23 model/runtime core,
Objective-C++ Metal host, and hand-written Metal Shading Language kernels.
MLX is the primary learning backend. No external physics engine is used at
runtime.

## Working v0.1 slice

- Reduced-coordinate articulated-body dynamics on Metal
- Batched sphere/capsule contacts against a ground plane
- Effort-bounded PD joint actuation and hard/soft joint limits
- Device-resident resets, observations, rewards, and termination
- Franka 7-DoF reach environment
- Native C ABI and Python/MLX PPO training
- FP64 CPU reference and a CPU/Metal convention probe

This is not yet a MuJoCo replacement for general manipulation. Version 0.1
has fixed-base trees and compliant ground contact; free bodies, pair/self
collision, a frictional constraint solver, importers, rendering, and sensors
are explicit next milestones.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/metalrobo_cpu_probe
./build/bin/metalrobo_parity_probe
./build/bin/metalrobo_bench --envs 1024 --steps 1000
```

Python support lives under `python/` and loads the native library from the
CMake build tree by default.

```sh
python3 -m pip install -e python

metalrobo benchmark --envs 1024 --steps 500

metalrobo train \
  --envs 1024 \
  --rollout-steps 32 \
  --iterations 1000 \
  --minibatch-size 8192
```

The native engine has no third-party physics dependency. Python training
requires Python 3.10+, NumPy, and MLX. A 24 GB 10-core Apple M4 development
machine measured about 201k environment control-steps/s at 1,024 environments
and 218k/s at 4,096 environments, with four physics substeps per control step.
See [validation](docs/VALIDATION.md) for the exact commands and boundaries.

## Design and research

- [Architecture](docs/ARCHITECTURE.md)
- [Numerical contract](docs/NUMERICS.md)
- [Competitor landscape](docs/LANDSCAPE.md)
- [Heavy-lifting roadmap](docs/ROADMAP.md)
- [Provenance](docs/PROVENANCE.md)
