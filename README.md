# MetalRobo

MetalRobo is an Apple-native robotics physics and reinforcement-learning
runtime. C++ owns the simulation model and authoring logic, Metal executes
batched physics and sensing, MLX owns learning, and Swift/Python expose native
application and experiment interfaces.

The runtime does not call an external physics engine.

## Current architecture

```text
authored robot + world + sensors
                |
                v
       deterministic compiler
                |
                v
          current world pack
                |
                v
      persistent Metal execution
                |
                v
       native observation buffers
                |
                v
          MLX policy/training
```

The main product paths are:

- rigid and articulated dynamics with deterministic contact;
- authored world families and batched environment execution;
- Visual Presentation V2 as the only presentation scene;
- geometry-native tactile depth, solver wrench, and center of pressure;
- native MLX, Swift, and Python integration.

There is no collision-derived presentation fallback. Perception contracts
whose names contain V1 are current wire formats, not alternate renderers.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

While changing one subsystem, build and run its one owning check instead of
running every executable. The repository rules in [AGENTS.md](AGENTS.md)
define the versioning, fingerprint, hot-loop, and verification policy.

## Tactile example

```sh
cmake --build build --target \
  metalrobo_tactile_check \
  metalrobo_franka_tactile_example
./build/bin/metalrobo_tactile_check
./build/bin/metalrobo_franka_tactile_example
```

The example uses the normal authored world pipeline, two Franka fingertip
atlases, actual solved contact evidence, native Metal observation buffers, and
a deterministic tactile feedback controller. It does not claim real-hardware
transfer.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Persistent Metal execution](docs/METAL_WORLD.md)
- [World authoring and packs](docs/WORLD_ENGINE.md)
- [Visual platform](docs/VISUAL_PLATFORM.md)
- [Geometry-native tactile sensing](docs/TACTILE_GEOMETRY_BRIDGE.md)
- [Numerical contract](docs/NUMERICS.md)
- [Validation evidence](docs/VALIDATION.md)
- [Detailed capability ledger](docs/ENGINE_CAPABILITY_LEDGER.md)

Robot-model provenance and upstream notices are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
