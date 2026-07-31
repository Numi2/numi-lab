<div align="center">

# MetalRobo

**Apple-native robotics simulation, sensing, rollout, and learning infrastructure**

`Swift 6` · `C++23` · `Metal` · `MLX Swift`

<img src="docs/media/metalrobo-unitree-g1.webp" alt="Official Unitree G1 geometry rendered by MetalRobo" width="100%" />

</div>

MetalRobo is a pre-release robotics engine for Apple Silicon. Its production
direction is one compiled simulation, one persistent Metal session, native
Swift rollout scheduling, and MLX used only for learning.

```text
URDF / SRDF / MJCF / WorldPack
              +
         TaskPack + PolicyPack
              |
              v
       SimulationCompiler
              |
              v
       CompiledSimulation
              |
              v
    MetalSimulationSession
    physics + tasks + sensors + policy
              |
              v
       compact rollout views
              |
              v
           MLX learner
```

The diagram is the required product architecture. It is not a declaration
that every component is complete. The generated
[capability matrix](docs/CAPABILITIES.md) is the sole present-tense feature
record and refuses qualified claims without an owning passing check and
evidence manifest.

## Current boundary

The qualified path includes native rigid and articulated dynamics, compiled
joint/contact tasks, persistent Swift-scheduled Metal rollouts, dense policy
inference, authored visual presentation, and native tactile output. Important
release blockers remain:

- NumiSolver does not yet accept rod-bearing contact worlds.
- General MJCF import is not implemented.
- TaskIR lacks general body, site, and SE(3) goal operators.
- Sensor declarations now compile into one fingerprinted SensorIR layout, but
  presentation, tactile, and state sampling do not yet execute on one session
  schedule or bind directly into TaskIR observations.
- PolicyIR does not yet support convolution, recurrence, or attention.
- Native validity-aware adjoints are not implemented.
- G1 standing and walking are not qualified.
- The ten-million-transition release soak has not run.

MetalRobo therefore does not currently claim MuJoCo/Isaac equivalence,
state-of-the-art performance, sim-to-real transfer, or complete
differentiability.

## Product rules

- Physics, contact, actuator, RNG, reset, and sensor history remain in native
  persistent buffers.
- Swift owns submission scheduling, rollout lifetimes, policy revisions,
  failures, and deployment.
- MLX owns networks, losses, optimizers, minibatches, and checkpoints—not the
  simulation command encoder or simulator state.
- Python is permitted only for offline import, conversion, dataset, and
  external-comparator tooling.
- A new supported robot or task must compile from assets and packs without a
  new robot-specific C++, Swift, or Metal runtime path.
- Visual Presentation V3 is the only authored visual route. Collision geometry
  is never a presentation fallback.
- Unsupported topology fails compilation explicitly; the runtime does not
  silently substitute another physics engine.

## Build and focused checks

MetalRobo targets Apple Silicon and the macOS 26 SDK.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target \
  metalrobo_runtime_abi_check \
  metalrobo_capability_matrix_check \
  metalrobo_metal_world_contact_probe \
  metalrobo_task_program_check \
  metalrobo_tactile_check
ctest --test-dir build --output-on-failure -L owner
```

Run the Swift-owned native session:

```sh
./build/bin/metalrobo_simulation \
  --metallib build/shaders/MetalRobo.metallib \
  --envs 32 --steps 48 --chunk 8 \
  --scene terrain --numi-iteration-policy fixed --native-policy
```

Run learning through the in-process Swift/MLX boundary:

```sh
cmake --build build --target metalrobo_train
./build/bin/metalrobo_train --help
```

## Repository map

| Path | Ownership |
|---|---|
| `include/metalrobo` | Public/native contracts and shared generated ABI. |
| `src/core` | Compilers, reference mechanics, authored packs, and CPU oracle code. |
| `src/metal` | Production physics, contact, task, policy, visual, and tactile kernels. |
| `bindings/swift` | Public scheduling and learner boundary. |
| `schemas` | Persisted formats, generated ABI source, and capability registry. |
| `evidence` | Revision- and hardware-bound qualification manifests. |
| `python` | Code generation and offline tooling; not a production scheduler. |
| `apps` | Owning checks, product tools, diagnostics pending consolidation, and benchmarks. |

## Documentation

- [Architecture and ownership](docs/ARCHITECTURE.md)
- [Authoring and compilation](docs/AUTHORING.md)
- [Native runtime](docs/RUNTIME.md)
- [Numerical contract](docs/NUMERICS.md)
- [Validation and release gates](docs/VALIDATION.md)
- [Generated capabilities](docs/CAPABILITIES.md)
- [Media provenance](docs/media/README.md)

Robot assets and third-party attributions are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
