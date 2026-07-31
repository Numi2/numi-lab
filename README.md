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

- The fixed-budget NumiSolver path accepts retained-operator rod contacts, but
  the residual-converged profile and the full rod mechanics corpus are not yet
  qualified.
- General MJCF import is not implemented.
- TaskIR supports selected-articulation body frames, static SE(3) goals, and
  fingerprint-bound SensorIR values/validity; scene-object sites, twist,
  acceleration, sampled goals, and generic reductions remain incomplete.
- State sensors execute on the session timeline and bind directly into TaskIR.
  Presentation and tactile sensing remain native but have not yet been folded
  into that same schedule.
- Swift PPO now collects into a three-slot shared Metal rollout ring and MLX
  consumes those buffers without Swift array concatenation or an MLX input
  copy. Native inference keeps two private policy banks, locks their compiled
  topology fingerprint, and swaps monotonic revisions at submission
  boundaries. Direct command-buffer rollout publication and a no-copy
  MLX-to-policy-bank blit remain unfinished.
- Dense PolicyIR inference now cooperatively reduces each output in an Apple
  GPU SIMDgroup instead of serially accumulating a neuron in one thread.
  Shape-specialized matrix tiles, reduced-precision weights, convolution,
  recurrence, and attention remain incomplete.
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
./build/bin/metalrobo_train \
  --envs 32 --steps 48 --chunk 8 --updates 100 \
  --initialize-policy g1-standing \
  --policy-pack artifacts/g1-standing.initial.policypack \
  --updated-policy-pack artifacts/g1-standing.policypack \
  --rollout-pack artifacts/g1-standing.rolloutpack
```

The native command buffer writes compact rollout streams directly into an
opaque leased ring. Swift schedules chunks and MLX consumes managed no-copy
views; neither owns simulator state.

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
