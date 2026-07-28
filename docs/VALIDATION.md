# Validation record

Snapshot: 2026-07-28.

## Machine and toolchain

- MacBook Air `Mac16,12`
- Apple M4: 10 CPU cores, 10 GPU cores, 24 GB unified memory
- macOS 26.6, Metal 4
- Apple clang 21.0.0
- MLX 0.26.5 on `Device(gpu, 0)`
- NumPy 2.2.5
- CMake Release build with four 240 Hz physics substeps per 60 Hz control step

## Clean native build

```sh
metalrobo_check_dir=$(mktemp -d /tmp/metalrobo-clean-build.XXXXXX)
cmake -S . -B "$metalrobo_check_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$metalrobo_check_dir" -j 8
```

The C++, Objective-C++, Metal 4 shader, dynamic library, and all three native
executables compiled without warnings.

## Dynamics evidence

```sh
./build/bin/metalrobo_cpu_probe
./build/bin/metalrobo_parity_probe
```

The FP64 CPU reference completed 600 control steps / 2,400 physics steps with
deterministic reset, finite state, bounded effort and velocity, and all seven
joints inside their limits.

The Metal ABA and CPU reduced dynamics were initialized from the exact same
state and advanced through one control step:

```text
max_q_error_rad=9.313226e-10
max_qd_error_rad_s=3.278255e-07
```

This narrow probe detects joint-frame, spatial-transform, gravity, drive, and
integration disagreement. It does not establish long-horizon equivalence or
contact accuracy against an independent simulator.

A separate randomized Metal soak advanced 1,024 environments for 1,000
control steps: 1,024,000 environment-steps total. Observations, rewards, body
positions, and body quaternions remained finite; maximum quaternion norm error
was `1.19e-7`. Terminal events include success and the 600-step horizon, so
their count is not a non-finite-state count.

## Native throughput

```sh
./build/bin/metalrobo_bench --envs 256  --steps 500
./build/bin/metalrobo_bench --envs 1024 --steps 500
./build/bin/metalrobo_bench --envs 4096 --steps 200
```

| Environments | Environment control-steps/s | Last GPU control step |
| ---: | ---: | ---: |
| 256 | 139,939 | 1.210 ms |
| 1,024 | 200,904 | 5.015 ms |
| 4,096 | 218,110 | 18.158 ms |

Each environment control-step includes four complete ABA/contact/integration
substeps plus observation, reward, termination, and pose work. Results are one
local run, not a cross-engine benchmark. The host submits synchronously and
the C++ benchmark also fills actions and reads rewards on the CPU.

## Real MLX PPO update

```sh
PYTHONPATH=python python3 -m metalrobo train \
  --envs 1024 \
  --rollout-steps 32 \
  --iterations 1 \
  --update-epochs 2 \
  --minibatch-size 8192 \
  --hidden-sizes 128 128 \
  --checkpoint-dir /tmp/metalrobo-final-ppo-smoke
```

This collected 32,768 transitions, executed eight clipped PPO minibatch
updates on MLX, reported finite loss/KL/gradient metrics, and wrote
`checkpoint-000001` with model, optimizer, and run state. Rollout collection
measured 117,645 environment-steps/s in that combined Python/MLX run.

## What remains unvalidated

- Trajectory/contact comparison against a pinned MuJoCo or Genesis Franka
  scene
- Pair collision, self-collision, free-body manipulation, friction cones, and
  warm-started constraints; those systems do not exist in v0.1
- Learned-policy convergence beyond a one-iteration integration run
- Cross-machine performance and reproducibility
- G1 floating-base locomotion; G1 is the next robot milestone, not part of
  this slice
