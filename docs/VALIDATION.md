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

The C++, Objective-C++, Metal 4 shaders, dynamic library, and configured native
probes compiled without warnings.

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

A separate randomized Franka Metal soak advanced 1,024 environments for 1,000
control steps: 1,024,000 environment-steps total. Observations, rewards, body
positions, and body quaternions remained finite; maximum quaternion norm error
was `1.19e-7`. Terminal events include success and the 600-step horizon, so
their count is not a non-finite-state count. This is evidence for the original
fixed-base Franka runtime, not G1.

## Generic free-body dynamics

```sh
./build/bin/metalrobo_free_body_probe
./build/bin/metalrobo_free_body_gpu_probe
```

The FP64 implicit-midpoint oracle advanced a torque-free anisotropic body for
10,000 steps at 1 kHz:

```text
q_norm_error=5.115205e-09
angular_momentum_drift=3.065610e-06
max_newton_iterations=2
max_residual=8.295525e-16
```

The Apple M4 Metal probe covers static, kinematic, and anisotropic dynamic
bodies under both integrators:

```text
implicit_max_error=2.384186e-07
symplectic_max_error=2.384186e-07
energy_drift=4.951812e-08
angular_momentum_drift=2.890093e-08
midpoint_iterations=2
motion_contract=yes
range_preflight=yes
```

These are focused convention/conservation probes, not a claim of high-order
accuracy under contact.

## Generalized articulated CPU reference

```sh
./build/bin/metalrobo_articulated_dynamics_probe
```

The generalized reference uses FP64 intermediates, an actual
world-coordinate composite-rigid-body recursion, dense Cholesky, and
recursive Newton-Euler bias/inverse dynamics. It accepts fixed or floating
trees with revolute, continuous, or fixed joints. Floating state is
`nq != nv`: root COM position plus quaternion in `q`, and world linear plus
angular COM velocity in `v`. External forces and torques are applied at body
COMs.

The probe covers an analytical anisotropic free body with external wrench and
gravity, an analytical one-link pendulum, an eight-velocity floating chain,
the compiled branched 30-body/29-joint G1 topology, transactional
limit/non-finite rejection, and a 400-step unforced implicit-midpoint
conservation run:

```text
free_body_error=4.440892e-16
pendulum_error=5.329071e-15
forward_inverse_error=8.881784e-16
mass_symmetry_error=0
g1_forward_inverse_error=7.170653e-14
energy_drift=5.569560e-11
linear_momentum_drift=8.940619e-10
angular_momentum_drift=2.019079e-09
q_norm_error=0
midpoint_iterations=2
```

This establishes internal analytical and forward/inverse consistency for the
CPU reference, including the actual G1 topology and inertial records. It does
not establish agreement with an external simulator, articulated contact,
long-horizon G1 stability, or Metal G1 execution. The reference is not yet
wired into the composed rigid-body world.

## Collision and manifolds

```sh
./build/bin/metalrobo_collision_probe
./build/bin/metalrobo_collision_gpu_probe
```

The FP64 path generated deterministic sweep-and-prune pairs, analytic
sphere/sphere plus sphere/capsule/box-to-plane witnesses, and persistent
four-point manifolds. The corpus reported 29 SAP pairs with zero false
negatives, stable IDs, `8_to_4` deterministic manifold reduction, canonical
filters, and transactional overflow.

The Metal correctness baseline matched sorted CPU sphere pairs/witnesses:

```text
shapes=13 pairs=7 raw_contacts=6
max_witness_error=5.960464e-08
canonical_filters=yes stable_features=yes deterministic_replay=yes
overflow_transactional=yes finite_validation=yes
```

This Metal kernel is deliberately one-thread `O(n²)` and is not a production
GPU broadphase throughput result.

## Contact solver portfolio

```sh
./build/bin/metalrobo_reference_conic_probe
./build/bin/metalrobo_quality_contact_probe
./build/bin/metalrobo_contact_gpu_probe
```

The independent accelerated projected-gradient oracle reached an exact-cone
optimality residual of `4.939340e-12`. The globalized FP64 semismooth-Newton
quality path solved a coupled four-contact problem in six iterations:

```text
kkt=1.513167e-12
primal_cone_violation=9.464532e-18
dual_cone_violation=3.768652e-12
reference_impulse_error=4.741880e-10
permutation_impulse_error=5.958796e-16
warm_iterations=0
```

The fixed-budget CPU/Metal PGS block agreed on a coupled two-contact stack:

```text
max_linear_error=7.450581e-08
max_angular_error=1.192093e-07
max_impulse_error=2.980232e-08
max_cone_violation=0
momentum_xy_error=2.215217e-07
arithmetic_rollback=yes
```

The PGS block uses radial friction projection and is neither TGS nor the
exact effective-mass-metric quality solve. One throughput solver dispatch is
hard-limited to 128 contacts. The composed CPU world partitions independent
constraint islands before dispatch, so the operative ceiling is 128 contacts
for any one connected island; an oversized connected island reports explicit
capacity overflow. The current Metal kernel has the same per-dispatch limit,
and its caller must provide the island partition.

The Metal contact kernel preflights validation and capacity errors and backs
up only touched dynamic velocities plus contact impulses/flags. The
`arithmetic_rollback` case forces the second warm-start impulse to overflow
after the first contact has already mutated state, then verifies exact body
and contact restoration. Static endpoints are excluded from backup/restore so
independent islands can share static geometry without a write race.

## Composed rigid-body world

```sh
./build/bin/metalrobo_rigid_body_world_probe
```

The transactional CPU pipeline held a two-sphere stack for 1,200 steps with
two persistent warm-started contacts and deterministic bitwise replay. The
same pipeline ran 240 steps through the semismooth quality solver:

```text
penetration_max=0
bottom_y=0.5 top_y=1.5
quality_kkt_max=4.812123e-17
manifold_constraints=4
rest_offsets=yes
island_batched_contacts=129
quality_friction_rejection=transactional
overflow_transactional=yes
```

This validates composition and signs for a narrow maximal-coordinate scene.
The world advances independent rigid bodies through free motion and resolves
their contacts; it does not invoke the generalized articulated CRBA/RNEA
path. It is not an articulated-contact or impact/CCD benchmark.

## Pinned G1 compilation

```sh
./build/bin/metalrobo_g1_model_probe
```

The compiled mode-machine-5 model contains 30 dynamic bodies, 29 joints,
`nq=36`, `nv=35`, 33.34114204 kg total mass, 12 official primitive records,
two IMUs, full positive-definite inertia tensors, COM-centred runtime anchors,
and a COM root reset at `z=0.72396994`. Eight foot spheres are executable; the
four shoulder cylinders are retained but simulation-disabled until cylinder
narrowphase exists.

## Native throughput

```sh
./build/bin/metalrobo_bench --envs 256  --steps 500
./build/bin/metalrobo_bench --envs 1024 --steps 500
./build/bin/metalrobo_bench --envs 4096 --steps 200
```

| Environments | Environment control-steps/s | Last GPU control step |
| ---: | ---: | ---: |
| 256 | 139,939 | 1.210 ms |
| 1,024 | 195,900 | 4.381 ms |
| 4,096 | 218,110 | 18.158 ms |

Each environment control-step includes four complete ABA/contact/integration
substeps plus observation, reward, termination, and pose work. Results are one
local machine snapshot; the 1,024 row is the final clean v0.2 run and the
other rows are earlier runs on the same machine. This is not a cross-engine
benchmark. The host submits synchronously and the C++ benchmark also fills
actions and reads rewards on the CPU.

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
- Parallel Metal broadphase/narrowphase performance; the current generic GPU
  collision path is a correctness baseline
- Metal persistent-manifold refresh/reduction; the executable persistence
  path is CPU
- Capsule/box/cylinder Metal narrowphase, convex/mesh/heightfield collision,
  articulated self-collision, and CCD
- Coupling the generalized CPU articulated solver to the composed contact
  world
- Batched Metal floating-articulation execution and long-horizon G1 dynamics
- Connected throughput islands above 128 contacts and production spill/replay
- TGS; the current throughput solver is correctly identified as PGS
- Learned-policy convergence beyond a one-iteration integration run
- Cross-machine performance and reproducibility
- G1 locomotion training; the physical model is compiled but not yet wired to
  a batched Metal environment
