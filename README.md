# MetalRobo

MetalRobo is a C++23/Metal robotics physics and reinforcement-learning runtime
for Apple silicon. It is being built as a standalone, GPU-native alternative
to the MuJoCo and NVIDIA simulation/training stacks, with Franka first and
Unitree G1 second. MLX is the learning backend. No external physics engine is
linked or called at runtime.

## Executable v0.3 engine spine

- Versioned, pointer-free CPU/Metal ABI with separate `nq`/`nv`, fixed and
  floating roots, bodies, joints, shapes, materials, contacts, and capacities
- FP64 free-body dynamics plus matching Metal symplectic and implicit-midpoint
  kernels, including gyroscopic motion and SO(3) quaternion integration
- Generalized CPU FP64 articulated dynamics for fixed or floating trees:
  world-coordinate CRBA plus Cholesky, RNEA bias/inverse dynamics, external
  COM wrenches, and transactional SO(3) integration; the actual 29-DoF G1
  topology passes forward/inverse consistency
- Analytic articulated COM poses, twists, point Jacobians, `J`/`Jᵀ` actions,
  and factor-solve `J M⁻¹ Jᵀ` contact response; an exact-cone two-foot solve
  executes on the actual 35-velocity G1 without finite differencing
- Transactional common-contact-to-articulation adapter with endpoint/basis
  swaps, kinematic target compensation, and compliance conversion; real CPU
  collision/manifolds feed eight G1 foot contacts into the generalized solve
- Deterministic FP64 collision with sweep-and-prune, analytic primitive pairs,
  stable features, and persistent four-point manifolds
- Correct Metal collision baseline with analytic sphere/sphere,
  sphere/plane, capsule/plane, and box/plane witnesses
- Parallel deterministic Metal micro broadphase using flag, two-level
  exclusive scan, and canonical scatter with no global append atomic
- Pointer-free constraint IR v2 with canonical validation, v1 contact
  adaptation, one timestep/material evaluator shared by quality and
  throughput consumers, and a solver-independent exact-cone residual
- Three contact paths: independent FP64 exact-cone reference, globalized FP64
  semismooth Newton quality solve, and a CPU/Metal fixed-budget PGS block; the
  throughput block has a hard 128-contact limit per connected island
- Transactional CPU rigid-body world step composing motion, collision,
  materials, warm starts, contact solve, and configuration integration for
  maximal-coordinate free bodies
- Pinned 29-DoF Unitree G1 model with floating COM root, full inertias,
  COM-centred joint anchors, 12 official primitive records, limits, drives,
  foot frames, and IMUs; eight foot spheres are executable and four shoulder
  cylinders are explicitly disabled pending cylinder narrowphase
- Existing batched Metal Franka ABA/reach environment and MLX PPO path

This is a serious numerical foundation, not yet a complete MuJoCo/PhysX
replacement. The new parallel Metal broadphase and expanded one-thread
narrowphase are focused, separately proven components rather than an assembled
batched world. The throughput contact kernel is PGS rather than TGS, and any
connected island above 128 contacts fails explicitly rather than spilling.
The generalized CPU articulated-contact operator is not yet a batched Metal
articulated step or locomotion environment. Metal manifold persistence,
LBVH, convex/mesh/heightfield geometry, CCD, the complete joint/loop constraint
language, importers, rendering, sensors, and qualified differentiability
remain open. The dated requirements and claim rules are in
[ENGINE_TARGET](docs/ENGINE_TARGET.md).

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/metalrobo_cpu_probe
./build/bin/metalrobo_parity_probe
./build/bin/metalrobo_articulated_dynamics_probe
./build/bin/metalrobo_articulated_contact_probe
./build/bin/metalrobo_g1_collision_contact_probe
./build/bin/metalrobo_free_body_gpu_probe
./build/bin/metalrobo_collision_gpu_probe
./build/bin/metalrobo_deterministic_broadphase_probe
./build/bin/metalrobo_constraint_ir_probe
./build/bin/metalrobo_quality_contact_probe
./build/bin/metalrobo_rigid_body_world_probe
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

The native engine has no third-party physics dependency. Factual robot model
data retains its upstream notices in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). Python training requires Python
3.10+, NumPy, and MLX. A clean v0.2 build of the original fixed-base Franka
slice measured 195,900 environment control-steps/s at 1,024 environments; an
earlier local run reached 218k/s at 4,096 environments on a 24 GB,
10-GPU-core Apple M4, with four physics substeps per control step. Those are
local results, not cross-engine benchmarks. See
[validation](docs/VALIDATION.md) for exact commands and boundaries.

## Design and research

- [Architecture](docs/ARCHITECTURE.md)
- [v0.3 operator-first architecture](docs/V03_OPERATOR_ARCHITECTURE.md)
- [State-of-the-art acceptance target](docs/ENGINE_TARGET.md)
- [Production collision design](docs/COLLISION_PIPELINE.md)
- [Pinned G1 specification](docs/G1_SPEC.md)
- [Numerical contract](docs/NUMERICS.md)
- [Competitor landscape](docs/LANDSCAPE.md)
- [Heavy-lifting roadmap](docs/ROADMAP.md)
- [Provenance](docs/PROVENANCE.md)
