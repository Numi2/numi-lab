# MetalRobo

MetalRobo is a C++23/Metal robotics physics and reinforcement-learning runtime
for Apple silicon. It is being built as a standalone, GPU-native alternative
to the MuJoCo and NVIDIA simulation/training stacks, with Franka first and
Unitree G1 second. MLX is the learning backend. No external physics engine is
linked or called at runtime.

## Executable v0.4 engine spine

- ABI-v2, pointer-free CPU/Metal model with separate `nq`/`nv`, fixed and
  floating roots, bodies, joints, one authoritative record per DoF, shapes,
  materials, contacts, and explicit capacities
- FP64 free-body dynamics plus matching Metal symplectic and implicit-midpoint
  kernels, including gyroscopic motion and SO(3) quaternion integration
- Generalized CPU FP64 articulated dynamics for fixed or floating trees with
  revolute, continuous, prismatic, and fixed joints: world-coordinate CRBA
  plus Cholesky, RNEA bias/inverse dynamics, external COM wrenches, and
  transactional SO(3) integration; the actual 29-DoF G1 topology passes
  forward/inverse consistency; per-DoF armature is included consistently in
  CRBA, RNEA, energy, contact, and impulse response
- Transactional articulated actuation with disabled, model-PD, custom-PD,
  and effort modes; effort clamping precedes passive loss, continuous-joint
  PD uses shortest-angle error, and near-zero dry friction is explicitly a
  controller-local approximation rather than falsely claimed full stiction
- Analytic articulated COM poses, twists, point Jacobians, `J`/`Jᵀ` actions,
  and factor-solve `J M⁻¹ Jᵀ` contact response; an exact-cone two-foot solve
  executes on the actual 35-velocity G1 without finite differencing
- Transactional composed CPU articulation step: free dynamics, collision,
  manifolds, canonical ConstraintIR compilation/evaluation, exact-cone solve,
  factor-backed impulse application, common residual, and SO(3) integration;
  real G1 ground contact executes end to end with atomic state/cache rollback
- Evaluated-contact-to-articulation adapter consumes one fingerprinted
  material/timestep decision, including endpoint/basis swaps and kinematic
  compensation; no solver is allowed to re-derive contact semantics
- Production quality contact solves consume physical `J M⁻¹ Jᵀ` and return
  contact velocity without materializing dense `M⁻¹`; the independent FP64
  oracle retains an explicit checked compatibility adapter
- Deterministic FP64 collision with sweep-and-prune, analytic primitive pairs,
  stable features, and persistent four-point manifolds
- Correct Metal collision baseline with analytic sphere/sphere,
  sphere/plane, capsule/plane, box/plane, and oriented cylinder/plane
  witnesses
- Parallel deterministic Metal micro broadphase using flag, two-level
  exclusive scan, and canonical scatter with no global append atomic
- Pointer-free constraint IR v2 with canonical validation, v1 contact
  adaptation, one timestep/material evaluator shared by quality and
  throughput consumers, and a solver-independent exact-cone residual
- Three contact paths: independent FP64 exact-cone reference, a safeguarded
  FP64 semismooth-Newton quality solve with four-merit GLL globalization,
  Gauss-Newton retry, and projected-gradient safety fallback, plus a
  CPU/Metal fixed-budget PGS block; the throughput block has a hard
  128-contact limit per connected island
- Transactional CPU rigid-body world step composing motion, collision,
  materials, warm starts, contact solve, and configuration integration for
  maximal-coordinate free bodies
- Pinned 29-DoF Unitree G1 model with floating COM root, full inertias,
  COM-centred joint anchors, 12 official primitive records, authoritative
  per-DoF limits, named RL Lab drive/armature data, foot frames, and IMUs;
  eight foot spheres are executable
- Open, pinned dVRK-style Patient Side Manipulator research model with a true
  prismatic insertion axis, eight driven coordinates, exact serial
  remote-center geometry, Classic Large Needle Driver, independent jaws, and
  20 executable primitive colliders, including four 0.35 mm distal teeth at
  0.8 mm longitudinal pitch; a validated seven-target policy map expands one
  logical aperture into symmetric physical jaw commands with tangent closure
  and a monotonically increasing distal surface gap
- Procedural GS-21-scale curved needle, training ring, and peg-board assets
  with stable compound colliders and geometry-derived mass, COM, inertia, and
  semantic grasp/tip zones; source facts and research defaults remain
  explicitly separated
- FP64 mixed articulation/maximal-coordinate contact operator with analytic
  articulated point Jacobians and compact dynamic-body blocks; articulated,
  dynamic, static, and kinematic endpoints share one exact circular-Coulomb
  solve with prescribed point velocity subtracted exactly once
- Transactional PSM/scene CPU world that separates force prediction from
  configuration integration, generates articulation-dynamic,
  articulation-prescribed, dynamic-dynamic, and dynamic-prescribed contacts,
  solves them with active joint stops in one exact-cone island, and persists
  generation-safe world-space warm starts
- Physics-owned PSM needle pickup from a six-button static cradle: open-jaw
  approach, legal computed-torque closure on needle segment 17, support load
  transfer, and an 8 mm off-COM lift are verified without a weld, teleport,
  or hidden attachment. Bilateral shape-17 load persists for all 2,000 lift
  frames; the finite jaw patch is load-bearing for 1,859 frames with a
  1,395-frame continuous run while resisting an 8.407 µN·m gravity moment
- Correctness-first generic Metal articulation operator for fixed/floating
  trees, exercised on actual 30-body/35-velocity G1: poses, analytic point
  Jacobians, `Jᵀp`, checked mass factorization, and `M⁻¹Jᵀp`, with
  deterministic replay and transactional rejection
- Persistent `MetalWorldContext` for one compiled canonical articulation:
  immutable model buffers and five pipelines are cached, a 27-buffer
  grow-only arena is reused, and one asynchronous command buffer advances an
  entire environment-major control horizon through reset, ABA substeps,
  transactional ping-pong commit, and q/v/acceleration observation capture.
  A failed GPU substep rolls that environment back to its control-step
  checkpoint while unrelated environments continue; no CPU-visible
  intermediate count or command-buffer wait occurs between encoded control
  steps
- Checked public Metal host boundary with owned compact buffers, overflow and
  32-bit shader-address preflight, device memory limits, typed zero-length
  bindings, per-environment statuses, and atomic result publication
- Existing batched Metal Franka ABA/reach environment and MLX PPO path

This is a serious numerical foundation, not yet a complete MuJoCo/PhysX
replacement. The first generic Metal world graph is persistent and
asynchronous, but deliberately advertises
`MetalWorldSolverMode::freeMotionABA`: it has not yet composed collider
projection, GPU collision/manifolds, contact solve, rewards, or MLX-owned
buffers. Its ABA implementation is still the deterministic lane-zero
correctness path rather than the final level-parallel tree kernel. The public
ticket publishes host vectors only after the whole rollout completes, so this
is device-resident physics across one submitted horizon—not yet a fused
physics/learner command stream.
The throughput contact kernel is PGS rather than TGS, and any connected island
above 128 contacts fails explicitly rather than spilling. Cylinder support is
currently cylinder/plane only, so G1 shoulder cylinders remain disabled.
Metal manifold persistence, LBVH, convex/mesh/heightfield geometry, CCD,
multi-articulation islands, rolling/torsional contact resistance, calibrated
surgical jaw surfaces and generic force-closure certification, the full
joint/loop constraint language, importers, rendering, sensors, tissue/thread
mechanics, and qualified differentiability remain open. The dated
requirements and claim rules are in
[ENGINE_TARGET](docs/ENGINE_TARGET.md).

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/metalrobo_cpu_probe
./build/bin/metalrobo_parity_probe
./build/bin/metalrobo_articulated_dynamics_probe
./build/bin/metalrobo_articulated_actuation_probe
./build/bin/metalrobo_articulated_contact_probe
./build/bin/metalrobo_articulated_world_probe
./build/bin/metalrobo_articulated_operator_gpu_probe
./build/bin/metalrobo_articulated_operator_host_probe
./build/bin/metalrobo_metal_world_probe
./build/bin/metalrobo_surgical_psm_probe
./build/bin/metalrobo_surgical_assets_probe
./build/bin/metalrobo_surgical_metal_operator_probe
./build/bin/metalrobo_coupled_articulated_rigid_contact_probe
./build/bin/metalrobo_articulated_rigid_collision_probe
./build/bin/metalrobo_articulated_rigid_world_probe
./build/bin/metalrobo_supported_needle_pickup_probe
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
3.10+, NumPy, and MLX. On the local 10-GPU-core Apple M4, the new canonical
Franka Metal-world gate measured 242,100 device-timestamped and 239,771
end-to-end wall-timed
control-steps/s for 4,096 environments over a 16-step horizon, with four
physics substeps per control step. Its three-substep FP64/Metal parity case
had maximum q error `5.753e-7`, v error `4.745e-8`, and scaled acceleration
error `1.982e-6`; same-build replay was bitwise. These are free-motion
composition numbers, not contact or external-engine results. The earlier
clean v0.4 validation run of the original fixed-base Franka slice measured
216,313 environment control-steps/s at 1,024
environments on a 24 GB, 10-GPU-core Apple M4, with four physics substeps per
control step. That is a local legacy-path result, not generic G1 throughput or
a cross-engine benchmark. See
[validation](docs/VALIDATION.md) for exact commands and boundaries.

## Design and research

- [Architecture](docs/ARCHITECTURE.md)
- [Persistent Metal world graph](docs/METAL_WORLD.md)
- [v0.4 transactional generalized architecture](docs/V04_TRANSACTIONAL_ARCHITECTURE.md)
- [v0.3 operator-first architecture](docs/V03_OPERATOR_ARCHITECTURE.md)
- [State-of-the-art acceptance target](docs/ENGINE_TARGET.md)
- [Production collision design](docs/COLLISION_PIPELINE.md)
- [Pinned G1 specification](docs/G1_SPEC.md)
- [Numerical contract](docs/NUMERICS.md)
- [Competitor landscape](docs/LANDSCAPE.md)
- [Heavy-lifting roadmap](docs/ROADMAP.md)
- [Provenance](docs/PROVENANCE.md)
