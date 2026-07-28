# MetalRobo

MetalRobo is a C++23/Metal robotics physics and reinforcement-learning runtime
for Apple silicon. It is being built as a standalone, GPU-native alternative
to the MuJoCo and NVIDIA simulation/training stacks, with Franka first and
Unitree G1 second. MLX is the learning backend. No external physics engine is
linked or called at runtime.

## Executable v0.4 engine spine

- ABI-v3, pointer-free CPU/Metal model with separate `nq`/`nv`, fixed and
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
- CPU/Metal analytic collision for sphere/sphere, sphere/plane,
  capsule/plane, box/plane, oriented cylinder/plane, sphere/capsule,
  capsule/capsule, sphere/box, capsule/box, and deterministic SAT box/box
  witnesses
- Parallel deterministic Metal micro broadphase using flag, two-level
  exclusive scan, and canonical scatter with no global append atomic
- Pointer-free constraint IR v2 with canonical validation, v1 contact
  adaptation, one timestep/material evaluator shared by quality and
  throughput consumers, and a solver-independent exact-cone residual
- Three contact paths: independent FP64 exact-cone reference, a safeguarded
  FP64 semismooth-Newton quality solve with four-merit GLL globalization,
  Gauss-Newton retry, and projected-gradient safety fallback, plus a
  CPU/Metal fixed-budget PGS block
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
- Persistent `MetalWorldContext` for one compiled articulation plus arbitrary
  dynamic/kinematic/static scene bodies: twenty-one pipelines and a typed
  grow-only arena compose ABA, body/collider projection, precompiled-pair
  broadphase, analytic narrowphase, persistent manifolds, canonical
  ConstraintIR, mixed islands, coupled exact-cone PGS/TGS, constrained
  integration, observations, and transactional publication in one
  asynchronous command buffer. A failed environment restores q/v, scene
  bodies, and manifolds while unrelated environments continue
- Exact per-environment capacity requirements, high-water counts, manifold
  retention, solver residuals, and stable first-failure indices, plus optional
  fixed-capacity contact/ConstraintIR/island evidence
- MLX 0.32 active-encoder custom primitive for Franka/G1 free-motion ABA with
  the same contact graph used by standalone Metal, explicit PyTree
  manifold/convex caches, MLX-owned output/scratch buffers, `mx.compile`,
  isolated transactional rollback, no CPU fallback, and explicit autodiff
  rejection. The Wave32 solver uses a fixed worker grid that persistently
  pulls compact packets because MLX's active encoder does not expose indirect
  dispatch. Policy inference, physics, reward/termination, GAE, rollout
  storage, and PPO updates have a NumPy-free MLX path
- Literal hybrid-CCD event splitting on standalone Metal and MLX: each
  microstep repeatedly advances to the earliest deterministic TOI cluster,
  solves it with impact-only restitution, and continues the unused time.
  Event state, manifolds, and pair caches remain transactional, and an
  uncertified remainder fails instead of silently losing time
- Pure-MLX G1 contact rollout and PPO path using implicit position drives,
  floating-root acceleration/load/contact-count sensor evidence, and
  transactional resets; rough terrain executes through the authoritative
  cooked static-mesh BVH4 contact path
- Contact-capable MLX PSM scene with the generic dynamic curved-needle asset,
  exact-CCD shape flags, persistent contact state, and a pure-array logical
  aperture-to-independent-jaw target map; a physics-owned needle hold/lift
  PPO task scores measured rigid-body pose and contact evidence without a
  weld or hidden grasp state
- Checked public Metal host boundary with owned compact buffers, overflow and
  32-bit shader-address preflight, device memory limits, typed zero-length
  bindings, per-environment statuses, and atomic result publication
- Existing batched Metal Franka ABA/reach environment and MLX PPO path
- Episodic-twin world compiler with independent semantic, render, collision,
  dynamics, and variation representations; a canonical Franka pick-and-place
  program covers appearance, object configuration, clutter, physics,
  robot/controller state, and cameras. A persistent Metal family context
  samples 4,096 compact worlds directly into private GPU buffers and exposes
  them to native/MLX graph stages without per-environment Python work

This is a serious numerical foundation, not yet a complete MuJoCo/PhysX
replacement. The device graph now has compact analytic/SAT/GJK/mesh queues,
Wave32 8/16/32-contact cohorts, deterministic tiled spill beyond 256
constraints, exact elliptic friction, private placement heaps, robust
cylinder/convex GJK-MPR-EPA, static mesh BVH4 traversal, literal hybrid-CCD
advance/solve/continue, and a contact-capable MLX primitive. Contact graph ABI
v4 carries the event-time and persistent-worker contract while ConstraintIR
remains ABI v2. Implicit position drives and joint-boundary projection are
executable. The 40,000-step/s Franka contact gate, trained 60-second G1
standing gate, dedicated tiled heightfields, multi-articulation islands,
patch rolling/torsional solve, complete joint/equality ConstraintIR,
importers, rendering breadth, thread/tissue mechanics, a Metal quality solver,
and qualified differentiation remain open. The dated requirements and claim
rules are in
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
./build/bin/metalrobo_metal_world_contact_probe
./build/bin/metalrobo_surgical_psm_probe
./build/bin/metalrobo_surgical_assets_probe
./build/bin/metalrobo_surgical_metal_operator_probe
./build/bin/metalrobo_coupled_articulated_rigid_contact_probe
./build/bin/metalrobo_articulated_rigid_collision_probe
./build/bin/metalrobo_articulated_rigid_world_probe
./build/bin/metalrobo_supported_needle_pickup_probe
./build/bin/metalrobo_world_compiler_probe
./build/bin/metalrobo_metal_world_family_probe
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
cd python
python3 probes/mlx_world_probe.py

metalrobo train \
  --backend mlx \
  --envs 1024 \
  --rollout-steps 32 \
  --iterations 1000 \
  --minibatch-size 8192
```

The native engine has no third-party physics dependency. Factual robot model
data retains its upstream notices in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). Python training requires Python
3.10+, MLX 0.32 for the device-native path, and NumPy only for the legacy
debug adapter. On the local 10-GPU-core Apple M4, the current canonical
Franka Metal-world gate measured 221,219 device-timestamped and 218,616
end-to-end wall-timed
control-steps/s for 4,096 environments over a 16-step horizon, with four
physics substeps per control step. Its three-substep FP64/Metal parity case
had maximum q error `5.753e-7`, v error `4.745e-8`, and scaled acceleration
error `1.982e-6`; same-build replay was bitwise. These are free-motion
composition numbers, not external-engine results. The 1,024-environment
Franka-plus-dynamic-cube TGS probe most recently measured 32,178 GPU and
31,346 wall
control-steps/s with two active contacts, a 32-contact capacity class, and a
246.1 MB retained arena. That is below the 40,000 release gate; the gate
remains open and a 32-active-contact saturation run is still required. The earlier
clean v0.4 validation run of the original fixed-base Franka slice measured
216,313 environment control-steps/s at 1,024
environments on a 24 GB, 10-GPU-core Apple M4, with four physics substeps per
control step. That is a local legacy-path result, not generic G1 throughput or
a cross-engine benchmark. See
[validation](docs/VALIDATION.md) for exact commands and boundaries.

## Design and research

- [Architecture](docs/ARCHITECTURE.md)
- [Persistent Metal world graph](docs/METAL_WORLD.md)
- [Real-to-sim world compiler and GPU world families](docs/WORLD_ENGINE.md)
- [v0.4 transactional generalized architecture](docs/V04_TRANSACTIONAL_ARCHITECTURE.md)
- [v0.3 operator-first architecture](docs/V03_OPERATOR_ARCHITECTURE.md)
- [State-of-the-art acceptance target](docs/ENGINE_TARGET.md)
- [Production collision design](docs/COLLISION_PIPELINE.md)
- [Pinned G1 specification](docs/G1_SPEC.md)
- [Numerical contract](docs/NUMERICS.md)
- [Competitor landscape](docs/LANDSCAPE.md)
- [Heavy-lifting roadmap](docs/ROADMAP.md)
- [Provenance](docs/PROVENANCE.md)
