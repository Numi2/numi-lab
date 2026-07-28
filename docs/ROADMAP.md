# Heavy-lifting roadmap

The normative capability and accuracy gates live in
[ENGINE_TARGET](ENGINE_TARGET.md). This file tracks implementation order.
Items in S0 are executable foundations, not an integrated simulator claim.
The composed CPU path advances one reduced-coordinate articulation against
static/kinematic/dynamic geometry. The persistent Metal graph now executes a
contact-capable correctness slice, and free-motion ABA is exposed through an
MLX active-encoder primitive. Parallel contact scheduling and MLX contact
promotion are the next S1 boundary.

## S0 — Working foundations

- Batched Metal Franka ABA/reach environment and MLX PPO
- Canonical floating-root `nq != nv` ABI and capacity/status records
- CPU/Metal free-body integration
- CPU FP64 generalized fixed/floating-tree dynamics using world-coordinate
  CRBA plus Cholesky and RNEA, including the actual 29-DoF G1 topology
- Analytic FP64 articulated COM kinematics, point Jacobians, factor-backed
  `J`/`Jᵀ`/Delassus actions, and actual-G1 two-foot quality contact
- Transactional one-articulation CPU world from free dynamics through real
  G1 collision, evaluated ConstraintIR, factor-backed contact correction,
  common residual, integration, and atomic state/cache publication
- Production quality contact-space solve that consumes physical Delassus and
  never materializes a dense generalized inverse; dense `M⁻¹` is oracle-only
- ABI-v2 authoritative per-DoF ownership, limits, drives, friction, and
  armature; armature is executable throughout CPU dynamics/contact/energy
- Transactional disabled/model-PD/custom-PD/effort actuation with effort
  saturation, continuous-angle error, and explicit passive-loss diagnostics
- Correctness-first generic Metal fixed/floating articulation operator,
  including actual 30-body/35-velocity G1 poses, mass, point Jacobians,
  `Jᵀp`, and factor-solved `M⁻¹Jᵀp`
- Checked synchronous public Metal encoder with owned compact buffers,
  32-bit address and device-memory preflight, typed statuses, and
  transactional publication
- Pointer-free ABI-v2 constraint IR reference with ABI-v1 contact adapter,
  shared semantic evaluator, and solver-independent exact-cone residual
- CPU SAP broadphase, analytic primitive contacts, persistent manifolds
- Correct CPU/Metal collision for ten analytic/SAT pair classes through
  capsule/box and box/box
- Deterministic parallel Metal micro broadphase using flag/scan/scatter with
  explicit capacity and transactional overflow
- CPU/Metal frictional PGS contact block with warm-start cache
- Independent projected-gradient exact-cone oracle
- Safeguarded semismooth-Newton exact-cone quality solver with nonmonotone GLL
  globalization, Gauss-Newton retry, and projected-gradient safety fallback
- Transactional CPU rigid-body world pipeline
- Pinned, COM-consistent 29-DoF G1 model data
- Mixed articulation/free-body island construction and coupled exact-cone
  response in the persistent Metal graph

## S1 — Generalized articulated execution

- **Landed contact tranche:** `CompiledWorld` scene bodies/eligible pairs/
  capacity profiles, persistent twenty-one-pipeline `MetalWorldContext`,
  multi-step asynchronous submission, free-body prediction, collider
  projection, precompiled-pair broadphase, ten primitive pair classes,
  persistent four-point manifolds, shared ConstraintIR v2, factor-backed
  mixed islands, exact-cone PGS/TGS, transactional q/v/body/cache publication,
  aggregated high-water evidence, and bitwise replay
- **Landed MLX tranche:** MLX 0.32 active-encoder ABA primitive for Franka/G1,
  explicit PyTree state, compiled reset/rollback, no CPU fallback, and a
  NumPy-free policy/physics/reward/GAE/PPO rollout path
- Replace the remaining lane-zero ABA body recursion with a level-parallel
  batched tree implementation and multiple simultaneous right-hand sides
- Extend the landed parallel collider projection and eligible-pair flag queue
  through compacted SIMD32 narrowphase/manifold/island/solve queues, segmented
  scans, capacity buckets, and private scratch heaps
- Promote the contact graph to the existing MLX encoder adapter with explicit
  scene/manifold workspace arrays and no reallocation inside a lazy graph
- Compile G1 joint limits into ConstraintIR, add self-collision exclusions,
  compose actuation, coupled stiction/implicit drives, and IMU paths
- Extend the CPU transaction to multiple articulations and dynamic free
  objects through deterministic island ownership
- Preserve the working Franka API while moving it onto the generic ABI

## S2 — Production collision and throughput solve

- Extend the landed parallel micro broadphase with segmented LBVH/SAP and
  integrate it with narrowphase/manifold streams
- Add cylinder/sphere/capsule/box/cylinder, convex GJK/EPA/MPR, mesh, and
  heightfield paths; complete clipped box/box face manifolds
- Add parallel island bucketing, conflict coloring for large free-body-only
  islands, sleeping, and explicit tiled spill/replay beyond the current
  512-constraint point-query bucket
- Profile and optimize landed temporal microstep relinearization against the
  exact-cone quality oracle
- Add conservative-advancement plus speculative CCD

## S3 — Franka manipulation, then G1 locomotion

- Free-object Franka push, grasp, lift, and place
- Contact/force sensing and pinned MuJoCo/Genesis comparisons
- G1 flat-ground velocity tracking, disturbance recovery, then rough terrain
- Domain randomization, curriculum, policy export, and reproducible training

## S4 — Simulator platform

- URDF, MJCF, and OpenUSD import/cooking
- Metal viewer, contact/constraint inspection, replay and serialization
- Batched depth, segmentation, ray/LiDAR, IMU, and force sensors
- Differentiable converged dynamics with declared topology boundaries
- Broader tendons, cables, particles, and deformables behind explicit solver
  capability interfaces
