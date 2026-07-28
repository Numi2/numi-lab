# Heavy-lifting roadmap

The normative capability and accuracy gates live in
[ENGINE_TARGET](ENGINE_TARGET.md). This file tracks implementation order.
Items in S0 are executable foundations, not an integrated simulator claim:
the generalized articulated reference is a standalone CPU path, while the
composed contact world currently advances maximal-coordinate free bodies.

## S0 — Working foundations

- Batched Metal Franka ABA/reach environment and MLX PPO
- Canonical floating-root `nq != nv` ABI and capacity/status records
- CPU/Metal free-body integration
- CPU FP64 generalized fixed/floating-tree dynamics using world-coordinate
  CRBA plus Cholesky and RNEA, including the actual 29-DoF G1 topology
- Analytic FP64 articulated COM kinematics, point Jacobians, factor-backed
  `J`/`Jᵀ`/Delassus actions, and actual-G1 two-foot quality contact
- Transactional common-contact adapter and real collision/manifold to
  eight-foot-contact generalized G1 quality solve
- Pointer-free ABI-v2 constraint IR reference with ABI-v1 contact adapter,
  shared semantic evaluator, and solver-independent exact-cone residual
- CPU SAP broadphase, analytic primitive contacts, persistent manifolds
- Correct Metal `O(n²)` collision baseline for sphere/sphere, sphere/plane,
  capsule/plane, and box/plane
- Deterministic parallel Metal micro broadphase using flag/scan/scatter with
  explicit capacity and transactional overflow
- CPU/Metal frictional PGS contact block with warm-start cache
- Independent projected-gradient exact-cone oracle
- Globalized semismooth-Newton exact-cone quality solver
- Transactional CPU rigid-body world pipeline
- Pinned, COM-consistent 29-DoF G1 model data
- Throughput-island partitioning with an explicit 128-contact limit for each
  connected island

## S1 — Generalized articulated execution

- Port generalized floating ABA and state integration to batched Metal
- Port the executable CPU articulated contact Jacobian and factor-solve
  impulse operator to Metal without materializing dense inverse mass
- Connect the FP64 generalized reference to collision, joint limits, contact
  constraints, and the composed step; the current CPU world is
  maximal-coordinate only
- Connect G1 limits, drives, self-collision exclusions, feet, and IMUs
- Preserve the working Franka API while moving it onto the generic ABI

## S2 — Production collision and throughput solve

- Extend the landed parallel micro broadphase with segmented LBVH/SAP and
  integrate it with narrowphase/manifold streams
- Add cylinder, general capsule/box, convex GJK/EPA/MPR, mesh, and heightfield
  paths
- GPU manifold refresh/reduction, stable friction patches, graph coloring,
  island bucketing, sleeping, and explicit spill/replay
- Replace the current hard 128-contact ceiling for one connected throughput
  island with size buckets and explicit spill/replay; independent islands are
  already partitioned on the composed CPU path
- Implement actual temporal Gauss-Seidel with substep relinearization
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
