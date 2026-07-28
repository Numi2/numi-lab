# Heavy-lifting roadmap

The normative capability and accuracy gates live in
[ENGINE_TARGET](ENGINE_TARGET.md). This file tracks implementation order.
Items in S0 are executable foundations, not an integrated simulator claim.
The composed CPU path advances one reduced-coordinate articulation against
static/kinematic/dynamic geometry. The persistent Metal graph and MLX
active-encoder primitive now execute the same contact graph, including
persistent Wave32 queues and literal event-time CCD splitting. The next
boundary is task closure and throughput, not another disconnected physics
prototype.

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
- **Landed execution tranche:** stable compact work packets, persistent MLX
  Wave8/16/32 pulling, deterministic tiled spill through distributed large
  islands, and exact circular/elliptical friction using matrix-free
  articulation/free-body response
- **Landed CCD tranche:** contact ABI v4 literal multi-event
  advance/solve/continue, deterministic simultaneous-impact clustering,
  impact-only restitution, zero-time replay limits, complete-time accounting,
  and per-environment transactional failure
- **Landed task plumbing:** implicit position drives, joint-boundary
  projection, floating-root acceleration/contact evidence, a pure-MLX G1
  standing/planar/yaw-command PPO collector, cooked-BVH4 rough terrain, and
  an MLX PSM plus dynamic curved-needle hold/lift PPO task with physics-owned
  contact
- Replace the remaining lane-zero ABA body recursion with a level-parallel
  batched tree implementation and multiple simultaneous right-hand sides
- Compile complete unilateral joint-limit warm starts and equality/loop
  constraints into shared ConstraintIR; finish self-collision exclusions,
  coupled stiction, and calibrated rolling/torsional patch blocks
- Extend the CPU transaction to multiple articulations and dynamic free
  objects through deterministic island ownership
- Preserve the working Franka API while moving it onto the generic ABI

## S2 — Production collision and throughput solve

- Extend the landed compiled-pair broadphase with segmented Morton/LBVH for
  heterogeneous scenes
- Keep the landed analytic/cylinder/convex GJK-MPR-EPA and static-mesh BVH4
  stack authoritative; add a dedicated tiled min/max heightfield path,
  compound cooking, and dynamic convex decomposition
- Add deterministic sleeping/wake compaction and tune the landed island
  buckets/distributed spill using M4 counter captures
- Profile and optimize landed temporal microstep relinearization against the
  exact-cone quality oracle
- Broaden the landed hybrid CCD corpus to articulated fast grippers,
  multi-body event clusters, and needle-scale curved geometry

## S3 — Franka manipulation, then G1 locomotion

- Close free-object Franka push, grasp, lift, 30-second hold, sliding, and peg
  insertion through the MLX graph
- Finish per-link force/torque sensor channels and pinned MuJoCo/Genesis
  comparisons
- Train and publish G1 flat-ground standing/velocity tracking/disturbance
  recovery, then use the landed mesh terrain path while the dedicated
  heightfield implementation is completed
- Domain randomization, curriculum, policy export, and reproducible training

## S4 — Multiple articulations and surgical autonomy

- Generalize island nodes and factor caches to multiple articulations per
  environment, including articulation-articulation contacts and loop blocks
- Compose two PSMs, trocar/RCM constraints, calibrated jaw patches, and
  physics-owned needle regrasp/ring/peg transfer without hidden attachments
- Add discrete-elastic-rod thread stretch/bend/twist, attachment, self/tool
  collision, and its batched block-tridiagonal inner solve
- Defer calibrated tissue puncture/cutting until rigid needle/thread evidence
  is stable and reproducible

## S5 — Quality, learning, and platform

- Add matrix-free Metal semismooth Newton with ABA Hessian-vector products,
  block-PCG, line search, and residual certificates
- Add implicit-adjoint MLX JVP/VJP only with explicit validity masks across
  topology, clipping, sleeping, and CCD event changes

- URDF, MJCF, and OpenUSD import/cooking
- Metal viewer, contact/constraint inspection, replay and serialization
- Batched depth, segmentation, ray/LiDAR, IMU, and force sensors
- Broader tendons, cables, particles, and deformables behind explicit solver
  capability interfaces
