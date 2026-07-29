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
  device-native reset-state randomization, plus an MLX PSM dynamic
  curved-needle hold/lift PPO task with physics-owned contact
- **Landed generalized multi-articulation tranche:** deterministic cooked ABA
  schedules/Jacobians/row packets, matrix-free Metal inverse-mass and
  Delassus assembly, residual-certified bounded solve, a persistent
  asynchronous standalone context, and a transactional MLX active-encoder
  primitive demonstrated on dual PSM plus G1
- **Landed parallel inverse-ABA tranche:** SIMD32 forward/reverse frontiers,
  stable parent-owned sibling reductions, shared factorization across RHS
  packets, and identical standalone/MLX execution on branching dual-PSM-plus-
  G1 worlds without a dense generalized inverse
- **Landed spatial-row frontend:** the Metal articulated operator can now emit
  body poses, point positions and analytic point Jacobians without assembling
  or factorizing a mass matrix. Multi-articulation contact can therefore feed
  those rows into the shared parallel inverse-ABA response stage without
  duplicating factor ownership
- **Landed standalone heterogeneous contact solve:** explicit contact rows
  spanning multiple articulations, dynamic scene-body 6D blocks and
  static/kinematic boundaries now flow through one Metal command buffer into
  the exact-cone quality solver, with deterministic replay and isolated
  per-environment rollback. Immutable model/frontier compilation is reusable,
  and the dual-PSM/needle bundle executes as a 34-DoF device island
- **Landed quality certification hardening:** Metal cone success requires both
  the semismooth natural residual and an explicit normalized primal/dual/
  complementarity KKT certificate; stiff cold starts fail rather than
  publishing a false zero-impulse solution
- **Landed coupled equality/contact solve:** FP64 and Metal
  multi-articulation contact eliminate model-owned unbounded equality and gear
  rows through a small Schur complement assembled with articulation-local
  inverse ABA. Dual-PSM base locks, jaw gears and needle contacts share one
  operator, with final equality impulses, null-space leakage and residual
  evidence
- **Landed heterogeneous point-loop frontend:** authored three-axis
  translational fixtures spanning articulations, dynamic scene bodies, and
  static/kinematic boundaries compile analytic point Jacobians directly into
  the coupled equality/contact operator. Free bodies contribute their full 6D
  inverse response and kinematic point motion shifts row targets.
  Environment-varying semantics stay in canonical ConstraintIR rows; the
  device graph solves and certifies them without a dense mass inverse
- **Landed angular/spatial equality frontend:** three-axis relative angular
  rows reconstruct each body angular Jacobian analytically from four
  body-local point queries. Translational and angular blocks can therefore
  form a full spatial weld spanning articulations, free bodies, and
  static/kinematic boundaries while sharing the contact Schur solve
- Move the actuation/free-motion ABA step itself from lane-zero recursion onto
  the cooked frontiers, then reuse its per-microstep factor cache across
  actuation, generalized constraints, and contact response
- Compile complete unilateral joint-limit warm starts and quaternion-derived
  orientation-error authoring into shared ConstraintIR;
  finish self-collision exclusions, coupled stiction, and calibrated
  rolling/torsional patch blocks
- Scatter persistent manifold endpoints into the landed Metal row frontend,
  then move the same encoder sequence into the private-buffer standalone
  context and MLX active encoder
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

- **Landed composition/force-transfer tranche:** two independently based PSM
  articulations, generalized base/jaw mechanism rows, dynamic curved needle,
  physically parameterized DER stretch/bend/twist, geometry-derived swage
  binding, fixed-slot attachment reaction evidence, and same-command-buffer
  equal/opposite needle wrench with transactional publication
- **Landed heterogeneous ownership tranche:** one fingerprinted executable
  bundle now owns composed robot/free-body topology, scene reset states, rod
  models/states and rigid bindings; the dual-PSM/needle/thread factory feeds
  the existing multi-articulation compiler and Metal DER path directly
- **Landed coupled-contact oracle:** FP64 articulation-articulation,
  self-contact and articulation-static point Jacobians assemble one global
  exact-cone contact problem through retained per-articulation factors. The
  operator builds `M^-1 J'` response columns and `J M^-1 J'` without a dense
  global mass inverse, then reconstructs generalized velocity from the
  converged quality solve transactionally. Dynamic scene bodies now append
  6D maximal-coordinate blocks, while static/kinematic endpoints contribute
  prescribed point velocity. The owned dual-PSM/needle bundle executes as one
  34-DoF coupled contact island in this oracle
- Generalize the shared contact island graph to articulation-articulation
  contacts, non-articulated 6D scene-body endpoints and loop blocks on Metal
  using the landed oracle's sign/frame/material semantics; the
  multi-articulation ABA/ConstraintIR primitive itself is already executable
  on standalone Metal and MLX
- Add trocar/RCM constraints, calibrated jaw patches, and physics-owned needle
  regrasp/ring/peg transfer without hidden attachments
- Add thread self/tool collision and replace the current SIMD32 DER projection
  with a measured batched block-tridiagonal rod solve where it wins
- Defer calibrated tissue puncture/cutting until rigid needle/thread evidence
  is stable and reproducible

## S5 — Quality, learning, and platform

- **Landed generalized-quality tranche:** GPU semismooth Newton for scalar
  bilateral/unilateral/bounded ConstraintIR, diagonally scaled natural maps,
  normal-equation CG, safeguarded line search, projected-gradient fallback,
  and physical residual certification on standalone and MLX graphs
- Replace the materialized contact-space Delassus in quality mode with direct
  ABA Hessian-vector products and block-PCG, then unify exact cone and scalar
  active-set blocks in one island quality solve
- Add implicit-adjoint MLX JVP/VJP only with explicit validity masks across
  topology, clipping, sleeping, and CCD event changes

- **Landed URDF/SRDF executable cooker:** deterministic fixed/floating trees,
  COM-centred inertials/anchors, primitive and capsule colliders, limits,
  damping/friction/armature, transmission-selected actuation, mimic gear
  ConstraintIR, SRDF passive joints/exclusions, fingerprints, and
  transactional publication
- **Landed articulated mesh bridge:** deterministic relative, absolute-file,
  and `package://` resolution; OBJ and binary/ASCII STL ingestion; content
  fingerprints; canonical asset deduplication; scale-aware bounds; and direct
  cooking into the existing convex half-edge arenas
- Add deterministic convex decomposition for authored concavity and DAE
  ingestion, then MJCF; defer OpenUSD workflow breadth
- Metal viewer, contact/constraint inspection, replay and serialization
- Batched depth, segmentation, ray/LiDAR, IMU, and force sensors
- Broader tendons, cables, particles, and deformables behind explicit solver
  capability interfaces
