# Numerical contract

## Coordinates and units

- Positions use metres, time seconds, mass kilograms, angles radians, and
  forces/torques SI units.
- Canonical rigid-body position and linear velocity are measured at the center
  of mass. Orientation is the body/link-frame orientation.
- Joint anchors are expressed from each body's COM, even when source robot
  data describes a joint relative to a URDF link origin.
- Quaternions use `(x, y, z, w)`, represent body-to-world rotation, and are
  normalized after composition.
- A floating articulation uses root COM `xyz` plus quaternion in `q`
  (`nq = 7 + joint nq`) and world COM linear velocity plus world angular
  velocity in `v` (`nv = 6 + joint nv`).
- Authored `InteractionPack` root targets, task root observations, and visual
  transforms use the root-link origin. Reset converts a target to generalized
  coordinates as `COM_world = link_world + R * COM_local`; state publication
  converts back as `link_world = COM_world - R * COM_local`. These conversions
  are mechanism boundaries, never presentation offsets or floor correction.
- Contact normals point from body A to body B. Geometric witnesses retain
  separate points on A and B and signed separation.

## Precision boundary

Metal physics is FP32 because Metal shaders do not provide native `double`.
The canonical ABI and compiled robot inertial records are FP32. CPU reference
paths promote those records and evaluate dynamics, collision, and quality
solver intermediates in FP64; promotion does not recover precision absent
from the compiled record. Collision admission uses direct FP32 source-field
checks rather than comparing independently rounded derived bounds at the same
hard threshold. Accepted geometry is then promoted for the FP64 narrowphase.

Every CPU reference path validates dimensions and finite inputs before
publication. Capacity, factorization, convergence, and unsupported-feature
failures have explicit status codes. The composed CPU maximal-coordinate and
one-articulation world paths are transactional: state and persistent caches
remain unchanged when a step fails. The bounded Metal contact kernel backs up
the dynamic velocities and contact records it can mutate, then restores them
after an arithmetic failure.
A fully contact-composed production Metal world remains open, but the
free-motion orchestration layer now has versioned dispatch/status buffers.
The standalone generic articulated operator retains its checked synchronous
host boundary. `MetalWorldContext` adds derived compact horizon strides,
32-bit element-address limits, actual allocation and working-set checks,
typed zero-length bindings, cached immutable topology, a grow-only arena, and
asynchronous multi-control-step execution.

Each control step takes an immutable q/v checkpoint after applying its reset.
ABA produces candidate state only. A successful substep is copied into the
accepted ping-pong state; the first failed substep latches its typed failure
and all remaining passes restore the checkpoint. Failed observations are
finite checkpoint state and their acceleration is exactly zero. Host
validation or malformed GPU output leaves the caller's previous result
unchanged. This is per-environment transactionality across one device
horizon; collision manifolds and impulses are not yet members of that
transaction.

## Articulated dynamics

The generalized CPU reference supports fixed or floating trees containing
revolute, continuous, and fixed joints. It:

- forms a dense FP64 mass matrix with a world-coordinate
  composite-rigid-body recursion;
- verifies positive definiteness and solves forward dynamics with Cholesky;
- computes velocity, gyroscopic, gravity, damping, and external-wrench terms
  through recursive Newton-Euler kinematics and analytic generalized-force
  projection;
- treats per-DoF armature as generalized inertia in CRBA, RNEA, invariant
  energy, contact effective mass, and impulse response;
- integrates with symplectic Euler or a converged implicit-midpoint solve;
- composes floating orientation with the SO(3) exponential of world angular
  velocity.

The FP64 articulated-actuation evaluator consumes the same immutable per-DoF
stream. It supports disabled, named-model PD, command-local PD, and direct
effort modes; forbids floating-root actuation; evaluates feed-forward plus PD;
uses the shortest signed modulo-\(2\pi\) error for continuous joints; clamps
actuator effort before applying passive dry friction; and publishes results
transactionally. Moving Coulomb friction strictly opposes velocity. Inside
the configured near-zero-speed band it can cancel only the local actuator
load, because gravity, bias, external, and contact loads are not inputs to
this evaluator. That branch is a controller-local approximation, not a
complete set-valued stiction solve.

Optional coordinate and body-speed limits are validation boundaries. They
reject an invalid state transactionally; they are not yet unilateral
joint-limit constraints that generate impulses. The actual G1 topology passes
the internal FP64 analytical and forward/inverse probes. Analytic point
Jacobians and a retained CRBA factor now provide `J`, `Jᵀ`, and
`J M⁻¹ Jᵀ` contact actions. The transactional CPU world composes this with
collision, evaluated ConstraintIR, exact-cone contact, the common residual,
and integration for G1 ground contact. A correctness-first Metal operator
executes the same G1 mass/Jacobian/impulse equations, but a batched parallel
Metal timestep remains open.

The original Franka runtime remains a separate compatibility API. The
canonical Metal world now reuses the generic FP32 articulated-body kernel and
checks multi-step q/v/acceleration against the FP64 generalized oracle. On the
same device and build its complete output/status stream replays bitwise.
Neither internal agreement is an external-simulator accuracy promise.

## Free-body integration

Independent free bodies include the gyroscopic term
`omega × (I omega)`. Symplectic Euler is the throughput-oriented option.
Implicit midpoint solves the anisotropic angular update nonlinearly and is a
conservation-oriented reference. Both CPU and Metal use SO(3) exponential
quaternion composition. Neither integrator alone supplies collision
time-of-impact handling.

## Collision and contact

The CPU collision oracle uses FP64 sweep-and-prune, analytic primitive
witnesses, stable feature identifiers, and deterministic four-point manifold
reduction. The Metal collision path is currently an FP32, one-thread `O(n²)`
correctness narrowphase baseline for sphere/sphere, sphere/plane,
capsule/plane, box/plane, and oriented cylinder/plane. A separate
deterministic parallel micro
broadphase uses flag/scan/scatter without global append atomics. The two are
not yet a production LBVH/manifold stream, and Metal does not perform
persistent-manifold refresh.

The executable collision ABI has two deliberate numerical domains. Authored
body/local positions, primitive dimensions, and contact/rest/bounding-radius
fields must lie in `[-100,000, 100,000]` metres. Derived transforms and finite
AABBs have a separate `[-1,000,000, 1,000,000]` metre sanity domain. For
normalized supported primitives, the first domain proves a derived bound
below roughly `5.5 * 100,000` metres, leaving deterministic slack instead of
placing backend-specific FP32/FP64 arithmetic on an admission boundary.
Active non-plane dimensions must be at least `1e-9` metres. Nonzero FP32
subnormals are rejected from collision records and external AABBs by raw-bit
classification, so Metal flush-to-zero cannot change acceptance. Larger
worlds must rebase each environment. These checks apply before shape-type
classification and use the same status policy on CPU and Metal.

Quaternion scale is not physical. Collision records admit canonical finite
components when `max(abs(q))` is in `[0.25, 1.001]`, then normalize. This
direct component contract admits every unit orientation and modest authored
drift without putting a hard decision on an FMA-sensitive `dot(q,q)`
tolerance.

Broadphase bounds are conservative, not nearest-rounded. CPU bounds start
from FP64 geometry, receive a scale-aware `64 * FLT_EPSILON` outward pad, and
are cast with directed `nextafter` correction. Metal applies the same
scale-aware outward pad. Contact eligibility on Metal uses a matching
roundoff band, making the GPU result a conservative superset near an exact
FP32/FP64 threshold. Contacts outside that band must agree within witness
tolerances; inside it, bounded speculative contacts are permitted but missing
an oracle contact is not.

The core rigid collision path does not implement CCD. Fast rigid bodies can
therefore tunnel; substeps are not a semantic substitute for conservative
advancement, speculative CCD, or time-of-impact island stepping. Matter's
deformable surface path is a separate owner: it rebuilds swept FEM triangles
and applies conservative-advancement vertex-triangle and edge-edge CCD inside
its nonlinear transaction.

The contact portfolio has three distinct numerical contracts:

- an independent FP64 accelerated projected-gradient exact circular-cone
  oracle;
- a safeguarded FP64 semismooth-Newton quality solve with overflow-safe
  residual/KKT scaling, a four-merit GLL globalization, bounded direct-Newton
  search, Gauss-Newton retry, projected-gradient safety fallback, and
  KKT/cone diagnostics, accepting either a legacy dense oracle problem or a
  production contact-space Delassus problem;
- a fixed-budget CPU/Metal PGS throughput block with normal, coupled
  two-tangent radial projection, torsional friction, and warm starts.

The throughput block is PGS, not temporal Gauss-Seidel: it does not advance
and relinearize contacts through internal TGS substeps. Rolling resistance is
explicitly unsupported. One throughput dispatch holds at most 128 contacts.
The composed CPU world partitions independent connected islands, so any one
connected island above 128 contacts returns capacity overflow. It does not
drop the excess contacts.

The Metal temporal-cone block symmetrizes and scale-normalizes each coupled
3x3 point response before inversion. Its deterministic CFM floor is one
percent of the dominant response. This bounds amplification from redundant or
nearly rank-deficient articulated contacts while leaving the full articulated
mass factor and response construction unchanged; it is not a post-step
velocity clamp.

The currently composed quality world accepts one isotropic Coulomb
coefficient. A material with distinct static and dynamic coefficients,
torsional/rolling friction, or a hard impulse cap returns `MR_STEP_UNSUPPORTED`
transactionally instead of changing material semantics when solver mode
changes. A shared convex stiction model remains open.

The reduced-coordinate quality path constructs the physical contact-space
operator `W = J M⁻¹ Jᵀ` with checked factor solves. It does not materialize
`M⁻¹`. Solver-reported contact velocity is compared with `J` applied to the
independently factor-corrected generalized velocity before the common
ConstraintIR residual can accept the step. The dense inverse adapter exists
only for the independent FP64 oracle.

Metal contact dispatch validates inputs and capacities before its in-place
solve and rolls back touched dynamic velocities, impulses, and warm-start
flags after an arithmetic failure. Static/kinematic endpoints are never
written during solve or rollback, allowing independent islands to share static
geometry safely.

## Matter generalized continuum KKT

Matter's continuum solve is one environment-wide Newton system. Its
matrix-free generalized unknown packs FEM velocity and mixed pressure,
thermal/pore/electric/activation fields, and active sparse MPM grid velocity.
Contact uses a primal logarithmic distance barrier and contributes gradients
and PSD-projected rank-one normal curvature directly to those mechanical
blocks. Restarted flexible GMRES
uses compensated SIMD32 reductions, selective reorthogonalization, device
Givens rotations, and an inexact-Newton forcing schedule. The right
preconditioner combines fine node-star mechanics blocks, overlapping
connectivity-aware tetrahedron-patch corrections, an object-scale Galerkin
translation/mean-pressure correction, a fixed-pass field polynomial smoother,
MPM lumped-mass blocks, and barrier curvature. FGMRES is the only linear
iteration owner; the patch and field smoothers
have no independent convergence or publication contract.

All coupled objects in one environment use the minimum admissible determinant
mixed-volume backtracking, conservative CCD, and a barrier
fraction-to-boundary cap. FEM surface
adjacency is cooked, while current and cohesive surface primitives, swept
bounds, stable Morton ordering, non-adjacent self-contact candidates, CCD
witnesses, and active barrier pairs are rebuilt on Metal. Stable compacted pairs
also produce deterministic contact-node CSR and a cross-environment indirect
work list; node gathers visit only incident rows and downstream contact kernels
dispatch only active work without a CPU synchronization. Deformable warm
starts store only stable source/frame and lagged-friction state and participate in
the same checkpoint/commit/rollback transaction as nodes, fields, topology,
materials, schedulers, and rigid contact warm starts.

Matter Language distinguishes accepted `state` from `next(state)`. An authored
`implicit state = residual;` declaration compiles local residual, pivoted
Jacobian, deformation-action, and stress-state derivative bytecode. Each
particle/tetrahedron executes damped bounded local Newton and the global operator uses
the consistent action `P_F - P_z R_z^-1 R_F`. Explicit `update` remains a
supported compatibility path. `model von_mises` and
`model drucker_prager` select multiplicative finite-strain elastic-predictor /
plastic-corrector policies. They require row-major `plastic_f00` through
`plastic_f22` state initialized to identity plus
`equivalent_plastic_strain transfer max`; the compiler rejects an incomplete
layout. Their radial return applies isotropic hardening, a transactional
second-order exponential update of `Fp`, and the directional derivative of
the active algorithmic stress branch. Drucker-Prager additionally compiles
friction angle and cohesion into its pressure-sensitive corrector.

Topology capacity is immutable during a borrowed submission. Existing
cohesive insertion, erosion, and crack/channel exposure execute in stable
command order and rebuild incidence on device. Exhaustion reports
`NM_STATUS_TOPOLOGY_GROWTH_REQUIRED`; after completion the runtime publishes a
geometric `TopologyGrowthRequest`. A larger initialized runtime imports the
accepted state through `encodeTopologyGrowth` on a borrowed command buffer.
The logical mesh can therefore grow across submissions until 32-bit indices or
the device working set is exhausted, without allocation inside a transaction.

The MPM block evaluates a backward-Euler residual at the current grid
candidate. Sparse active blocks deterministically gather candidate velocity
gradients into particles, evaluate the same implicit material projection and
consistent tangent used by FEM, and gather particle force directions back to
grid rows without floating-point scatter atomics. APIC particle state is
published only after the enclosing Newton candidate succeeds. Analytic rigid
barriers retain auxiliary force rows so the
same matrix-free action includes free-body and articulated sparse Delassus
response without placing articulated coordinates in Matter's primal vector.
FEM/MPM barrier pairs are not yet generated. The contact Hessian is not the full
mollified vertex-triangle/edge-edge IPC Hessian. Those are active solver
boundaries, not completed claims.

The implementation uses FP32 barrier arithmetic and conservative CCD. It does
not claim exact-real arithmetic or a mathematical proof of non-intersection;
those remain distinct from executable nonpenetration evidence.

## Current accuracy boundary

The implemented probes establish internal analytical cases, CPU/Metal
component parity, finite behavior, selected conservation cases,
deterministic replay where stated, native G1 standing, and batched contact
execution. They do not establish universal agreement with an external
simulator, complete unilateral joint-limit behavior, high-speed impact
accuracy, real-hardware fidelity, safety, or sim-to-real transfer. Those
claims require their own pinned comparisons and physical evidence.
