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
- Contact normals point from body A to body B. Geometric witnesses retain
  separate points on A and B and signed separation.

## Precision boundary

Metal physics is FP32 because Metal shaders do not provide native `double`.
The canonical ABI and compiled robot inertial records are FP32. CPU reference
paths promote those records and evaluate dynamics, collision, and quality
solver intermediates in FP64; promotion does not recover precision absent
from the compiled record.

Every CPU reference path validates dimensions and finite inputs before
publication. Capacity, factorization, convergence, and unsupported-feature
failures have explicit status codes. The composed CPU collision/world paths
are transactional: state and persistent caches remain unchanged when a step
fails. The bounded Metal contact kernel backs up the dynamic velocities and
contact records it can mutate, then restores them after an arithmetic failure.
A fully composed production Metal world still requires versioned
input/output buffers at the dispatch orchestration layer.

## Articulated dynamics

The generalized CPU reference supports fixed or floating trees containing
revolute, continuous, and fixed joints. It:

- forms a dense FP64 mass matrix with a world-coordinate
  composite-rigid-body recursion;
- verifies positive definiteness and solves forward dynamics with Cholesky;
- computes velocity, gyroscopic, gravity, damping, and external-wrench terms
  through recursive Newton-Euler kinematics and analytic generalized-force
  projection;
- integrates with symplectic Euler or a converged implicit-midpoint solve;
- composes floating orientation with the SO(3) exponential of world angular
  velocity.

Optional coordinate and body-speed limits are validation boundaries. They
reject an invalid state transactionally; they are not yet unilateral
joint-limit constraints that generate impulses. The actual G1 topology passes
the internal FP64 analytical and forward/inverse probes, but this reference is
not coupled to contact and has no generalized Metal implementation.

The original Franka runtime is separate. Its Metal path runs a linear-time
FP32 articulated-body algorithm, and its older FP64 reduced-dynamics oracle
provides a one-control-step convention check. Agreement there is not a
bitwise-equivalence or external-accuracy promise.

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
correctness baseline for sphere/sphere and sphere/plane. It is not a parallel
broadphase and does not perform GPU persistent-manifold refresh.

No CCD algorithm is implemented. Fast bodies can therefore tunnel; substeps
are not a semantic substitute for conservative advancement, speculative CCD,
or time-of-impact island stepping.

The contact portfolio has three distinct numerical contracts:

- an independent FP64 accelerated projected-gradient exact circular-cone
  oracle;
- a globalized FP64 semismooth-Newton quality solve with KKT and cone
  diagnostics;
- a fixed-budget CPU/Metal PGS throughput block with normal, coupled
  two-tangent radial projection, torsional friction, and warm starts.

The throughput block is PGS, not temporal Gauss-Seidel: it does not advance
and relinearize contacts through internal TGS substeps. Rolling resistance is
explicitly unsupported. One throughput dispatch holds at most 128 contacts.
The composed CPU world partitions independent connected islands, so any one
connected island above 128 contacts returns capacity overflow. It does not
drop the excess contacts.

The currently composed quality world accepts one isotropic Coulomb
coefficient. A material with distinct static and dynamic coefficients,
torsional/rolling friction, or a hard impulse cap returns `MR_STEP_UNSUPPORTED`
transactionally instead of changing material semantics when solver mode
changes. A shared convex stiction model remains open.

Metal contact dispatch validates inputs and capacities before its in-place
solve and rolls back touched dynamic velocities, impulses, and warm-start
flags after an arithmetic failure. Static/kinematic endpoints are never
written during solve or rollback, allowing independent islands to share static
geometry safely.

## Current accuracy boundary

The implemented probes establish internal analytical cases, CPU/Metal
component parity, finite behavior, conservation on narrow unforced scenes,
and deterministic replay where stated. They do not yet establish:

- trajectory/contact agreement with a pinned external simulator;
- articulated contact or joint-limit impulse accuracy;
- convex, mesh, heightfield, or deformable collision accuracy;
- CCD or high-speed impact accuracy;
- long-horizon G1 locomotion stability;
- production generic Metal throughput.

The dated acceptance thresholds for making broader claims are defined in
[ENGINE_TARGET](ENGINE_TARGET.md).
