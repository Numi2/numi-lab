# MetalRobo v0.4 transactional generalized architecture

Research and implementation snapshot: **2026-07-28**.

Version 0.4 turns the operator-first design into one executable generalized
CPU timestep and one correctness-first generic Metal articulation operator.
The central rule is simple: a step is accepted only when dynamics, collision,
constraint semantics, the contact solve, independent impulse application, the
common residual, and integration all agree. No partial state or cache update
is observable on failure.

This is an architecture milestone, not a claim that MetalRobo is already a
complete simulator or a throughput winner.

## One transaction, not a chain of loosely related solvers

For one executable articulation, the CPU path performs:

```text
validate immutable model + state
  -> compute FP64 free dynamics with the authoritative mass operator
  -> project articulated body poses/twists into collision state
  -> deterministic collision + persistent manifold update in a private cache
  -> compile canonical ConstraintIR
  -> evaluate material, stabilization, restitution, regularization, and warm
     start semantics exactly once
  -> adapt the fingerprinted evaluated rows to articulated contacts
  -> build J, a checked mass Cholesky factor, and W = J M^-1 J'
  -> solve the exact circular Coulomb problem
  -> apply M^-1 J' lambda through the retained factor
  -> evaluate the solver-independent ConstraintIR residual
  -> integrate q with the accepted post-contact velocity
  -> atomically publish q, v, manifolds, impulses, and step number
```

Every stage has an explicit failure code. Pair/contact capacity overflow,
unsupported semantics, invalid evaluated fingerprints, failed
factorizations, failed convergence, residual failure, and late integration
failure all leave the caller's state and caches unchanged.

The current composed implementation intentionally accepts exactly one
articulation plus static or kinematic environment bodies. Multi-articulation
islands, dynamic free objects, self-collision filtering, and a fully composed
Metal timestep remain subsequent work.

## Evaluated ConstraintIR is authoritative

Raw contacts are not allowed to choose solver behavior independently. The
ConstraintIR evaluator owns:

- stabilization and restitution targets;
- static-versus-dynamic friction selection;
- compliance, dissipation, and discrete regularization;
- feasible warm-start projection;
- cone and row activation;
- a semantic fingerprint over the evaluated program.

The articulated adapter consumes the evaluated targets, regularization,
friction, and warm impulses directly. It rejects semantics that the current
three-row exact-cone solver cannot represent, including rolling, torsion,
adhesion, anisotropy, caps, and zero-regularization blocks. Re-deriving those
values from the old contact record would be a semantic fork and is forbidden.

After the solve, the same evaluated program computes a natural residual from
the actual post-impulse world-point velocities. A solver's own convergence
flag is necessary but not sufficient for acceptance.

## No dense generalized inverse in the production quality path

The articulated contact operator retains a Cholesky factor of the physical
mass matrix and exposes:

```text
J v
J' lambda
M^-1 J' lambda
J M^-1 J' lambda
```

The quality solver consumes the precomputed physical Delassus matrix and
free contact velocity. It never constructs `M^-1`. Its returned
post-impulse contact velocity is compared with `J` applied to the velocity
produced by the independent factor solve.

The FP64 projected-gradient oracle still has an explicit dense compatibility
adapter. That adapter is built only when the oracle is selected, and its
`M M^-1 - I` residual is checked. Keeping the independent implementation is
valuable for differential validation; allowing it into the production path
is not.

The quality solve still materializes the small-island contact Hessian. This is
appropriate for the current direct semismooth Newton path, but it is not the
final large-island representation. Large quality islands need a matrix-free
Newton-PCG path over `applyDelassus`.

## Per-DoF truth and armature

Engine ABI v2 adds exactly one aligned, pointer-free
`MRDofPropertiesGPU` record per generalized velocity coordinate. A record
owns:

- articulation, joint, global `q`/`v`, and joint-local DoF indices;
- root, actuation, drive, and limit flags;
- position, velocity, and effort limits;
- stiffness, damping, armature inertia, and dry-friction loss.

Model validation proves that this stream is complete, globally ordered, and
consistent with every articulation and joint. Floating-root records are
strictly passive.

Armature is physical generalized inertia, not controller metadata. CPU CRBA
adds it to the mass diagonal, RNEA adds `armature * qdd`, invariant evaluation
adds its kinetic energy, and every forward/contact/impulse factorization
therefore observes the same operator. The generic Metal operator consumes the
same stream and applies the same diagonal term.

G1 compiles factual URDF position, velocity, and effort limits together with a
named Unitree RL Lab Kp/Kd/armature preset. These sources remain distinct in
the specification and provenance.

`evaluateArticulatedActuation` now executes disabled, model-PD, custom-PD, and
direct-effort modes transactionally. It forbids floating-root actuation,
evaluates feed-forward plus PD exactly, wraps continuous-joint position error
onto the shortest signed angle, clamps motor effort before passive loss, and
reports saturation and dissipation. Moving Coulomb loss strictly opposes
velocity. Its near-zero cancellation is intentionally controller-local:
gravity, bias, external, and contact loads require coupled set-valued stiction
inside the dynamics/constraint solve. The actuation evaluator can feed the
world's generalized-force input, but it is not yet embedded as an implicit
drive block in the composed step.

## Generic Metal articulation operator

The new Metal kernel accepts immutable canonical articulation, joint, body,
and DoF records plus environment-major `q` and point-impulse queries. For
fixed or floating revolute trees within its declared capacity it computes:

- body COM poses;
- queried world points and analytic point Jacobians;
- diagnostic dense `M`;
- `J' p`;
- `M^-1 J' p` through a checked Cholesky solve.

The kernel validates ownership, offsets, root conventions, finite data,
armature, and capacities before publication. It uses a scale-aware pivot
threshold and returns an explicit failure instead of adding hidden
regularization. G1 exercises 30 bodies, 36 configuration coordinates, and 35
velocity coordinates, exceeding the old Franka-era 32-DoF limit.

`runMetalArticulatedOperator` is the checked public boundary around that raw
kernel. It owns all 15 compact buffers, derives every stride and dispatch
dimension, checks host arithmetic and the shader's 32-bit element-address
ceiling, preflights each allocation and the aggregate recommended working set,
binds typed dummies for zero logical streams, dispatches exactly one
threadgroup per environment, validates typed GPU statuses, and publishes the
whole result transactionally. Callers cannot provide aliases, offsets, or
unchecked raw buffer layouts.

This first kernel deliberately executes the dense correctness path in lane
zero. It proves the shared ABI and equations on Apple GPU hardware; its timing
is not a throughput result. The checked wrapper is synchronous and recreates
its device objects, pipeline, and buffers on each call. The next
implementation must use parallel tree traversals, batched right-hand sides, a
persistent cached context, reusable/ping-pong buffers, and a composed
asynchronous device-resident command stream without changing these semantics.

The first part of that successor is now executable in
[METAL_WORLD](METAL_WORLD.md): pipelines/model buffers persist, q/v use
transactional ping-pong plus a control-step checkpoint, and a complete
free-motion horizon is one asynchronous command buffer. Parallel tree
traversal, right-hand-side fusion, and contact composition remain open.

## Cylinder geometry without a fake generic-convex claim

CPU FP64 and Metal FP32 now implement oriented cylinder AABBs and
cylinder/plane witnesses. Feature selection is deterministic:

- a four-point cap ring when the supporting cap is parallel to the plane;
- two ordered side-rim endpoints for a parallel cylinder axis;
- one extremal rim point in the general tilted case.

Collider-order reversal, stable feature IDs, tight outward AABBs, exact
capacity behavior, malformed-input rejection, replay determinism, and
CPU/Metal witness parity are executable gates.

This is not general cylinder collision. Cylinder/sphere, cylinder/capsule,
cylinder/box, cylinder/cylinder, convex, mesh, and heightfield pairs remain
unsupported. G1 shoulder cylinders therefore stay disabled until enabling
them cannot introduce unsupported self-collision or pair semantics.

## What v0.4 proves

- Actual G1 CPU free dynamics, collision-generated foot contact, evaluated
  exact-cone solving, residual acceptance, and configuration integration are
  composed transactionally.
- The production quality contact path uses factor/Delassus actions and does
  not materialize a dense generalized inverse.
- CPU and Metal use one authoritative per-DoF armature stream.
- A generic Metal operator executes actual floating G1 kinematics, mass,
  Jacobian, transpose, and impulse response with deterministic replay.
- The public host encoder closes raw-buffer length, address-wrap,
  zero-binding, status-publication, and transactionality boundaries.
- Oriented cylinder/plane geometry has CPU/Metal parity and adversarial
  capacity/input gates.

It does **not** yet prove:

- a contact-composed, level-parallel, MLX-buffer-native Metal world step; the
  successor free-motion graph is batched and device-resident within a
  submitted horizon;
- long-horizon G1 locomotion stability or RL throughput;
- coupled implicit drives, set-valued joint stiction, joint-limit impulses,
  loop constraints, and self collision;
- large-scene broadphase, GPU manifold persistence, convex/mesh/terrain
  collision, or CCD;
- benchmark superiority over MuJoCo, Genesis Metal, or NVIDIA systems;
- qualified differentiability across contact events.

## Next heavy lifts

1. Turn the correctness Metal operator into parallel ABA/CRBA actions with
   multiple right-hand sides and no dense diagnostic matrix in ordinary use.
2. Compose the landed actuation evaluator into batched worlds, add implicit
   drive/stiction blocks, and compile joint limits into ConstraintIR.
3. Extend the landed free-motion work tickets to
   multi-articulation/free-object islands.
4. Add deterministic GPU manifold refresh/reduction, segmented LBVH, convex
   support mapping, mesh/heightfield collision, and certified CCD intervals.
5. Implement exact-block temporal substeps for throughput and matrix-free
   Newton-PCG for large quality islands, both ending in the common residual.
6. Only then compose Franka manipulation and G1 locomotion benchmarks against
   pinned competitors on the same Apple hardware.

## Research basis

- [MuJoCo computation and constraint islands](https://mujoco.readthedocs.io/en/latest/computation/)
- [MuJoCo 3.10 changes](https://mujoco.readthedocs.io/en/stable/changelog.html#version-3-10-0-june-22-2026)
- [Newton solver matrix](https://newton-physics.github.io/newton/stable/solvers/index.html)
- [Newton articulation model](https://newton-physics.github.io/newton/latest/concepts/articulations.html)
- [Drake SAP paper](https://arxiv.org/abs/2110.10107)
- [Drake compliant contact](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html)
- [Apple silicon Metal porting guidance](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)
- [Apple Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Karras parallel LBVH construction](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees)
- [Tight Inclusion CCD](https://continuous-collision-detection.github.io/tight_inclusion/)
