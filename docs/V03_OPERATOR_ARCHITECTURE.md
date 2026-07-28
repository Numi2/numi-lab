# MetalRobo v0.3 operator-first architecture

Research and implementation snapshot: **2026-07-28**.

MetalRobo will not become state of the art by accumulating unrelated
collision kernels and solver names. Its defining architecture is that
collision, high-accuracy simulation, RL throughput, diagnostics, and
differentiation consume one stable constraint program and one generalized
free-motion operator.

This document is both a design decision and a claim boundary. A component
listed as a target is not implemented merely because its record layout is
defined.

## The mistakes we are designing out

1. **One monolithic dense solve.** Dense per-world inverse mass and constraint
   Jacobian storage scales badly once scenes grow beyond small robots. MuJoCo
   Warp currently documents dense constraint Jacobians and performance
   degradation beyond roughly 60 DoF, while native MuJoCo has moved further
   toward sparse mass storage, exact current-configuration Delassus
   information, and parallel island solves.
2. **One solver for an entire scene.** Independent islands have different
   sizes, conditioning, contact churn, and accuracy needs. The hardest island
   must not determine the work or numerical method for every other island.
3. **Solver-dependent physics semantics.** Materials, restitution, drive
   targets, friction cones, activation, and warm-start validity are physical
   program inputs. A throughput solver may approximate the shared problem on
   a fixed budget; it may not silently solve a different problem.
4. **Atomic append collision streams.** Global append counters make ordering
   depend on GPU scheduling and make exact overflow handling awkward.
   MetalRobo collision stages use count, exclusive scan, and deterministic
   scatter into canonical key order.
5. **Post-hoc anti-slip corrections.** A cascade that modifies friction after
   the main solve no longer optimizes one declared objective. Static friction
   belongs inside the common cone model and residual.
6. **Unqualified “differentiable physics.”** Gradients through a converged
   smooth region do not remain valid across impact ordering, contact creation,
   feature switches, stick/slip changes, grazing CCD roots, or sleep events.
   Those boundaries must be machine-readable results, not documentation
   footnotes.

## One semantic constraint program

Every contact, joint limit, loop closure, weld, distance row, tendon, gear,
drive, and dry-friction element compiles into the same structure-of-arrays
program:

- a stable key and typed block header;
- one or more articulated or free-body endpoints;
- body-local anchors and axes from which Jacobian actions are evaluated;
- continuous position error, target velocity, compliance, dissipation, time
  constant, and damping ratio;
- impulse bounds or a circular/elliptic cone;
- static/dynamic, torsional, and rolling friction;
- restitution, impact threshold, adhesion, and maximum impulse;
- warm-start cache and event-ledger slots.

A shared evaluator derives timestep-dependent bias and regularization from
those continuous semantics. Quality Newton, temporal Gauss-Seidel, and the
FP64 oracle all call the same activation, material mixing, restitution, cone
projection, target/bias, warm-start validation, and residual routines.

The current `MRContactConstraintGPU` remains an ABI-v1 compatibility record.
It is an adapter target, not the long-term source of truth.

The first executable CPU slice now defines the fixed-layout ABI-v2 records,
strict canonical validator, transactional v1 contact adapter, shared
timestep/material evaluator, semantic fingerprint, and common residual. The
residual handles scalar bounds plus capped elliptic normal/tangent/torsion as
one coupled projection. Quality and throughput views alias the same evaluated
buffers. Contacts are the only compiled v1 source today; limits, loops,
drives, tendons, rolling, and adhesion still require executable adapters and
Metal records.

## The free-motion operator

The central dynamics object is the action of

\[
A^{-1},\qquad
A=M(q)+hD+h^2K ,
\]

where the damping and stiffness terms are present only for the selected
implicit free-motion model. Required operations are:

```text
applyA(x)                  -> A x
solveA(rhs)                -> A^-1 rhs
applyJ(v)                  -> block-space relative velocities
applyJT(impulses)          -> generalized impulses
applyDelassus(impulses)    -> J A^-1 J' impulses + R impulses
buildBlockDiagonal()       -> exact 1x1, 2x2, or 3x3 effective masses
```

Backends share this interface:

- independent free bodies use diagonal spatial-inertia blocks;
- tree articulations use an articulated-body impulse response;
- implicit articulations use a factored free-motion Hessian;
- the CPU oracle uses FP64 CRBA and a checked factorization.

The ordinary Metal step must not materialize a dense inverse. Batched
normal/tangent basis impulses should share one tree traversal so a contact's
full 3x3 Delassus block is obtained together. Dense FP64 matrices are allowed
as small-system solver adapters and audit artifacts, with that role stated
explicitly.

## Collision is a deterministic stream compiler

The production stream is:

```text
poses and swept AABBs
  -> candidate count
  -> exclusive scan
  -> canonical pair scatter
  -> pair-class count and scan
  -> narrowphase count and scan
  -> raw contact sort
  -> sorted old/new manifold merge
  -> deterministic manifold reduction
  -> constraint-program compilation
  -> island construction and profiling
```

The v0.3 micro broadphase already implements parallel upper-triangular pair
classification followed by a two-level scan and scatter. It does not use a
global append atomic. It has an explicit 65,536-logical-pair scan limit;
larger scenes require segmented/recursive scan or the LBVH path rather than
silent truncation. Its executable probe covers five scan blocks, CPU/Metal
candidate parity, bit-identical replay, exact-capacity success, and untouched
output on one-short or non-finite failure.

This micro path is appropriate for fixed-topology Franka and G1 scenes. The
large-scene path will use segmented Morton ordering and a binary radix LBVH.
Tree refit is levelized deepest-to-root so correctness does not depend on
parent-arrival atomics. Counts and scans own output ranges at every
variable-length stage.

The current Metal narrowphase covers sphere/sphere, sphere/plane,
capsule/plane, and box/plane. Box/plane emits the full geometric witness set
before deterministic four-point CPU reduction. Convex clipping, GJK/EPA/MPR,
mesh/heightfield, persistent Metal manifolds, and CCD remain open.

## Per-island numerical dispatch

Every island produces a deterministic profile:

- generalized velocity count and scalar constraint dimension;
- block count and graph-color count;
- new impacts, CCD events, and contact churn;
- diagonal/effective-mass spread;
- warm-start residual, minimum time-constant ratio, and penetration;
- backend capability bits.

The profile selects a solve ticket:

- no constraints: free motion only;
- small quality island: tiled direct semismooth Newton;
- larger quality or differentiable island: matrix-free Newton-PCG;
- throughput request: colored exact-block TGS;
- reliability request: replay a failed-residual throughput island through
  quality mode transactionally.

Size thresholds are versioned device-profile data and are calibrated over an
overlap interval. They never change material or constraint meaning. Every
path ends in the same natural-residual, cone-violation, complementarity, and
energy report.

The current quality solver and fixed-budget PGS are not yet this dispatch
system. PGS remains an explicitly named compatibility baseline until temporal
microstep relinearization and the common residual are implemented.

## Differentiability is an event contract

Every accepted step will emit a sorted event ledger containing:

- stable constraint/pair key;
- event type and projection region before/after;
- a time bracket for impacts and CCD;
- event margin and transversality;
- a reason mask when the local derivative contract is invalid.

Supported gradient meanings are distinct:

- `piecewise_exact`: implicit VJP of a converged quality solve while keys,
  active regions, and event ordering remain unchanged;
- `smooth_surrogate`: a separately tagged smooth forward model without hard
  restitution, sleep, or event CCD;
- `algorithmic_tgs`: derivative of a fixed iteration map, never labeled as
  the converged physical derivative.

CCD returns a certified safe interval, not only a Boolean hit. An isolated
transversal root with positive event-order margin can be differentiated.
Grazing, simultaneous impact, feature changes, contact creation/removal,
stick/slip transitions, sleep/wake, overflow, and fallback explicitly
invalidate the piecewise derivative.

## Implementation sequence

1. **Landed on CPU:** compile ABI-v1 contacts into the semantic program and
   prove one shared evaluator/residual path.
2. Make CPU `solveA`, `applyJ`, `applyJT`, and `applyDelassus` executable;
   retain dense matrices only as FP64 compatibility adapters.
3. **Landed on CPU for G1 feet:** connect collision-generated manifolds to
   reduced-coordinate impulses. Next, compose the same boundary with Franka
   free-object manipulation and the full articulated timestep.
4. Port floating ABA/impulse response and the common block diagonal to Metal.
5. Complete deterministic Metal manifold streams, then add segmented LBVH.
6. Add island profiles, direct/PCG quality queues, colored TGS, and residual
   replay.
7. Add certified primitive/convex CCD and its event ledger.
8. Add implicit quality VJP only after forward semantics and event reporting
   are stable.

## Research basis

- [MuJoCo computation and constraint islands](https://mujoco.readthedocs.io/en/latest/computation/index.html)
- [MuJoCo 3.10 thread-pool/island changes](https://mujoco.readthedocs.io/en/stable/changelog.html#version-3-10-0-june-22-2026)
- [MuJoCo Warp documented scaling, density, determinism, and autodiff limits](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html)
- [Drake SAP compliant-contact formulation](https://arxiv.org/abs/2110.10107)
- [Newton solver capability matrix](https://newton-physics.github.io/newton/stable/solvers/index.html)
- [PhysX GPU rigid-body limitations](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/GPURigidBodies.html)
- [Karras parallel binary radix tree](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees)
- [Tight Inclusion CCD benchmark](https://continuous-collision-detection.github.io/tight_inclusion/)
- [Dojo implicit differentiable contact](https://arxiv.org/abs/2203.00806)
