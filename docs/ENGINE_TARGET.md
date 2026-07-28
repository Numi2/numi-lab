# MetalRobo rigid-body engine target

Research snapshot: **2026-07-28**. This is the engineering contract for the
engine MetalRobo is intended to become. It is not a description of the current
implementation.

MetalRobo v0.2 is an executable engine spine: fixed-base batched Franka ABA,
a generic floating-root `nq != nv` model, CPU FP64 CRBA/RNEA forward and
inverse articulated dynamics (including the pinned G1), CPU/Metal free-body
integration, deterministic primitive collision and persistent manifolds,
exact-cone reference/quality contact solvers, a CPU/Metal fixed-budget PGS
block, and a transactional CPU maximal-coordinate world. It still lacks
floating articulated contact, production parallel Metal collision, convex and
mesh narrowphase, CCD, loop/joint/limit constraints in the common solver,
importers, and physics derivatives. It must not currently be called a complete
or state-of-the-art physics engine.

The target in this document is narrower and more testable than “everything in
Isaac Sim”: a **state-of-the-art robotics rigid-body engine on Apple GPUs**.
Rendering, deformables, fluids, synthetic-data generation, and an editor are
separate products. They cannot substitute for rigid-contact correctness.

## What “state of the art” means here

The phrase is earned only when all of the following are true:

1. The engine covers the mechanics required by modern manipulation and
   locomotion: reduced-coordinate fixed and floating articulations, independent
   free bodies, collision manifolds, loop/equality constraints, unilateral
   contact and limits, Coulomb friction, robust impact handling, and CCD.
2. The default quality solver exposes a precise mathematical objective,
   convergence diagnostics, and physical compliance parameters. A user can
   tell whether a step converged; capacity overflow or fallback is never silent.
3. A deterministic mode replays bit-for-bit on the same Apple device and build,
   while a double-precision CPU implementation acts as the numerical oracle.
4. A high-throughput solver and a high-accuracy solver share model, collision,
   constraint, material, and diagnostic semantics. “Fast” is not a separate,
   incompatible physics engine.
5. Differentiability is described honestly. Smooth, fixed-topology regions have
   verified JVP/VJP support; collision-set changes, stick/slip transitions,
   impact events, CCD time-of-impact changes, and sleep/wake decisions are
   explicitly treated as nonsmooth event boundaries.
6. MetalRobo is on the measured accuracy/throughput Pareto frontier against
   pinned competitors on published scenes. A large empty-scene FPS number is
   not evidence.

This definition is intentionally stronger than feature presence. A contact
solver that returns a plausible animation but cannot report residuals, drops
contacts when a buffer fills, or requires scene-specific hidden tuning does not
meet the bar.

## Primary-source baseline

No current engine wins every axis. The target should take the best ideas while
retaining one coherent set of semantics.

| System | What the current primary source establishes | What MetalRobo should take, and not copy blindly |
| --- | --- | --- |
| **MuJoCo 3.10** | MuJoCo defines soft equality, limit, dry-friction, and contact forces through a convex optimization problem. Newton and nonlinear CG solve the reduced primal; PGS solves the dual. Newton uses analytical second derivatives, Cholesky and exact line search; both pyramidal and elliptic cones are supported. It warm-starts constraint forces and independently solves constraint islands. Version 3.10 improved FP32 primal convergence and changed CG to Hager-Zhang; 3.9 added an exact Delassus diagonal option. ([constraint computation](https://mujoco.readthedocs.io/en/stable/computation/index.html#constraint-solver), [3.10 changelog](https://mujoco.readthedocs.io/en/stable/changelog.html#version-3-10-0-june-22-2026)) | Adopt a documented convex model, exact-cone quality mode, cost-aware warm starting, residual certificates, islanding, and separate direct/iterative linear algebra. Do not assume MuJoCo’s continuous-time acceleration formulation or optional NoSlip correction is uniquely correct; its own documentation calls NoSlip an ad-hoc cascade that can destabilize complex multi-contact scenes. |
| **NVIDIA PhysX 5.9** | The current SDK release is 5.9. PhysX has reduced-coordinate articulations and free rigid bodies. Its PGS and TGS solvers run on CPU and CUDA GPU; TGS advances and relinearizes through internal substeps, improving mass-ratio, drive, and high-frequency behavior. It uses persistent contact manifolds and friction patches. PhysX supplies sweep-based conservative-advancement CCD plus speculative CCD, but GPU mode has fixed capacities and some operations still take CPU paths; D6 is the only joint with a fully GPU-resident preparation path. ([5.9 release](https://github.com/NVIDIA-Omniverse/PhysX/releases/tag/110.1-omni-and-physx-5.9.0), [5.8 simulation and TGS reference](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html#constraint-solver), [articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Articulations.html), [GPU simulation](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/GPURigidBodies.html), [CCD and PCM](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/AdvancedCollisionDetection.html)) | Adopt reduced joints, persistent manifolds, a TGS throughput mode, hybrid sweep/speculative CCD, explicit capacity planning, and per-island work. Do not accept CPU fallbacks, dropped GPU contacts, per-axis friction asymmetry, or solver-order artifacts as MetalRobo’s final semantics. |
| **Drake SAP** | Drake treats compliance as part of the physical model rather than a hidden numerical patch. SAP eliminates the contact constraints analytically to obtain a strongly convex, unconstrained velocity problem, uses exact friction-cone projections and a semi-analytic Newton method, has global convergence guarantees, warm-starts effectively, and supports a midpoint scheme that is second-order accurate in the authors’ tests. Drake also supports point and hydroelastic contact. ([contact-model rationale](https://drake.mit.edu/doxygen_cxx/group__drake__contacts.html), [SAP paper](https://arxiv.org/abs/2110.10107), [current contact defaults](https://drake.mit.edu/doxygen_cxx/group__contact__defaults.html)) | Make a SAP-like convex velocity formulation the quality reference, use real stiffness/dissipation parameters, retain the exact circular cone, and use implicit differentiation of a converged solve. Do not copy Drake’s CPU-only implementation constraints or claim its current `AutoDiffXd` restrictions are inherent to SAP; Drake’s own API says SAP contact currently throws under `AutoDiffXd`. |
| **Genesis World** | Genesis uses reduced-coordinate dynamics and follows MuJoCo’s soft-constraint model. It offers Newton-Cholesky and preconditioned CG, warm starts across steps, uses a dense tiled GPU factorization, and represents each contact with a four-edge friction pyramid. Its collision pipeline includes Sweep-and-Prune, analytical pairs, GJK/MPR/SDF paths, and differentiable GJK selection. Metal is a documented FP32 backend. ([rigid constraint model](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_collision/rigid_constraint_model.html), [collision pipeline](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_collision/collision_contacts_forces.html), [backend and precision](https://genesis-world.readthedocs.io/en/latest/user_guide/configuration/initialization.html)) | Treat Genesis Metal as the immediate same-hardware throughput competitor. Reuse the successful pattern of compiled, batched scene fields and direct-vs-iterative solvers, but make exact isotropic friction and solver/capacity diagnostics first-class rather than leaving a four-sided cone as the only path. |
| **Newton Physics / MuJoCo Warp** | Newton exposes a solver portfolio rather than pretending that XPBD, maximal-coordinate semi-implicit dynamics, reduced Featherstone dynamics, and MuJoCo-style constraints are interchangeable. Its feature tables disclose which joint and material properties each solver actually supports. MuJoCo Warp is its primary large-scale rigid backend, while the official MJX documentation explicitly says the Warp implementation does not support autodiff. ([Newton solver matrix](https://newton-physics.github.io/newton/stable/solvers/index.html), [MJX implementations](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-implementations)) | Adopt an explicit capability matrix and one common state/model ABI. Do not market “differentiable” at the framework level when the chosen production rigid solver or collision path is not differentiable. |

The architectural conclusion is that MetalRobo needs **one constraint
semantics layer and multiple numerical realizations**, selected per island and
quality mode. Merely adding PGS to the present penalty forces would not reach
the target.

## Canonical model and state

### Coordinates

A compiled world is a collection of kinematic trees plus static geometry:

- A fixed-base articulation has only its joint coordinates.
- A floating articulation has a root position and unit quaternion in `q`, a
  six-dimensional root twist in `v`, and its reduced joint coordinates.
- A free rigid body is a one-link floating articulation and goes through the
  same dynamics and constraint interfaces.
- Revolute, continuous, prismatic, spherical, planar, fixed, and free joints
  are required. A generic D6 joint is represented by the corresponding subset
  of those tangent-space degrees of freedom.
- Closed chains are represented as reduced-coordinate trees plus bilateral
  loop rows. They are not converted wholesale to drift-prone maximal
  coordinates.

`nq` and `nv` are deliberately different. Quaternions are never treated as
four independent velocity coordinates. Rotations integrate on SO(3) with an
exponential map and are normalized defensively after each accepted step.

Every dynamic state has versioned buffers for `q`, `v`, actuation state,
external wrench, sleep state, and previous-step constraint cache. Derived link
poses, twists, accelerations, Jacobians, contacts, impulses, and residuals are
separate buffers. A reset restores all of them, not just `q` and `v`.

### Mass and actuation

The shared spatial-algebra layer implements:

- RNEA for bias and inverse dynamics;
- CRBA for an explicit joint-space mass matrix when factorization or reporting
  needs it;
- ABA for unconstrained forward dynamics and matrix-free inverse-mass actions;
- geometric, body, and point Jacobians;
- rotor/armature inertia, viscous and dry joint friction;
- force, position, velocity, and physically parameterized implicit PD drives;
- effort, velocity, and position limits.

Fixed and floating trees use the same algorithms. The CPU oracle instantiates
them in FP64. Metal kernels use FP32, constraint scaling, compensated or
pairwise reductions where useful, and condition estimates. There is no claim
of FP64 Metal physics because Apple GPU execution does not provide it.

## One constraint language

Collision, limits, loops, and user joints compile into typed blocks. A block
contains stable body/feature IDs, dimension, Jacobian action, configuration and
velocity error, target velocity, impulse bounds or cone, compliance,
dissipation, restitution state, and its warm-start value.

The mandatory block types are:

- bilateral connect, weld, distance, gear/mimic, tendon, and loop closure;
- unilateral lower/upper joint and tendon limits;
- contact normal plus two coupled tangential dimensions;
- optional torsional and rolling friction dimensions;
- bounded dry-friction and actuator-force rows.

For a contact impulse
\(\gamma=(\gamma_n,\boldsymbol{\gamma}_t)\), the quality model uses the
elliptic Coulomb cone

\[
\gamma_n \ge 0,\qquad
\left\|\operatorname{diag}(1/\mu_1,1/\mu_2)
\,\boldsymbol{\gamma}_t\right\|_2 \le \gamma_n .
\]

The isotropic case has \(\mu_1=\mu_2\). A configurable 4- or 8-edge pyramid is
allowed only as a named throughput approximation. The selected cone is stored
in traces and benchmark output. Static and dynamic friction, optional rolling
and torsional friction, and maximum-dissipation direction are material
semantics, not solver-specific switches.

Restitution is applied only to a newly impacting contact whose incoming normal
speed exceeds a material velocity threshold. Persistent resting contact gets a
zero normal-velocity target. This prevents restitution from continually
injecting energy into a resting manifold.

### Quality objective

Let \(v^*\) be free-motion velocity and let \(A\succ0\) be the free-motion
Hessian. For semi-implicit motion \(A=M(q)\); implicit drives and passive
stiffness add their positive contributions. The default quality solve is the
strongly convex velocity problem

\[
v_{n+1}=\arg\min_v
\frac{1}{2}\|v-v^*\|_A^2+
\sum_i \ell_i(J_i v+b_i;R_i).
\]

Each \(\ell_i\) is the analytically reduced convex loss for a bilateral row,
half-line limit, bounded dry-friction row, or friction cone. Its impulse is an
analytic projection onto the corresponding set. \(R_i\) is physically derived
from compliance and dissipation and also keeps the problem well conditioned.
Constraint time constant, damping ratio, material stiffness, Hunt-Crossley
dissipation, and stiction regularization are public SI-valued inputs. Hidden
“magic stiffness” is forbidden.

For a scalar holonomic error \(g(q)\) with relative speed \(w=Jv\), the
default stabilization target is the critically or user-damped second-order
law

\[
a_{\mathrm{ref}}=-\frac{2\zeta}{\tau}w-\frac{1}{\tau^2}g ,
\]

discretized consistently by the selected integrator, with a lower bound on
\(\tau/h\) and an explicit depenetration-speed cap. The default bound is
\(\tau\ge2h\); quality mode can admit a smaller physical time constant only
when its condition check passes, and records that override. Materials may
instead supply stiffness and dissipation directly, in which case the compiler
derives the equivalent discrete regularizer and reports it.

This is a compliant time-stepping approximation to real contact. The exact
elliptic cone means that the chosen convex model does not add polygonal tangent
bias; it does **not** mean that nonsmooth, perfectly rigid Coulomb contact has a
unique physical solution. The friction regularization and effective compliance
are therefore part of every trace and comparison.

Reduced-coordinate tree joints are exact by construction. Loop constraints can
be either:

- **compliant**, using the same convex objective; or
- **hard bilateral**, eliminated in a mass-weighted null space or solved by a
  rank-revealing KKT/Schur complement in CPU-reference mode.

The Metal quality path uses a documented minimum compliance when exact
elimination would be ill-conditioned. It reports the effective value. “Hard”
must never secretly mean “an arbitrary large spring.”

## Solver portfolio

All solvers consume the same blocks and produce the same impulse, residual,
status, and sensor records.

### `quality_newton`

This is the default for validation, manipulation, and release claims.

1. Solve free motion with ABA or the factored implicit free-motion operator.
2. Warm-start `v` and block impulses from the persistent cache.
3. Evaluate exact cone projections, the objective, gradient, and generalized
   Hessian.
4. Take semi-smooth Newton steps with a monotone line search.
5. Stop on scaled optimality and step-improvement tolerances, not an arbitrary
   fixed visual iteration count.

Small and medium islands use a tiled SPD Cholesky factorization in generalized
velocity space. Large sparse islands use the same Newton outer iteration with
matrix-free Hessian-vector products and a tree/constraint-block
preconditioned-CG inner solve. Islands are bucketed by size so a large island
does not force dense storage on every batched environment.

The initial dispatch contract is direct solve for islands with at most 64
generalized velocities and 256 scalar constraint dimensions, PCG above either
limit. Those thresholds are versioned tuning data, not physics semantics, and
overlap cases around the boundary must demonstrate solver agreement before a
threshold changes.

The direct and PCG variants are two linear-algebra paths for the **same**
objective. Their solutions must agree to tolerance. If factorization,
line-search, or convergence fails, the per-environment status records the
failure and optional regularization retry. The runtime does not silently return
the last iterate as a successful physics step.

### `throughput_tgs`

This is the fixed-budget RL solver. A frame is divided into temporal
microsteps. At each microstep it refreshes poses, contact arms, errors, and
Jacobians; applies one block projected Gauss-Seidel sweep; then advances the
state. Contact tangents update as a coupled 2D cone block, avoiding per-axis
friction bias. Independent blocks are graph-colored for parallel execution,
while accumulated impulses make every sweep warm-started.

TGS exposes:

- microstep count;
- optional final unbiased velocity sweep;
- compliance, bias and maximum depenetration speed;
- cone approximation;
- achieved residual after the fixed budget.

It is not allowed to use different contact signs, material mixing, restitution,
or limit semantics from `quality_newton`. The quality solver is the reference
that quantifies TGS approximation error.

### `reference_fp64`

The CPU solver shares parsers, model compilation, constraint generation, and
contact manifolds but evaluates dynamics and `quality_newton` in double
precision. It can use pivoted sparse factorizations and very tight tolerances.
It is an oracle and debugging mode, not an excuse to send unsupported Metal
features to the CPU during training.

## Collision and persistent contact

### Broad phase

- Static geometry is cooked into a quantized BVH.
- Dynamic shapes use swept AABBs and either a deterministic radix-sorted LBVH
  or sweep-and-prune for small coherent worlds.
- Collision masks, explicit excludes, parent/adjacent-link filtering, and
  allowed self-collision pairs are compiled ahead of stepping.
- Pair generation, prefix sums, overflow checks, island construction, and
  compaction remain on the GPU.

Every buffer has a declared capacity and a high-water mark. Overflow is a
reported step failure with the required capacity; contacts and pairs are never
discarded silently.

### Narrow phase

The required discrete shape matrix is:

- analytical sphere, capsule, plane, box, cylinder, and heightfield pairs;
- SAT/clipping for box and polyhedral face manifolds;
- GJK distance plus EPA penetration for general convex pairs;
- BVH-pruned convex-to-triangle-mesh contact;
- static/kinematic triangle meshes and heightfields;
- compound convex decomposition for dynamic concave assets.

SDF collision is a later quality/performance option, not the only way to make a
bad mesh collide. Degenerate geometry, zero-area triangles, invalid hulls, and
non-positive inertias fail model compilation with a useful diagnostic.

Each shape pair owns a **persistent contact manifold**. Contacts carry stable
shape-pair and geometric feature IDs, local-space anchors on both bodies,
normal, separation, lifetime, and material. The manifold refreshes old points,
drops invalid ones, adds newly exposed features, and reduces to at most four
well-spread points without changing total patch behavior unnecessarily.
Friction anchors and cached impulses rotate into the new contact frame and are
projected back into the current cone. A stale cache competes against a zero
start; the lower-objective start wins.

For high-fidelity grasping, the final target includes an optional
area-aware/hydroelastic patch representation whose quadrature forces and
moments enter the same constraint interface. Point manifolds remain the fast
default.

## Continuous collision detection

“CCD” must mean continuous-in-time collision handling, not merely the GJK/EPA
convex collision detector that some projects call a CCD library.

MetalRobo uses a hybrid device-resident path:

1. A speed/thickness test opts bodies or pairs into CCD.
2. Swept AABBs generate candidates.
3. Analytical primitive TOI and conservative advancement using convex distance
   compute the earliest linear/angular-safe impact time.
4. The island advances to the TOI, solves an impact constraint, and advances
   the remaining time. A bounded number of impact events is allowed per frame.
5. A motion-inflated speculative contact handles angular motion and is the
   fallback when the event budget is exhausted.

The engine must report event-budget exhaustion and remaining time. It may not
silently drop the remainder, a limitation documented for PhysX’s finite-pass
conservative advancement. CCD is required for free bodies, articulation links,
kinematics, compounds, and dynamic-vs-dynamic pairs. Self-connected adjacent
links are filtered unless explicitly enabled.

## Integration and stabilization

Three named modes share force and constraint semantics:

- symplectic Euler for the cheapest RL path;
- TGS temporal microstepping for contact-heavy fixed-budget RL;
- implicit midpoint/velocity integration for quality and convergence studies.

Constraint error is stabilized through the physical time constant and damping
in the discrete objective. Bias and depenetration velocity are capped.
TGS relinearizes after each microstep. Optional post-integration projection is
mass weighted and restricted to hard bilateral loop drift; it is never used to
teleport penetrating contact bodies apart.

Required conservation hygiene includes gyroscopic torque for free bodies,
SO(3) integration, equal-and-opposite contact wrenches, velocity-consistent
position correction, and no default global damping. Sleeping operates on whole
constraint islands with energy and force thresholds and is disabled in
gradient mode.

## Determinism and failure semantics

MetalRobo has two explicit execution modes:

- `fast`: permits atomic accumulation and convergence-dependent work;
- `deterministic`: stable radix ordering by world/body/shape/feature,
  deterministic island and graph-color construction, fixed iteration budgets,
  fixed reduction trees, counter-based per-environment random numbers, and no
  unordered floating-point atomics.

On the same model hash, initial-state hash, executable, OS build, and Apple GPU,
`deterministic` must produce bit-identical state, contact, impulse, reward, and
termination hashes. Cross-device or cross-compiler bitwise identity is not
promised; PhysX likewise scopes numerical determinism to a platform/build
because compilers and floating-point hardware can differ
([PhysX determinism contract](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/API.html#determinism)).

Every world produces a compact step status:

- converged / fixed-budget complete;
- non-finite input or result;
- broadphase, contact, constraint, or CCD capacity overflow;
- factorization failure or ill-conditioning;
- Newton/PCG/TGS iteration and residual summary;
- CCD event-budget exhaustion;
- unsupported differentiability event.

A policy may choose reset, retry at a smaller step, or fail the batch. The
physics kernel does not silently reset a world or hide a bad solve.

## Differentiability boundary

The production forward simulator and the differentiable simulator share
dynamics, materials, and constraint equations, but they are not falsely
presented as the same mathematical map everywhere.

### Supported derivatives

- Smooth kinematics, RNEA/CRBA/ABA, actuation, passive forces, integration, and
  signed-distance derivatives.
- A converged `quality_newton` solve inside a fixed contact/constraint topology.
  The backward pass uses implicit differentiation: solve the transposed
  generalized Hessian system for the adjoint instead of unrolling solver
  iterations.
- JVP and VJP with respect to state, controls, masses, inertias, joint
  parameters, friction/compliance parameters, and selected geometry
  parameters.
- An MLX custom primitive that encodes forward and adjoint Metal kernels into
  MLX’s active command stream. MLX officially supports GPU primitives with
  custom JVP/VJP rules
  ([MLX extension API](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)).

### Two honest gradient modes

1. `piecewise_exact` freezes the detected manifold and active generalized
   projection region for the local derivative. It differentiates the actual
   quality step but is valid only while those discrete choices do not change.
2. `smooth_contact` uses a padded proximity candidate set, smooth force onset,
   compliant normal response, and regularized friction. It disables
   restitution, sleep/wake, hard clipping, and event-driven CCD. This is the
   mode for system identification and differentiable policy work.

The runtime marks gradients invalid or one-sided at contact creation/removal,
GJK/EPA feature changes, friction cone apex and stick/slip transitions,
limit activation, impact/restitution events, CCD event-order changes,
sleep/wake, and capacity changes. These are real nonsmooth boundaries, not
implementation bugs. MuJoCo’s own derivative documentation notes that its
default contact onset is not differentiable without changing impedance
([MuJoCo derivatives](https://mujoco.readthedocs.io/en/stable/computation/index.html#derivatives)).

## Metal execution architecture

The engine is compiled around immutable model topology and capacity classes:

- structure-of-arrays storage with environment as the outer batch;
- GPU-resident model, state, collision, constraint, warm-start, observation,
  reward, reset, and diagnostic buffers;
- static BVHs and model constants shared across cloned environments;
- per-environment dynamic ranges produced by prefix sums, never host loops;
- size-bucketed kernels for small direct solves, large PCG islands, and TGS;
- one SIMD-group per small articulation traversal and one or more threadgroups
  per constraint island, rather than permanently binding an entire world to
  one serial thread;
- indirect dispatch for active worlds/islands and sleeping compaction;
- no command-buffer completion or CPU-visible read in a normal rollout step;
- an MLX array/primitive boundary rather than a NumPy synchronization boundary.

All memory is preflighted against the Metal device’s maximum buffer length and
recommended working-set size. Runtime telemetry reports current and peak pair,
contact, row, island, Newton workspace, and total bytes.

## Build order and release gates

The order below preserves “Franka first, then G1” while building generic engine
pieces rather than one-off robot kernels.

### S0 — Numerical spine

- Generalize the fixed-size ABI to `nq != nv`, multi-DOF joints, free roots,
  free bodies, and multiple trees.
- Finish shared FP64/FP32 RNEA, CRBA, ABA, Jacobians, SO(3) integration, and
  implicit drives.
- Add model/state serialization, per-step status, and CPU-reference tracing.

**Exit:** all dynamics and free-motion metrics below pass for fixed Franka,
floating synthetic trees, and randomized free bodies.

### S1 — Collision spine

- GPU broad phase, filters, analytic primitives, convex GJK/EPA, manifolds,
  stable IDs, and capacity diagnostics.
- Static mesh/heightfield BVHs and compound convex bodies.
- Contact visualization/debug dumps consume the same buffers but are not in
  the headless step.

**Exit:** the collision and manifold metrics pass with no false negatives or
silent overflow in the canonical corpus.

### S2 — Franka constraint-quality milestone

- Unified constraint blocks and material mixing.
- `reference_fp64` and `quality_newton` with exact elliptic cones, limits,
  equalities, loops, warm starting, residuals, and point manifolds.
- A free cube and tool objects; Franka push, stable grasp, lift, slide, and
  insertion scenes.

**Exit:** Franka manipulation and solver-quality metrics pass. This is the
first point where MetalRobo can credibly claim a serious robotics contact
solver, but not yet a complete locomotion engine.

### S3 — G1 and throughput TGS

- Floating-base Unitree G1, 29 actuated joints, self-collision filters,
  actuator/armature semantics, foot patches, IMU quantities, and terrain.
- `throughput_tgs`, graph coloring, island bucketing, sleeping, and fixed-budget
  diagnostics.
- Flat standing, commanded velocity, impacts/falls, stairs, and rough terrain.

**Exit:** G1 stability, contact, determinism, and throughput targets pass while
the same scenes remain within the declared error from `quality_newton`.

### S4 — CCD and hard cases

- Hybrid conservative-advancement plus speculative CCD.
- High mass-ratio stacks, fast thin objects, kinematic tools, compounds, loops,
  rolling/torsional friction, and optional area-aware contact.

**Exit:** the CCD and stress suite passes without tunneling, lost time,
unreported overflow, NaNs, or solver failure.

### S5 — Differentiable Metal step

- `piecewise_exact` and `smooth_contact` forward modes.
- Implicit adjoint, MLX custom JVP/VJP/vmap, checkpointed rollouts, and gradient
  validity flags.

**Exit:** derivative metrics pass away from declared event boundaries, and an
end-to-end system-identification example converges on Metal without NumPy or
CPU stepping.

### S6 — Evidence release

- Pin engine commits and exact Franka/G1 assets.
- Publish every scene, parameter, seed, raw result, trace schema, hardware/OS
  identity, and competitor configuration.
- Run the same harness against MuJoCo 3.10 CPU and Genesis Metal locally, plus
  PhysX 5.9 and Drake SAP on their supported reference machines.
- Add at least one instrumented real Franka contact dataset for parameter and
  sim-to-real validation. Simulator-to-simulator agreement alone cannot prove
  physical accuracy.

**Exit:** only after the claim rules at the end of this document are met.

## Decisive acceptance metrics

These are release gates, not thousands of low-value unit tests. Each metric has
a canonical scene and machine-readable result.

### Dynamics and integration

| Check | `reference_fp64` | Metal quality |
| --- | ---: | ---: |
| Inverse/forward residual \(\|\tau-(M\dot v+c)\|/(1+\|\tau\|)\), randomized valid trees | \(<10^{-10}\) | \(<5\times10^{-5}\) |
| ABA result versus factored CRBA solve, normalized RMS | \(<10^{-11}\) | \(<5\times10^{-5}\) |
| Mass-matrix symmetry error and minimum-eigenvalue validation | \(<10^{-12}\), positive | \(<2\times10^{-6}\), positive after scaling |
| One-step CPU/Metal free-motion state error | reference | \(q<5\times10^{-5}\), \(v<10^{-4}\) normalized |
| Torque-free body angular-momentum drift over 10 s at 1/1000 s | \(<10^{-7}\) | \(<2\times10^{-3}\) |
| Midpoint convergence under timestep halving | measured order \(1.9\)–\(2.1\) | measured order \(1.8\)–\(2.2\) |

### Constraint solution and physical behavior

For the quality objective define a scaled optimality residual

\[
r_{\mathrm{opt}} =
\frac{\|\nabla\ell(v)\|_\infty}
 {1+\|A(v-v^*)\|_\infty+\|J^\mathsf{T}\gamma\|_\infty}.
\]

The terms are first nondimensionalized with the model compiler’s recorded
typical mass, length, and time scales; the `1` above is therefore
dimensionless.

For each elliptic contact let
\(D_\mu=\operatorname{diag}(1/\mu_1,1/\mu_2)\). Define

\[
r_{\mathrm{cone}} =
\max_i
\frac{\max(0,-\gamma_{n,i},
\|D_{\mu,i}\boldsymbol{\gamma}_{t,i}\|_2-\gamma_{n,i})}
 {1+\|\gamma_i\|_\infty},
\]

with the frictionless case evaluated as
\(\|\boldsymbol{\gamma}_t\|/(1+|\gamma_n|)\). For hard bilateral rows let
\(S_v\) be the recorded diagonal characteristic-velocity scale for each row
and define

\[
r_{\mathrm{eq}} =
\|S_v^{-1}(J_\mathrm{eq}v+b_\mathrm{eq})\|_\infty .
\]

Also report penetration or expected compliant deformation, momentum balance,
iteration count, final cost, condition estimate, regularization retries, and
the worst block ID per island.

| Check | Required result |
| --- | --- |
| Quality solve | `reference_fp64`: \(r_\mathrm{opt}<10^{-10}\); Metal: \(r_\mathrm{opt}<10^{-5}\), \(r_\mathrm{cone}<2\times10^{-5}\), and \(r_\mathrm{eq}<2\times10^{-5}\). No successful status above tolerance. |
| Direct versus PCG | Impulses and `v` agree within \(2\times10^{-5}\) normalized on all islands both solve. |
| TGS approximation | \(r_\mathrm{opt}<10^{-3}\) in the standard budget and task return within 2% of quality mode; exceptions are reported scene-by-scene. |
| Bilateral drift | In the 60 s loop/weld suite: \(<20\,\mu\mathrm{m}\) translation and \(<20\,\mu\mathrm{rad}\) rotation in quality mode; \(<0.5\) mm and \(<0.5\) mrad in TGS. |
| Friction | Incline breakaway coefficient within 2%; exact-cone response varies by \(<1\%\) as the tangent basis rotates; resting slip \(<0.1\) mm/s under sub-cone load. |
| Impact | Linear/angular momentum error \(<10^{-4}\) scaled for closed islands; measured restitution within 2% above its activation threshold and no bounce below it. |
| Mass ratio | Stable finite 60 s stacks and articulated contacts from \(10^{-4}\) to \(10^4\) mass ratio at the published timestep, with no constraint divergence and deformation within the configured compliance plus 0.25 mm. |
| Warm start | Same converged solution as zero start and at least 2× lower median Newton/PCG work in persistent resting/manipulation scenes; stale cache never raises initial objective above the zero start. |

### Collision and CCD

| Check | Required result |
| --- | --- |
| Distance/normal corpus | No missed overlap in the nondegenerate analytic/convex corpus; distance error \(<10^{-5}\) m and normal-angle error \(<10^{-4}\) rad versus FP64 reference outside documented degeneracies. |
| Manifold persistence | Resting box/mesh contacts retain stable IDs for \(>99\%\) of unchanged frames; refresh never leaves an anchor outside its shape tolerance. |
| Pair/contact capacity | Exact high-water marks reported; deliberate overflow returns the required size and leaves no world marked successful. |
| CCD | Zero tunneling in 100,000 seeded primitive/convex/mesh trials with per-step travel up to 100× minimum thickness; TOI error \(<10^{-4}\) of the frame or \(10^{-5}\) s, whichever is larger. Event exhaustion is observable and never drops time silently. |

### Robotics behavior

| Scene | Required result |
| --- | --- |
| Franka free-space | Inverse-dynamics hold and commanded trajectories agree with FP64 reference within 0.05° RMS and 0.5 mm end-effector RMS over 30 s. |
| Franka manipulation | Push, grasp/lift, sustained hold, sliding transition, and peg insertion complete for the pinned controller/seed suite; a held 1 kg cube drifts \(<1\) mm and \(<0.2^\circ\) over 30 s without an artificial weld. |
| G1 | A pinned controller stands 60 s, tracks the standard flat-ground command suite, recovers from declared pushes, falls without tunneling/exploding, and keeps stance-foot unintended slip below 5 mm per 10 s interval. |
| Cross-simulator | Contact-free motion agrees with pinned MuJoCo to the integration tolerance. Contact scenes compare impulses, energy, deformation, slip, and task outcome under matched physical parameters; raw trajectories are published rather than forcing unlike contact models to coincide. |

### Determinism and derivatives

| Check | Required result |
| --- | --- |
| Deterministic replay | 100 repeats of every canonical scene produce identical per-step 128-bit state/contact/impulse hashes on the same executable and device. Adding a noninteracting world does not change existing-world hashes. |
| JVP/VJP adjoint identity | Relative error \(<2\times10^{-4}\) on Metal and \(<10^{-9}\) in FP64 away from declared nonsmooth events. |
| Finite-difference agreement | Median relative error \(<10^{-3}\), 99th percentile \(<10^{-2}\) over conditioned smooth-contact scenes; invalid-event samples are counted, not discarded silently. |
| Long-horizon gradient | A 256-step checkpointed rollout matches the FP64 directional derivative within 2% and has bounded, reported adjoint residuals. |

### M4 performance floor

The performance gate uses the repository’s Apple M4 machine, headless release
build, 4 physics substeps per control step, device-resident policy-compatible
buffers, and no per-step host synchronization:

| Workload | Minimum sustained rate |
| --- | ---: |
| 4,096 Franka free-space environments | 150,000 environment control-steps/s |
| 1,024 Franka + one free object, up to 32 active contacts/environment | 40,000 environment control-steps/s |
| 2,048 G1 flat-ground environments, up to 64 active contacts/environment | 25,000 environment control-steps/s |

Each result includes p50/p95 step latency, active pairs/contacts/rows,
iterations, peak working-set bytes, thermal state, timestep, substeps, and
solver status counts. Throughput is invalid if a capacity overflow, contact
drop, CPU physics fallback, or hidden synchronization occurred.

## Claim rules

MetalRobo may say **“state-of-the-art robotics rigid-body solver on Apple
silicon”** only when:

1. S0–S6 are complete and every decisive gate above has a published result.
2. At matched scenes, assets, timestep, observation/reset work, and declared
   error, MetalRobo’s geometric-mean throughput is at least 20% above the
   pinned Genesis Metal version, or it has at least 2× lower physical/numerical
   error while staying within 10% of its throughput.
3. It is Pareto-nondominated across the published manipulation and locomotion
   suite: no compared engine is both faster and more accurate on a majority of
   the canonical workloads.
4. MuJoCo, Drake, Genesis, and PhysX comparisons include failures and
   unsupported features, not only wins.
5. The FP64 oracle, source, scenes, raw traces, and benchmark harness are
   reproducible by an external user.

MetalRobo may say **“complete robotics rigid-body engine”** only after free and
floating bodies, all constraint types, collision shape coverage, persistent
manifolds, CCD, deterministic replay, URDF and MJCF ingestion, force/contact
sensing, and Franka/G1 end-to-end scenes ship in the public repository.

It may not say **“Isaac Sim replacement”** until rendering, cameras, ray/LiDAR,
asset workflows, domain randomization, task composition, and deployment tools
are separately implemented and benchmarked. That platform work is outside this
solver target.

State of the art is a dated, evidence-backed claim. Every release that uses the
phrase must repeat the pinned comparison; this document alone never grants it.
