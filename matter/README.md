# Numi Matter

Numi Matter is a compiled multiphysics subsystem for Numi Lab. It keeps material semantics separate from numerical representation, then cooks a fixed-capacity Apple-GPU execution graph for the exact world being simulated.

The compiler and numerical implementation remain self-contained under `matter/`, while the runtime is integrated into the production `MetalWorld` command graph through a typed device-physics program. Matter and rigid dynamics now share one borrowed command buffer, one global body-wrench arena and one transactional control-step publication boundary.

## Implemented systems

### Matter Language

`.nmatter` files define named parameters with SI units, admissible ranges, optional log scaling and identifiability, persistent internal state, stored energy, a rate-dependent dissipation potential, explicit next-state laws, validity margins, supported representations, interface properties and physical limits.

```text
material damageable_silicone {
    parameter density : kg/m^3 = 1100 in [900, 1300];
    parameter mu : kPa = 150 in [5, 1000] log identifiable;
    parameter lambda : MPa = 3 in [0.1, 30] log identifiable;
    parameter eta : Pa*s = 1500 in [10, 10000] log identifiable;
    parameter damage_rate : one/s = 20 in [0, 200];

    state damage : one = 0;
    state accumulated_strain : one = 0;

    energy = neo_hookean(mu * (1 - damage), lambda * (1 - damage));
    dissipation = 0.5 * eta * (
        pow(D(0, 0), 2) + pow(D(1, 1), 2) + pow(D(2, 2), 2)
    );
    update accumulated_strain = accumulated_strain + dt() * sqrt(
        pow(D(0, 0), 2) + pow(D(1, 1), 2) + pow(D(2, 2), 2)
    );
    update damage = clamp(damage + dt() * damage_rate *
        max(sqrt(I1() / 3) - 1.01, 0), 0, 0.95);

    valid = J() - 0.05;
    supports mpm, fem, rigid;
}
```

The parser performs dimensional analysis before producing executable physics. Stored energy must have pressure/energy-density units; dissipation must have power-density units. `F`, `D`/`Fdot`, `dt`, temperature, parameters and material state are separate typed inputs. `D` and `Fdot` are aliases for the material deformation-gradient rate `Fdot = dF/dt`: FEM evaluates it directly from rest-coordinate velocity gradients, while MPM reconstructs it as `L F` from the APIC spatial velocity gradient `L`. Logarithms require dimensionless arguments. State updates must preserve the declared state dimension. Parameter ranges, determinant limits, fixed state capacity and expression-stack requirements are compile-time contracts.

### Physics IR and constitutive compiler

The compiler lowers material expressions into a dimensioned scalar DAG. It symbolically derives the elastic and dissipative responses

```text
P_elastic = dPsi(F, z, theta) / dF
P_viscous = dPhi(D, z, theta) / dD
```

and their matrix-free directional tangents. Every authored next-state expression is also compiled into the same bounded bytecode. MPM particles and FEM tetrahedra therefore execute one authoritative material program for stress, rate response, validity and state evolution rather than maintaining backend-specific constitutive implementations. Known models may additionally emit scene-specialized Metal source, while the bytecode path remains authoritative and fingerprinted.

### Matter compiler

`compileWorld` combines material programs, object geometry, rigid contact proxies and execution constraints into a pointer-free `CompiledWorld`. Cooking performs:

- deterministic content fingerprinting;
- fixed-capacity buffer sizing;
- representation selection;
- material parameter, initial-state and state-evolution program construction;
- power-of-two timestep planning from material wave speed and object scale;
- sparse MPM grid/block domains, Morton lookup and active-work capacities;
- FEM tetrahedron, rest-matrix, lumped-mass and node-incidence compilation;
- continuum/rigid contact-pair and gather-incidence compilation;
- adaptive-transfer and identification program construction;
- binary package serialization with typed sections.

`writePackage` and `readPackage` use a versioned `.nmatterpack` format. No pointers, runtime names or source-language expressions enter the GPU ABI. A single canonical layout validator is shared by compilation, package writing, package loading and runtime initialization. It statically proves scalar-stack contracts, non-overlapping material arenas, sparse MPM block lookup, FEM ownership/incidence, contact ownership, adaptive bindings and the complete execution fingerprint before any Metal resource is allocated.

### MPM backend

The MPM path is an updated-Lagrangian sparse APIC/MLS-MPM solid backend with quadratic B-spline support over 27 nodes per material point. The cooker builds fixed-capacity 8×8×8 block domains and Morton-ordered lookup tables. Runtime kernels classify particles into active blocks, form deterministic per-block particle ranges, dispatch only active nodes, perform APIC particle-to-grid transfer, apply constitutive and contact forces, and reconstruct particle velocity, affine motion and deformation from the grid.

The active-block pipeline avoids global floating-point scatter atomics, retains deterministic reduction order and prevents inactive authored grid regions from consuming transfer work. Persistent material state is stored per particle with accepted, candidate and control-step checkpoint arenas. Stress, validity and state evolution use the same live deformation and material deformation-rate values, and a rejected enclosing rigid transaction restores both mechanical and material state byte-for-byte. This backend targets large-deformation solids; it is not presented as a general free-surface fluid implementation.

### FEM backend

The FEM path uses linear tetrahedral kinematics with nonlinear constitutive stress. Its authoritative implicit update is an environment-wide semismooth Newton solve. Each Newton system is applied matrix-free and solved by restarted, right-preconditioned FGMRES over one generalized unknown containing nodal velocity/pressure, thermal-pore-electric-activation fields, analytic-proxy contact multipliers and dynamic deformable-contact multipliers:

```text
[ mechanics  mechanics-field  J_rigid^T  J_deform^T ] [dv, dp]
[ field-mechanics   transport       0           0   ] [dfields]
[ J_rigid              0       natural-map          ] [dlambda_rigid]
[ J_deform             0            friction-map    ] [dlambda_deform]
```

Element work is parallel and node assembly uses cooked incidence. Dynamic contact additionally builds a deterministic contact-corner-to-node CSR from stable compacted row order, so node residual and Jacobian gathers scale with actual incident rows rather than scanning contact capacity. FGMRES uses compensated SIMD32 reductions, modified Gram-Schmidt with selective reorthogonalization, device Givens rotations, restart cycles and an inexact-Newton forcing schedule. Its three-level right preconditioner combines fine node-star mechanics diagonals, overlapping connectivity-aware tetrahedron-patch corrections, an object-scale Galerkin correction for rigid translation and mean pressure, a bounded matrix-free polynomial field smoother, contact-space response diagonals, and an approximate contact-to-mechanics cross lift. FGMRES remains the sole convergence owner. One environment-wide line search combines constitutive determinant and mixed-volume bounds with a contact fraction-to-boundary cap, keeping every cross-object block on the same feasible Newton candidate.

Each tetrahedron owns an independent persistent material-state record. Rate-dependent stress and exact tangent-vector evaluation use the element velocity gradient. For stateful FEM materials, every Newton residual evaluates the authored next-state map at the current candidate, and the matrix-free tangent adds the local state-chain directional contribution. State updates still publish only with an accepted nonlinear candidate, and nodal, field, constitutive, topology and contact-warm-start state roll back to the same control-step checkpoint.

### Unified continuum contact

MPM grid nodes and FEM nodes use one contact-pair ABI. The current implementation supports plane, sphere, capsule and oriented-box rigid proxies. Each contact computes:

- signed separation and witness point;
- normal and tangential relative velocity;
- compliant normal impulse;
- Coulomb stick/slip projection;
- equal and opposite continuum and rigid reactions;
- fixed-slot contact evidence for events and diagnostics.

Contact records are cleared on every microtick before evaluation, preventing inactive-rate domains from replaying stale impulses. Analytic-proxy rows use the full cooked sparse Delassus response, including same-command-buffer inverse-ABA columns for articulated bodies.

FEM surfaces additionally compile immutable exposed-face ownership and adjacency. Each nonlinear candidate rebuilds current/cohesive surface primitives, swept AABBs and a deterministic GPU Morton ordering, then emits fixed-capacity cross-object and non-adjacent self-contact candidates. Conservative-advancement vertex-triangle and edge-edge CCD produce dynamic KKT rows. Their normal and two tangent multipliers use a semismooth projection natural map with a Coulomb disk; tangent warm starts are transported between contact frames and participate in checkpoint, commit and rollback. Stable active slots are flattened across environments into one GPU-authored work list and indirect dispatch record, eliminating capacity-wide downstream contact launches without a CPU count readback.

### Rigid-to-continuum coupling

Rigid reactions are reduced per proxy without floating-point atomics, then gathered by global rigid-body index through one deterministic writer per body. Matter streams point-impulse columns through MetalWorld's inverse ABA on the borrowed command-buffer timeline and gathers the resulting articulated Delassus response `J M^-1 J^T` into the same generalized contact operator:

- articulated targets receive `MRABABodyWrenchGPU` force and torque before inverse ABA; their exact link twists are materialized from the same accepted `q/v` consumed by ABA rather than estimated from frame differences;
- free scene bodies receive the same global body-wrench representation, which the scene predictor consumes during the rigid candidate step;
- multiple contact proxies may contribute to one body without aliased writes;
- static unbound proxies participate in contact but receive no state update.

The runtime consumes borrowed body, wrench, scene-state and environment-status arenas. It never commits or waits on the borrowed command buffer. Device-physics capabilities distinguish programs that write external body wrenches from programs that require accepted rigid-contact evidence. Continuum-only contact can therefore drive free-motion ABA or free scene bodies without constructing an unused rigid contact graph, while adaptive promotion still requires the post-solve contact arena. A Matter failure is translated into the enclosing `MetalWorld` status before rigid publication, and a later rigid failure restores MPM, FEM, persistent material state, scheduler state and event evidence to the same control-step checkpoint. Inverse-identification update and antithetic sampling execute at most once at the beginning of a fully encoded rollout command buffer; all later control steps in that submission consume the same candidate overlay.

### Adaptive representation

Adaptive objects retain both continuum topology and a body-backed rigid proxy. The runtime publishes explicit collision ownership on every transition, so the rigid fallback collider is disabled while the continuum representation is active and enabled only while the object is rigid. The GPU measures:

- total mass and centre of mass;
- linear and angular momentum;
- inertia and inverse inertia;
- average deformation rotation;
- peak and RMS strain;
- minimum determinant and deformation residual.

Hysteresis controls demotion to rigid representation after a sustained low-strain interval. State transfer preserves centre of mass, linear momentum, angular momentum and the measured full inertia tensor. Demotion publishes an environment-specific world inverse inertia that MetalWorld validates and rotates through ordinary and CCD integration instead of replacing it with a mass-scaled authored tensor. The compiler retains an immutable mass-weighted rest centre for every continuum object; promotion reconstructs positions from that rest frame and writes the current rigid orientation as the deformation rotation, avoiding translation leakage or inertia-derived orientation. Adaptive bindings must be valid, body-backed and unique; the compiler rejects ambiguous mappings.

After a demotion, the borrowed final `MetalWorld` contact arena becomes the
contact authority for that fallback body. Matter scans the accepted constraint
prefix, rejects generalized and disabled rows, and maps pre-solve normal speed
and solved impulse only to the uniquely bound adaptive body. That signal
re-promotes the authored continuum before collision ownership is republished,
so unrelated rigid contacts cannot wake a dormant adaptive object. The
dedicated GPU probe validates this mapping with a typed accepted post-solve
contact record; the adapter forwards the corresponding MetalWorld arena in a
full coupled submission.

### Inverse material identification

Identifiable material parameters receive a GPU-resident distribution. Candidate environments use deterministic antithetic normal perturbations, optional log-space sampling and bounded parameter overlays. After candidate losses are written into the exposed loss buffer, the update kernel performs temperature-weighted distribution fitting and republishes the learned mean and variance. Automatic candidate perturbation is disabled for ordinary rollout and is enabled explicitly through `RuntimeConfiguration::automaticIdentification`; otherwise every environment uses the current identified mean.

The compiler requires an even candidate count no larger than the environment count, so every candidate is actually simulated and every antithetic partner exists.

### Event-driven multirate scheduler

Every continuum object has a power-of-two rate exponent. The fixed command graph contains the maximum number of microticks; each object executes only on its scheduled ticks. GPU event reduction observes contact onset/release, slip, strain/yield, damage, determinant risk, solver residual and requested rate changes. Events raise local substep frequency immediately and lower it only after a quiet-frame hysteresis period.

Events are written to deterministic fixed slots rather than an unordered append queue. Every token carries absolute episode time, time since that object's previous event, severity, and the previous/current signal values. Timing advances even when event readback is disabled, so later event deltas remain meaningful. They can be consumed by Numi Lab's event-token learning layer without CPU synchronization.

## Apple Silicon execution model

The runtime is designed around Apple GPU constraints rather than CUDA assumptions:

- immutable cooked tables and persistent state use `MTLStorageModePrivate`;
- unified/shared memory is used only for initialization, explicit status/event publication and identification losses;
- one borrowed command buffer contains identification, parameter overlay, all microticks, constitutive-state evolution, contact, rigid coupling, adaptive transfer and event reduction;
- no CPU counter read, command-buffer commit or wait occurs inside `Runtime::encode`;
- fixed-capacity deterministic gathers replace floating-point append/scatter atomics;
- SIMD32 object reductions map to Apple GPU execution width;
- deformable surfaces and contact rows are compacted into stable active lists, deterministic node-incidence CSR, and GPU-authored indirect dispatches before downstream KKT work;
- Krylov vector arenas are private and sized to the authored restart depth rather than the compile-time maximum;
- environment and object parallelism remain independent;
- all runtime identities are content-fingerprinted, including the exact loaded Matter metallib;
- the rigid microstep duration must exactly match the cooked Matter duration, preventing a caller from silently changing constitutive or multirate semantics after fingerprinting.

## Build

From the Numi Lab repository root:

```bash
cmake -S matter -B build-matter -G Ninja
cmake --build build-matter -j
```

The compiler and package tools are portable C++23. The runtime requires Apple Silicon macOS and Metal 4.0. With the Apple runtime enabled, the module builds:

```text
libnumi_matter_compiler.a
libnumi_matter_runtime.a
NumiMatter.metallib
numi-matterc
```

Compile the included material and a cooked MPM/FEM demonstration world:

```bash
build-matter/numi-matterc \
  --material matter/materials/silicone.nmatter \
  --output silicone.nmatterpack \
  --generated-metal silicone.generated.metal \
  --mode both \
  --envs 16
```

## Numi Lab integration boundary

Initialize the cooked Matter world once, then attach its adapter to the ordinary `MetalWorldStepConfig`:

```cpp
numi::matter::Runtime matterRuntime;
const auto matterDiagnostics = matterRuntime.initialize(
    compiledMatterWorld,
    {.metallib = matterMetallib}
);

metalrobo::MetalWorldStepConfig stepConfig;
stepConfig.devicePhysicsProgram =
    numi::matter::makeMetalWorldDevicePhysicsProgram(matterRuntime);
```

`MetalWorld` invokes the adapter twice for every rigid physics substep. The pre-dynamics phase runs continuum microticks, accumulates equal-and-opposite reactions into the global body-wrench arena and latches Matter failures before ABA/contact publication. The post-commit phase reconciles rigid failure, performs final-step adaptive transfer, publishes collision ownership and emits event tokens. The adapter receives only borrowed device resources; it cannot commit, wait or retain the command buffer.

A body-backed rigid proxy requires the global current-body arena. Dynamic free-body proxies also carry an explicit scene-body index. Adaptive bindings are restricted to unique dynamic scene bodies so promotion and demotion always have one writable rigid authority.

## Current fidelity boundary

The subsystem implements the architecture and executable kernels listed above, but it does not claim one numerical representation solves all matter:

- the MPM path currently targets updated-Lagrangian APIC/MLS-MPM solids on a fixed-capacity sparse block domain;
- the generalized Krylov vector includes FEM mechanics, mixed fields, and active MPM grid velocity, but the MPM block currently contains lumped inertia and analytic-rigid barrier curvature rather than the particle constitutive tangent or FEM/MPM contact;
- internal state and rate-dependent dissipation are executable; authored implicit residuals use a damped local Newton solve and consistent condensed tangent while explicit next-state bytecode remains a compatibility path;
- deformable contact has swept broadphase, non-adjacent self-contact, vertex-triangle/edge-edge CCD and a primal logarithmic barrier, but its PSD rank-one normal curvature is not the full mollified IPC Hessian and it does not claim an unconditional non-intersection theorem;
- articulated continuum contact consumes same-command-buffer inverse-ABA response columns in the full sparse Delassus operator and certifies the circular cone and natural map;
- cut/puncture, cohesive separation, erosion and node-incidence rebuild are transactional; capacity exhaustion publishes a geometric replacement-runtime request after completion, while conservative split/collapse/flip remeshing and material-state transfer remain outside the current operator;
- thermal, pore, electric and activation fields execute in the outer KKT operator with Joule, activation and Biot off-diagonal actions; their right preconditioner is intentionally an approximate fixed-pass transport smoother;
- polyconvex ICNN energy gradients and directional tangents execute with transactional trained weights; broad objectivity, growth and adversarial material-oracle qualification remains later evidence work;
- the new monolithic/deformable-contact path has received only compile-level verification in this development pass; long GPU probes, matched performance profiling and qualification gates are intentionally deferred.

## Validation

The portable compiler and package code is built with:

```text
-std=c++23 -Wall -Wextra -Wpedantic -Werror
```

The included elastic and damageable-viscoelastic materials compile into worlds containing both MPM and FEM objects, write versioned packages, emit specialized Metal and round-trip through `readPackage`. The portable stateful regression evaluates compiled elastic stress degradation, viscous stress/tangent, explicit state evolution and canonical fingerprint sensitivity. It also corrupts material-state offsets, scalar-program spans, sparse block lookup, tetrahedron topology, contact incidence and identification ownership and requires a precise rejection from the same validator used by production loading. Source-level audits additionally check shared-ABI sizes, kernel names, buffer bindings and delimiter balance.

On an Apple Silicon Metal 4 build host, run the physics qualification suite
after building the probe target:

```bash
cmake -S . -B build -G Ninja
cmake --build build --target metalrobo_matter_physics_probe numi-matter-stateful-check -j 3
ctest --test-dir build -L matter --output-on-failure -j 1
```

The suite executes the actual `NumiMatter.metallib` through borrowed Metal
command buffers. It checks MPM freefall, a coupled eight-particle MPM plane
impact, stateful MPM and FEM evolution with exact rejected-transaction rollback, an articulated-body continuum reaction consumed by MetalWorld ABA, a
full-height implicit FEM impact, a near-plane CFL-resolved FEM contact,
byte-identical MPM/FEM/scheduler rollback after an enclosing rigid transaction
rejects the tentative continuum update, byte-identical adaptive/scheduler
rollback after a rejected rigid-contact promotion, a 30-frame MPM-to-rigid ownership
transfer with valid measured inverse inertia and scene authority publication,
the inverse rigid-contact transfer back to MPM through the typed borrowed
post-solve arena, and antithetic inverse-parameter sampling followed by a GPU
posterior update from asymmetric candidate losses.
It is deliberately serial because all cases use the active GPU.

This is continuum and analytic-proxy contact evidence, including one
articulated rigid-reaction handoff through a full `MetalWorld` step. It also
qualifies the inverse-identification kernel and its explicit loss-buffer
contract—not a material-calibration result. It does not qualify longer-running
material calibration or physical material identification.
