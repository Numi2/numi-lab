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

    state damage : one = 0 transfer max;
    state accumulated_strain : one = 0 transfer max;

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

State declarations accept `transfer average|max|sum` (volume-weighted average by default). A state may define either `update name = expression;` or `implicit name = residual;`; `name` reads accepted state and `next(name)` reads the local candidate. Implicit programs compile residual/Jacobian/deformation/stress-state derivative bytecode, run bounded pivoted local Newton per particle or tetrahedron, and contribute `P_F - P_z R_z^-1 R_F` to the matrix-free operator. Von Mises and Drucker-Prager hints select specialized multiplicative finite-strain return maps with isotropic hardening and consistent active-branch tangents. Local nonconvergence records `NM_STATUS_LOCAL_MATERIAL_FAILURE` and rolls back only that environment.

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

The active-block pipeline avoids global floating-point scatter atomics, retains deterministic reduction order and prevents inactive authored grid regions from consuming transfer work. A SIMD32 prefix pass compacts positive-mass grid nodes into the shared generalized vector. Backward-Euler residuals and matrix-free material actions transfer directions grid-to-particle and gather forces particle-to-grid in stable sparse-block order. Lumped grid diagonals, particle-patch smoothing, and object translation modes participate in the same right preconditioner as FEM and rigid blocks.

The cooker proves a per-environment compact-slot bound of `min(grid nodes, 27 * particles)` for each quadratic-support MPM object. Active-index storage, every Krylov vector, and every retained basis column use that compact capacity; only physical grid state and the inverse node-to-slot map retain the authored dense-grid stride.

Persistent particle, APIC affine, deformation, and implicit material state have accepted, candidate, and control-step checkpoint arenas. G2P reconstruction publishes them only after the global Newton candidate passes material, field, barrier, and residual acceptance. This backend targets large-deformation solids; it is not presented as a general free-surface fluid implementation.

### FEM backend

The FEM path uses linear tetrahedral kinematics with nonlinear constitutive stress. Its authoritative implicit update is an environment-wide Newton solve. Each Newton system is applied matrix-free and solved by restarted, right-preconditioned FGMRES over one generalized unknown containing FEM velocity and pressure, thermal/pore/electric/activation fields, active MPM grid velocity, and articulated/free-body generalized velocity increments:

```text
[ FEM/MPM mechanics   mechanics-field   continuum-rigid ] [dv_fem, dv_mpm]
[ field-mechanics         transport             0       ] [dfields]
[ rigid-continuum              0           rigid mass   ] [dv_rigid]
```

IPC's squared-distance logarithmic potential contributes primal gradients and PSD Hessian actions; per-node timestep ratios apply the action chain rule to cross-rate FEM/MPM rows. There are no contact multiplier unknowns, response CSR, Delassus rows, or post-contact correction solves. Element work is parallel and node assembly uses rebuilt incidence. FGMRES uses compensated SIMD32 reductions, modified Gram-Schmidt with selective reorthogonalization, device Givens rotations, restart cycles and an inexact-Newton forcing schedule. The environment-owned SIMD32 Arnoldi wave continues directly from orthogonalization into its norm, Givens update and next-basis publication. Restart-residual reconstruction and the tiny triangular solve likewise share one per-environment cycle-finalization dispatch, and the final cycle does not materialize coefficients that no later cycle can consume. These fusions retain the original reduction and arithmetic order while removing command traffic. The right preconditioner combines FEM node-star diagonals, overlapping tetrahedron patches, MPM lumped/particle-patch/object modes, field smoothing, and rigid inverse-mass action. FGMRES remains the sole convergence owner. One environment-wide line search combines constitutive determinant and mixed-volume bounds with conservative CCD, barrier fraction-to-boundary caps, and barrier Armijo backtracking.

ABI v23 marks an explicit needle capsule or circular arc as the puncture
dilator. The sharp tip remains the only fracture authority, while each local
mass-conserving channel segment records the largest flagged same-body gauge;
this lets the 0.35 mm-radius shank establish a physically passable tract for
the smaller PDO strand without allowing generic shaft or gripper contact to
initiate a cut.

ABI v22 added an analytic circular-tube arc for the curved needle shank and a
live DER strand capsule role. Strand proxy endpoints borrow MetalWorld's
resident rod-node arena, participate in the same IPC/FGMRES acceptance path as
the tissue, and scatter accepted equal-and-opposite contact impulse back to
the two DER nodes before rod integration. The Matter timestep must be an
integer grouping of the owning rod cadence, so tissue, needle, hard swage and
thread remain one deterministic device transaction. A live-strand runtime may
select a power-of-two coupled timestep multiplier only between submissions;
this retains accepted state while grouping that exact number of base DER
substeps and never runs hidden Matter microticks against frozen rod geometry.
Puncture admission sums
only closing, direction-aligned, tip-local surface rows: mesh refinement may
distribute a physical sharp-tip resultant across nodes, while shaft, arc and
strand rows cannot manufacture a cut.

ABI v21 added an explicit body-backed tapered-tip capsule role for
physics-triggered puncture. Shaft, swage, gripper and generic capsule contact
cannot create a tissue channel. A positive-clearance sharp-tip contact must
be closing, exceed the accepted impulse gate, align with the authored tip
direction, and project a finite inward entry tract through active tissue.
The resulting needle channel is an embedded, mass-conserving discontinuity:
it releases contact only for the originating needle inside the finite tract
instead of deleting every intersected FEM tetrahedron. Explicit authored
cylinder mutations retain their erosion semantics. As the admitted sharp tip
advances, the live terminal tangent grows a deterministic chain of overlapping
one-diameter segments; this follows curved needle motion without pre-cutting a
straight tract through the complete tissue wall.

ABI v20 removes the former standalone FEM, pressure, and field iteration
budgets and kernels. `MixedSolverSource` now cooks only Newton/FGMRES budgets,
one bounded `fieldSmootherPasses` preconditioner budget, mutation restarts, and
equilibrium/volume/pressure/transport residual tolerances. The certificate's
relative correction remains finite telemetry and is not a competing stopping
rule. `minimumContactSeparationRatio * WorldSource::contactSlop`, combined
with a coordinate-scale FP32 floor, is read by both Metal fraction-to-boundary
kernels and final candidate certification.

The FEM mechanical diagonal and every MPM fine, particle-patch, and
object-translation denominator include the componentwise diagonal of the same
PSD IPC blocks applied by the matrix-free operator. Deformable feature weights
and multirate chain factors enter quadratically, including the mollifier outer
product. This is a matrix-free right-preconditioner approximation, not a
retained contact matrix or an additional contact solve. The fine MPM pass
publishes its ephemeral componentwise denominator into operator scratch that
the patch and object modes consume before the matrix action overwrites it;
contact incidence is therefore traversed once per node and Krylov column with
no added arena, dispatch, or synchronization.

Each tetrahedron owns an independent persistent material-state record. Rate-dependent stress and exact tangent-vector evaluation use the element velocity gradient. For stateful FEM materials, every Newton residual evaluates the authored next-state map at the current candidate, and the matrix-free tangent adds the local state-chain directional contribution. State updates still publish only with an accepted nonlinear candidate, and nodal, field, constitutive, topology and primal-contact history roll back to the same control-step checkpoint.

### Unified continuum contact

MPM grid nodes and dynamically exposed FEM faces use one continuum-surface ABI. Plane, sphere, capsule, and oriented-box proxies use the same primal contact energy. Each candidate computes:

- signed separation and witness point;
- normal and tangential relative velocity;
- adaptive IPC barrier impulse and analytic gradient;
- a product-rule mollified gradient and PSD-projected spatial/Gauss-Newton
  Hessian action mapped into complete nodal blocks;
- transported lagged friction with a smooth static/dynamic transition;
- equal and opposite continuum and rigid reactions;
- fixed-slot contact evidence for events and diagnostics.

Contact records are cleared on every microtick before evaluation, preventing inactive-rate domains from replaying stale history. Conservative swept bounds and fail-closed CCD cover FEM/FEM, FEM self-contact, MPM/FEM, and MPM/MPM point/face or point/point candidates. Near-degenerate vertex-triangle and edge-edge features use a smooth IPC feature mollifier; candidate overflow and uncertified/nonfinite CCD invalidate only that environment.

Each nonlinear candidate derives exposed FEM faces from the current tetrahedron graph, adds compact active MPM grid points, builds swept AABBs and a deterministic GPU Morton order, and emits non-adjacent self/cross-object candidates. Stable source and transported tangent-potential history participate in checkpoint, commit, and rollback; no normal/tangent multiplier state remains. Stable active slots are flattened into a GPU-authored indirect work list without a CPU count readback.

### Rigid-to-continuum coupling

Rigid generalized increments are part of Matter's primal Krylov vector. MetalWorld remains the sole owner of ABA, generalized coordinates, body kinematics, and rigid publication; its borrowed coupled-candidate service supplies candidate kinematics, mass action, inverse-mass preconditioning, and accepted-candidate publication on the same command-buffer timeline:

- articulated targets use exact candidate link kinematics and ABA-owned generalized mass action;
- every articulated collision proxy shares the one v1 articulation reserve instead of multiplying generalized storage by proxy count;
- free scene bodies use six-coordinate velocity increments in the same block;
- multiple contact proxies may contribute to one body without aliased writes;
- static unbound proxies participate in contact but receive no state update.

The runtime consumes borrowed body, wrench, scene-state and environment-status arenas. It never commits or waits on the borrowed command buffer. Device-physics capabilities distinguish programs that write external body wrenches from programs that require accepted rigid-contact evidence. Continuum-only contact can therefore drive free-motion ABA or free scene bodies without constructing an unused rigid contact graph, while adaptive promotion still requires the post-solve contact arena. A Matter failure is translated into the enclosing `MetalWorld` status before rigid publication, and a later rigid failure restores MPM, FEM, persistent material state, scheduler state and event evidence to the same control-step checkpoint. Inverse-identification update and antithetic sampling execute at most once at the beginning of a fully encoded rollout command buffer; all later control steps in that submission consume the same candidate overlay.

### Transactional topology and arena growth

Mutable FEM objects cook dormant node/tetrahedron slots into private arenas. GPU commands support cohesive separation, plane/cylinder erosion, explicit deactivation, edge split/collapse, 2–3 and 3–2 flips, and vertex smoothing. `target = NM_INVALID_INDEX` requests a deterministic on-device quality proposal; stable priority, identifier, target, and source order resolve competing commands. Every accepted generation rebuilds mass, node/tetrahedron incidence, exposed contact faces, active contact work, and preconditioner connectivity.

Split, collapse, flip, and smoothing transfer the complete affected cavity according to each material state's `transfer average|max|sum` policy. Split interpolation and cavity projection preserve mechanical and mixed-field state. After rebuilt lumped mass, compensated object sums drive a constant correction plus one bounded residual refinement on free active nodes, closing momentum and each volume-integrated field component while retaining relative variation. Volume, mass, momentum, and field certificates scale with the represented quantity instead of a one-SI-unit floor; any representable increase in the monotone removal ledger is treated as erosion. The FP32 certificate rejects low-volume/inverted elements, invalid ledgers, or remaining conservation error before publication. Failed environments restore the common mechanical/material/topology checkpoint.

No shader allocates. Capacity exhaustion records `NM_STATUS_TOPOLOGY_GROWTH_REQUIRED` (or a contact-work capacity overflow), and completion publishes a geometric `TopologyGrowthRequest`. `encodeTopologyGrowth` either migrates into an already initialized compatible runtime or validates and allocates an empty destination from a larger recook before encoding migration in the borrowed command buffer. It advances allocation generation, rebuilds derived incidence/mass, and mirrors the rebuilt accepted state into candidate/checkpoint arenas. A canonical source-physics fingerprint excludes allocation-only capacities and must match across migration; each recook retains a distinct full package fingerprint, while snapshots record the accepted allocation generation for exact replay. Growth repeats across completed submissions until a 32-bit ABI or device working-set limit is reached.

Before derived-data rebuild, migration rebases every active tetrahedron and cohesive face from its source object's node/element offsets to the expanded destination offsets. Invalid global references fail closed, and the rebased cohesive state is mirrored with the other accepted topology arenas.

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
- continuum surfaces and primal barrier candidates are compacted into stable active lists, deterministic node incidence, and GPU-authored indirect dispatches before Krylov work;
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

The subsystem implements the owning paths listed above, with these explicit boundaries:

- the MPM path targets updated-Lagrangian APIC/MLS-MPM solids on a sparse block domain; it is not a free-surface fluid solver;
- the generalized Krylov vector contains FEM mechanics, mixed fields, compact active MPM grid velocity, and articulated/free-body generalized increments; MPM/FEM and continuum/rigid coupling enter through the same primal barrier action;
- internal state and rate-dependent dissipation are executable; authored implicit residuals use a damped local Newton solve and consistent condensed tangent, while specialized multiplicative von Mises/Drucker-Prager predictors update `Fp` and accumulated plastic strain transactionally with an active-branch algorithmic tangent;
- contact uses the IPC squared-distance logarithmic potential with mollified VT/EE and point features, a PSD-projected spatial Hessian, smooth lagged friction, multirate action scaling, and fail-closed FP32 conservative CCD; this is an executable FP32 contract, not an exact-real nonintersection theorem;
- MetalWorld retains sole ownership of ABA and `q/v`; Matter owns Newton sequencing and generalized increments through the typed candidate callback;
- split/collapse/flip/smooth/cohesive/erosion mutations are transactional, conservation-certified, and geometrically growable only between completed submissions;
- thermal, pore, electric and activation fields execute in the outer KKT operator with Joule, activation and Biot off-diagonal actions; their right preconditioner is intentionally an approximate fixed-pass transport smoother;
- polyconvex ICNN energy gradients and directional tangents execute with transactional trained weights; broad objectivity, growth and adversarial material-oracle qualification remains later evidence work;
- live Apple-M4 Metal evidence covers the complete serial Matter suite, strict MPM impact, articulated foot/pad contact, poroelastic compression and monolithic multiphysics; repeated long-horizon remeshing, matched external-solver comparisons, material calibration and physical-system evidence remain separate qualification work.

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
build/bin/metalrobo_matter_physics_probe --mpm-batch
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

The coupled production defaults use seven outer Newton corrections and one
ten-column FGMRES restart cycle. On the Apple M4 qualification host, the
24-test suite completed without failure in 299.09 seconds. The strict
eight-particle impact executed 1,024 material microsteps, used at most two
Krylov iterations, observed nine contacts, retained `minimum_J = 0.99742`,
reported a `4.53284e-06` KKT residual, and completed in 89.696 GPU seconds
without transaction rollback.
These measurements establish this implementation and workload on that host;
they are not a matched claim against another simulator or Apple GPU.

A controlled one-particle contact A/B on that host disabled only the IPC
diagonal. Enabling it reduced the Krylov high-water from four columns to two,
the reported KKT residual from `8.64169e-08` to `6.53475e-08`, and GPU time for
1,024 microsteps from 16.7051 to 15.0944 seconds. Both variants retained four
contacts, positive separation, and no rollback; this is finite-tolerance
solver evidence, not a bitwise-identical trajectory claim.

Reusing the fine-pass diagonal from operator scratch then reduced the same
enabled case from 15.0944 to 13.6992 GPU seconds. All printed physical and
certificate values were identical before and after that change; it removed
redundant incidence traversal without changing the preconditioner arithmetic.

The dedicated batch probe additionally completed 32 identical contact
environments and 8,192 environment-microsteps on one command-buffer timeline.
Two untraced final-code runs measured 121.471 and 121.251
environment-microsteps/second, a 121.361 mean, with 67.501 mean GPU seconds
and 9,486,496 retained bytes. The same-session pre-change baseline measured
103.817 environment-microsteps/second and 78.908 GPU seconds. Every final run
reported nine contacts per environment, `minimum_J = 0.999695`, no failed
environment, and a two-column FGMRES high-water. The observed candidate mean
is 16.90 percent faster with 14.46 percent less GPU time than that baseline;
it is a matched measurement on this host, not a universal speedup. The contact
diagonal changes no command count: the graph still encodes 189,473 dispatches
and requests 2,517,280 threadgroups. The ten-column default remains 60.8
percent below the former two-cycle dispatch graph and retains 12.3 percent
less memory than that build. Eight-
and nine-column defaults were rejected by the articulated rigid-reaction gate;
ten is the measured suite-wide minimum. This is native batching and
command-graph evidence, not an external frontier comparison.

This is continuum and analytic-proxy contact evidence, including one
articulated rigid-reaction handoff through a full `MetalWorld` step. It also
qualifies the inverse-identification kernel and its explicit loss-buffer
contract—not a material-calibration result. It does not qualify longer-running
material calibration or physical material identification.
