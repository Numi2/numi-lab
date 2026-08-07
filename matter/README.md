# Numi Matter

Numi Matter is a compiled multiphysics subsystem for Numi Lab. It keeps material semantics separate from numerical representation, then cooks a fixed-capacity Apple-GPU execution graph for the exact world being simulated.

The compiler and numerical implementation remain self-contained under `matter/`, while the runtime is integrated into the production `MetalWorld` command graph through a typed device-physics program. Matter and rigid dynamics now share one borrowed command buffer, one global body-wrench arena and one transactional control-step publication boundary.

## Implemented systems

### Matter Language

`.nmatter` files define named parameters with SI units, admissible ranges, optional log scaling and identifiability, internal state, stored energy, validity margins, supported representations, interface properties and physical limits.

```text
material silicone {
    parameter density : kg/m^3 = 1100 in [900, 1300];
    parameter mu : kPa = 150 in [5, 1000] log identifiable;
    parameter lambda : MPa = 3 in [0.1, 30] log identifiable;

    model neo_hookean;
    energy = neo_hookean(mu, lambda);
    valid = J() - 0.05;
    supports mpm, fem;

    interface {
        static_friction = 0.8;
        dynamic_friction = 0.65;
        restitution = 0.0;
        adhesion = 0.0;
    }
}
```

The parser performs dimensional analysis before producing executable physics. Stored energy must have pressure/energy-density units. Logarithms require dimensionless arguments. Parameter ranges, determinant limits and expression-stack requirements are compile-time contracts.

### Physics IR and constitutive compiler

The compiler lowers material expressions into a dimensioned scalar DAG. It symbolically derives all nine components of first Piola stress,

```text
P = dPsi / dF
```

and all nine matrix-free tangent-vector outputs,

```text
dP = (dP / dF) : dF
```

then emits bounded stack bytecode shared by MPM and FEM. This removes handwritten stress/Jacobian duplication between backends. Known models may also emit scene-specialized Metal source, while the bytecode path remains authoritative and fingerprinted.

### Matter compiler

`compileWorld` combines material programs, object geometry, rigid contact proxies and execution constraints into a pointer-free `CompiledWorld`. Cooking performs:

- deterministic content fingerprinting;
- fixed-capacity buffer sizing;
- representation selection;
- material parameter-table construction;
- power-of-two timestep planning from material wave speed and object scale;
- MPM stencil and node-incidence compilation;
- FEM tetrahedron, rest-matrix, lumped-mass and node-incidence compilation;
- continuum/rigid contact-pair and gather-incidence compilation;
- adaptive-transfer and identification program construction;
- binary package serialization with typed sections.

`writePackage` and `readPackage` use a versioned `.nmatterpack` format. No pointers, runtime names or source-language expressions enter the GPU ABI.

### MPM backend

The MPM path is a total-Lagrangian solid backend using fixed quadratic B-spline support over 27 nodes per material point. The cooker builds an object-owned fixed background-grid table and deterministic node-owned incidence lists. Runtime kernels perform:

1. particle-to-grid mass, PIC momentum and constitutive-force gathering;
2. grid acceleration and integration;
3. unified continuum contact projection;
4. grid-to-particle velocity and deformation-gradient updates;
5. transactional candidate publication.

Node-owned gathers avoid floating-point scatter atomics and produce stable accumulation order across runs. The current path deliberately uses PIC rather than an APIC affine update, and its cooked full background-grid table is a fixed-capacity execution contract rather than a sparse-grid implementation. This backend is intended for elastic, viscoelastic and elastoplastic solids with large deformation. It is not presented as a general free-surface fluid implementation.

### FEM backend

The FEM path uses linear tetrahedral kinematics with nonlinear constitutive stress. It computes element forces and a matrix-free tangent-vector operator, then solves the implicit velocity increment with fixed-iteration PCG entirely on the GPU:

```text
A p = M p - dt^2 (df_internal / dx) p
```

Element work is parallel. Node assembly uses cooked incidence. Per-object SIMD32 reductions produce deterministic `r.r` and `p.Ap` scalars without global floating-point atomics. The final state remains a candidate until environment status permits publication.

### Unified continuum contact

MPM grid nodes and FEM nodes use one contact-pair ABI. The current implementation supports plane, sphere, capsule and oriented-box rigid proxies. Each contact computes:

- signed separation and witness point;
- normal and tangential relative velocity;
- compliant normal impulse;
- Coulomb stick/slip projection;
- equal and opposite continuum and rigid reactions;
- fixed-slot contact evidence for events and diagnostics.

Contact records are cleared on every microtick before evaluation, preventing inactive-rate domains from replaying stale impulses.

### Rigid-to-continuum coupling

Rigid reactions are reduced per proxy without floating-point atomics, then gathered by global rigid-body index through one deterministic writer per body:

- articulated targets receive `MRABABodyWrenchGPU` force and torque before inverse ABA; their exact link twists are materialized from the same accepted `q/v` consumed by ABA rather than estimated from frame differences;
- free scene bodies receive the same global body-wrench representation, which the scene predictor consumes during the rigid candidate step;
- multiple contact proxies may contribute to one body without aliased writes;
- static unbound proxies participate in contact but receive no state update.

The runtime consumes borrowed body, wrench, scene-state and environment-status arenas. It never commits or waits on the borrowed command buffer. A Matter failure is translated into the enclosing `MetalWorld` status before rigid publication, and a later rigid failure restores MPM, FEM, scheduler state and event evidence to the same control-step checkpoint. Inverse-identification update and antithetic sampling execute at most once at the beginning of a fully encoded rollout command buffer; all later control steps in that submission consume the same candidate overlay.

### Adaptive representation

Adaptive objects retain both continuum topology and a body-backed rigid proxy. The runtime publishes explicit collision ownership on every transition, so the rigid fallback collider is disabled while the continuum representation is active and enabled only while the object is rigid. The GPU measures:

- total mass and centre of mass;
- linear and angular momentum;
- inertia and inverse inertia;
- average deformation rotation;
- peak and RMS strain;
- minimum determinant and deformation residual.

Hysteresis controls demotion to rigid representation after a sustained low-strain interval. Contact, strain or numerical events promote the object back to its authored MPM or FEM representation. State transfer preserves centre of mass, linear momentum, angular momentum and the measured full inertia tensor. Demotion publishes an environment-specific world inverse inertia that MetalWorld validates and rotates through ordinary and CCD integration instead of replacing it with a mass-scaled authored tensor. The compiler retains an immutable mass-weighted rest centre for every continuum object; promotion reconstructs positions from that rest frame and writes the current rigid orientation as the deformation rotation, avoiding translation leakage or inertia-derived orientation. Adaptive bindings must be valid, body-backed and unique; the compiler rejects ambiguous mappings.

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
- one borrowed command buffer contains identification, parameter overlay, all microticks, contact, rigid coupling, adaptive transfer and event reduction;
- no CPU counter read, command-buffer commit or wait occurs inside `Runtime::encode`;
- fixed-capacity deterministic gathers replace floating-point append/scatter atomics;
- SIMD32 object reductions map to Apple GPU execution width;
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

The subsystem implements the complete architecture and executable kernels listed above, but it does not claim one numerical representation solves all matter:

- the MPM path currently targets total-Lagrangian solids;
- the FEM path uses tetrahedral solids and fixed-iteration matrix-free PCG;
- contact supports continuum nodes against analytic rigid proxies; continuum-continuum/self-contact and arbitrary deforming triangle-triangle IPC are not yet implemented;
- fracture/topology mutation, Eulerian fluids, thermal coupling and porous flow require additional operators in the same Physics IR;
- learned residual constitutive terms can be represented as future expression/program kinds but are not silently enabled.

## Validation

The portable compiler and package code is built with:

```text
-std=c++23 -Wall -Wextra -Wpedantic -Werror
```

The included material compiles into a world containing both MPM and FEM
objects, writes a package, emits specialized Metal and round-trips through
`readPackage`. Source-level audits also check shared-ABI sizes, kernel names,
buffer bindings and delimiter balance.

On an Apple Silicon Metal 4 build host, run the physics qualification suite
after building the probe target:

```bash
cmake -S . -B build -G Ninja
cmake --build build --target metalrobo_matter_physics_probe -j 3
ctest --test-dir build -L matter --output-on-failure -j 1
```

The suite executes the actual `NumiMatter.metallib` through borrowed Metal
command buffers. It checks MPM freefall, a coupled eight-particle MPM plane
impact, an articulated-body continuum reaction consumed by MetalWorld ABA, a
full-height implicit FEM impact, a near-plane CFL-resolved FEM contact, and
byte-identical MPM/FEM/scheduler rollback after an enclosing rigid transaction
rejects the tentative continuum update. It is deliberately serial because all
cases use the active GPU.

This is continuum and analytic-proxy contact evidence, including one
articulated rigid-reaction handoff through a full `MetalWorld` step. It is not
a blanket claim that every integration path is qualified: adaptive ownership
transfer, inverse identification, and longer-running material calibration
remain separate qualifications.
