# Numi Matter

Numi Matter is a compiled multiphysics subsystem for Numi Lab. It keeps material semantics separate from numerical representation, then cooks a fixed-capacity Apple-GPU execution graph for the exact world being simulated.

The module is deliberately self-contained under `matter/`. It can be built and evaluated without changing the existing Numi Lab production target, then linked into `metalrobo` once the owning `MetalWorld` call site is ready to schedule the borrowed-command-buffer encoder.

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

The MPM path is a total-Lagrangian solid backend using fixed quadratic B-spline support over 27 nodes per material point. The cooker builds sparse grid nodes and deterministic node-owned incidence lists. Runtime kernels perform:

1. particle-to-grid mass, APIC momentum and constitutive-force gathering;
2. grid acceleration and integration;
3. unified continuum contact projection;
4. grid-to-particle velocity, affine field and deformation-gradient updates;
5. transactional candidate publication.

Node-owned gathers avoid floating-point scatter atomics and produce stable accumulation order across runs. This backend is intended for elastic, viscoelastic and elastoplastic solids with large deformation. It is not presented as a general free-surface fluid implementation.

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

Rigid reactions are reduced per proxy without atomics. A final bridge executes once per environment:

- articulated targets receive `MRABABodyWrenchGPU` force and torque before inverse ABA;
- free scene bodies receive direct velocity and angular-velocity impulses through `MRBodyStateGPU`;
- static unbound proxies participate in contact but receive no state update.

The runtime consumes a borrowed body-state arena and borrowed rigid output buffers. It never commits or waits on the borrowed command buffer.

### Adaptive representation

Adaptive objects retain both continuum topology and a body-backed rigid proxy. The GPU measures:

- total mass and centre of mass;
- linear and angular momentum;
- inertia and inverse inertia;
- average deformation rotation;
- peak and RMS strain;
- minimum determinant and deformation residual.

Hysteresis controls demotion to rigid representation after a sustained low-strain interval. Contact, strain or numerical events promote the object back to its authored MPM or FEM representation. State transfer preserves centre of mass, linear momentum and angular momentum. Promotion reconstructs continuum positions and velocities from the current rigid pose and twist. Adaptive bindings must be valid, body-backed and unique; the compiler rejects ambiguous mappings.

### Inverse material identification

Identifiable material parameters receive a GPU-resident distribution. Candidate environments use deterministic antithetic normal perturbations, optional log-space sampling and bounded parameter overlays. After candidate losses are written into the exposed loss buffer, the update kernel performs temperature-weighted distribution fitting and republishes the learned mean and variance.

The compiler requires an even candidate count no larger than the environment count, so every candidate is actually simulated and every antithetic partner exists.

### Event-driven multirate scheduler

Every continuum object has a power-of-two rate exponent. The fixed command graph contains the maximum number of microticks; each object executes only on its scheduled ticks. GPU event reduction observes contact onset/release, slip, strain/yield, damage, determinant risk, solver residual and requested rate changes. Events raise local substep frequency immediately and lower it only after a quiet-frame hysteresis period.

Events are written to deterministic fixed slots rather than an unordered append queue. They can be consumed by Numi Lab's event-token learning layer without CPU synchronization.

## Apple Silicon execution model

The runtime is designed around Apple GPU constraints rather than CUDA assumptions:

- immutable cooked tables and persistent state use `MTLStorageModePrivate`;
- unified/shared memory is used only for initialization, explicit status/event publication and identification losses;
- one borrowed command buffer contains identification, parameter overlay, all microticks, contact, rigid coupling, adaptive transfer and event reduction;
- no CPU counter read, command-buffer commit or wait occurs inside `Runtime::encode`;
- fixed-capacity deterministic gathers replace floating-point append/scatter atomics;
- SIMD32 object reductions map to Apple GPU execution width;
- environment and object parallelism remain independent;
- all runtime identities are content-fingerprinted.

## Build

From the Numi Lab repository root:

```bash
cmake -S matter -B build-matter -G Ninja
cmake --build build-matter -j
```

The module requires Apple Silicon macOS, CMake 3.28+, C++23 and Metal 4.0. It builds:

```text
libnumi_matter_compiler.a
libnumi_matter_runtime.a
NumiMatter.metallib
numi-matterc
```

Compile the included material and a cooked MPM/FEM demonstration world:

```bash
build-matter/bin/numi-matterc \
  --material matter/materials/silicone.nmatter \
  --output silicone.nmatterpack \
  --generated-metal silicone.generated.metal \
  --mode both \
  --envs 16
```

## Numi Lab integration boundary

The owning `MetalWorld` graph should call `Runtime::encode` after its external-wrench arena is cleared and before articulated ABA consumes those wrenches. Supply:

```cpp
numi::matter::EncodeRequest request{
    .commandBuffer = borrowedCommandBuffer,
    .rigid = {
        .currentBodies = currentWorldBodyBuffer,
        .articulatedWrenches = externalWrenchBuffer,
        .sceneBodies = candidateSceneBodyBuffer,
        .currentBodyCount = bodyCount,
        .currentBodyStride = bodyStride,
        .articulatedBodyCount = articulatedBodyCount,
        .sceneBodyCount = sceneBodyCount,
        .articulatedStride = articulatedWrenchStride,
        .sceneStride = sceneBodyStride,
    },
    .controlStep = controlStep,
    .seed = seed,
};

const auto diagnostics = matterRuntime.encode(request);
```

A body-backed rigid proxy requires `currentBodies`. Unbound proxies are interpreted directly in world coordinates and are static. Adaptive representation additionally requires a unique body-backed proxy binding.

## Current fidelity boundary

The subsystem implements the complete architecture and executable kernels listed above, but it does not claim one numerical representation solves all matter:

- the MPM path currently targets total-Lagrangian solids;
- the FEM path uses tetrahedral solids and fixed-iteration matrix-free PCG;
- contact supports analytic rigid proxies, not arbitrary deforming triangle-triangle IPC;
- fracture/topology mutation, Eulerian fluids, thermal coupling and porous flow require additional operators in the same Physics IR;
- learned residual constitutive terms can be represented as future expression/program kinds but are not silently enabled.

## Validation performed in this delivery

The portable compiler and package code was built with:

```text
-std=c++23 -Wall -Wextra -Wpedantic -Werror
```

The included material compiled into a world containing both MPM and FEM objects, wrote a package, emitted specialized Metal and successfully round-tripped through `readPackage`. Source-level audits check shared-ABI sizes, kernel names, buffer bindings and delimiter balance.

The current execution environment is Linux and does not provide Apple Metal or Objective-C++ frameworks. Therefore this delivery does not claim that the `.metal` and `.mm` targets were compiled or executed here. The repository code is structured for the existing Apple build machine to perform that final platform compilation directly.
