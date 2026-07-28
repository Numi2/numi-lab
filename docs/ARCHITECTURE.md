# MetalRobo architecture

## Product boundary

MetalRobo owns the physics implementation, compiled model, runtime, GPU
memory, task execution, and public APIs. The current target is Apple silicon
macOS 26 / Metal 4. A Python package drives learning through MLX; the engine is
C++23, Objective-C++, and Metal Shading Language. No external physics engine
is linked or called at runtime.

The headless training path is the product core. Rendering is a future consumer
of body-pose buffers, not part of the current physics step.

## Two model generations

Version 0.4 contains a working compatibility runtime and a canonical engine
spine. They coexist; they are not yet one fully integrated execution path.

The compatibility `metalrobo::Model` owns the original fixed-base Franka
layout:

- `MRModelGPU`, `MRJointGPU`, `MRLinkGPU`, and `MRColliderGPU`;
- fixed capacities of 32 DoF, 33 links, and 64 colliders;
- one batched Metal ABA/reach-task runtime used by the C API and MLX PPO path.

The canonical `metalrobo::EngineModel` is the forward architecture:

- versioned, pointer-free `MRWorldGPU`, articulation, joint, per-DoF, body,
  shape, and material records;
- explicit counts, offsets, capacities, and status codes;
- separate generalized configuration and velocity dimensions (`nq != nv`);
- fixed or floating roots and packed articulation-local coordinate ranges;
- strict global `q`/`v` ownership plus limits, drive parameters, dry friction,
  and physical armature for every generalized velocity coordinate;
- the compiled COM-consistent, 29-DoF Unitree G1 model;
- a fixed-root dVRK-style PSM research model with true prismatic insertion.

Canonical rigid-body translation and linear velocity are measured at the
center of mass. Orientation remains the body/link-frame orientation. Joint
anchors are therefore stored relative to each body's COM. A floating root uses
world COM `xyz` plus quaternion `xyzw` in `q`, and world COM linear velocity
plus world angular velocity in `v`.

There is no URDF, MJCF, or OpenUSD importer yet. Franka, G1, and the surgical
PSM are compiled in-house model definitions with pinned provenance.

## Execution planes

### Batched Metal Franka runtime

The original production slice runs one fixed-base Franka environment per
threadgroup:

1. Decode normalized actions into bounded joint targets and efforts.
2. Build joint transforms and spatial velocities.
3. Assemble the existing primitive ground-contact forces.
4. Run FP32 articulated-body forward dynamics.
5. Apply limits and integrate four physics substeps per control step.
6. Compute body poses, observations, reward, termination, and device reset.

Parallelism is primarily across environments. This is the path behind the
published local throughput and PPO smoke results. It does not execute the
canonical G1 model or the generic contact solver.

### Generalized articulated CPU reference

`ArticulatedDynamics` is an FP64 implementation for fixed or floating trees
with revolute, continuous, prismatic, and fixed joints. It uses:

- a world-coordinate composite-rigid-body recursion to assemble the dense
  generalized mass matrix;
- Cholesky for forward dynamics;
- recursive Newton-Euler kinematics for bias and inverse dynamics;
- gravity, body damping, and external world-frame wrenches about body COMs;
- per-DoF armature in CRBA, RNEA, kinetic energy, contact, and impulse
  response;
- symplectic Euler or converged implicit midpoint integration with an SO(3)
  exponential update for the floating quaternion.

The reference executes the actual 30-body/29-joint G1 topology and passes
forward/inverse and conservation probes. Its analytic kinematics layer exposes
COM poses/twists and batch point Jacobians without configuration
perturbations. `ArticulatedContact` retains a checked CRBA Cholesky factor and
implements `J`, `Jᵀ`, and `J M⁻¹ Jᵀ` actions; the actual 35-velocity G1 passes
a two-foot exact-cone quality solve. A transactional adapter now converts the
common collision/contact ABI into articulated contacts, including endpoint
and cached-impulse basis swaps, static/kinematic velocity compensation, and
material semantics from evaluated ConstraintIR. The production quality solve
consumes physical Delassus and never materializes a generalized inverse; only
the independent projected-gradient oracle requests the checked dense
compatibility adapter.

`stepArticulatedWorldCpu` composes free dynamics, projected collision state,
real manifolds, ConstraintIR compilation/evaluation, exact-cone solving,
factor-backed impulse application, the common residual, and configuration
integration for exactly one articulation against static or kinematic
environment bodies. Actual G1 ground contact produces and solves all eight
foot-sphere contacts. Every output and persistent cache is published only
after the full step succeeds. Explicit model/custom PD and effort commands are
executable through the separate transactional actuation evaluator and can
feed this world's generalized-force input. Active position stops are compiled
into the same contact-space solve, so contact/limit cross terms are retained.
Coupled implicit drives, set-valued stiction, multiple articulations, and self
collision remain open.

### Surgical research slice

`makeDvrkPsmLargeNeedleDriverEngineModel` compiles a nine-body,
eight-coordinate serial remote-center equivalent of the dVRK Classic PSM. Its
insertion axis is a real prismatic joint in the CPU reference, Metal mass
operator, Metal inverse-mass action, and Metal ABA. The two jaws remain
separate physical coordinates and colliders. A validated seven-target policy
map expands one total angular aperture into symmetric jaw commands without
pretending to implement the missing tendon/transmission dynamics or replacing
contact by an attachment. The authored distal jaw surfaces are tangent at zero
logical aperture and separate monotonically across the allowed range.

`SurgicalAssets` procedurally builds a GS-21-scale half-circle needle, a
training ring, and a peg board. Compound capsules carry stable segment IDs and
the needle records swage, grasp, taper, and tip arc zones. Mass, COM, and
inertia are derived from explicit geometry and named research density/profile
defaults. These are rigid research/training assets: there is no suture strand,
tissue, puncture, cutting, biomechanical, or clinical model.

`stepArticulatedRigidWorldCpu` is the transactional mixed
articulation/maximal-coordinate world. It accepts exactly one articulation
and an external scene containing dynamic, static, and kinematic bodies, with
at least one dynamic body. In one step it:

1. predicts articulated and dynamic-body free velocities without advancing
   pose while preserving static/kinematic point velocities;
2. generates one deterministic collision stream for
   articulation-dynamic, articulation-prescribed, dynamic-dynamic, and
   dynamic-prescribed pairs;
3. compiles active articulated position stops;
4. solves contact and stops simultaneously through one block inverse-mass
   operator and exact circular Coulomb cones, compacting only dynamic scene
   bodies into solver coordinates;
5. integrates every configuration exactly once; and
6. atomically publishes state, manifolds, world-space contact warm starts,
   scalar limit warm starts, and dwell-filtered grasp evidence.

The grasp classifier is evidence only. It requires compressive impulses on
both configured jaws, opposing normals, sufficient friction, bounded
post-solve tangential slip, and consecutive qualifying steps. It never creates
a weld, attachment, or hidden force. Grasp dwell is keyed by the model, jaw
configuration, thresholds, rigid slot, and participating shape generations,
so it cannot carry across a replaced object or changed grasp definition.
The physical default preserves assembled witnesses. A caller may explicitly
request deterministic deepest-point conditioning per canonical endpoint-body
pair; the needle probes use one witness per pair so adjacent compound needle
segments do not dominate the small dense solve. Contact warm-start identity
includes endpoint kind, source body/shape/feature, slot generation, motion
type, and articulation index, so replacement or dynamic/prescribed role
changes cannot inherit stale impulses.

The supported pickup probe settles the procedural needle on six independently
owned static support bodies, approaches with open jaws, closes on an authored
COM-near grasp-zone segment, transfers load, and clears the fixture during an
8 mm lift. Grasp classification remains evidence only and rollback includes
all mixed-body state and cache streams.

The current mixed world is CPU FP64, contains one articulation, and uses
three-row point Coulomb contacts. Multiple articulations, finite jaw-patch
and rolling/torsional resistance, CCD/conservative substepping, and a
device-resident Metal composition remain open. The supported pickup therefore
uses a slow conservative-discrete approach/lift and makes no high-speed
time-of-impact claim.

### Generic maximal-coordinate rigid-body world

`stepRigidBodyWorldCpu` is the currently composed generic contact world. Every
dynamic object has an independent six-velocity maximal-coordinate
`MRBodyStateGPU`; articulated generalized coordinates do not enter this
pipeline.

The transactional CPU step performs:

1. Free-body unconstrained velocity prediction.
2. CPU sweep-and-prune collision and persistent-manifold refresh.
3. Material mixing and contact-constraint assembly.
4. Warm start and either the throughput PGS block or FP64 quality solve.
5. COM position and SO(3) orientation integration.

On reported failure, body state and persistent caches are unchanged. This
pipeline validates contact composition for free rigid bodies only. Calling it
an articulated G1 simulator would be incorrect.

Its public free-motion implementation now exposes separate velocity
prediction and configuration-only integration phases. The mixed surgical
world reuses those phases so gravity, damping, gyroscopic response, and pose
advancement are each applied exactly once around the coupled impulse solve.
Constrained split worlds currently require symplectic Euler. Implicit
midpoint is rejected rather than silently reconstructing its midpoint pose
increment from an endpoint velocity.

### Generic Metal components

The canonical ABI is also consumed by focused Metal kernels:

- symplectic and implicit-midpoint free-body integration;
- a parallel deterministic micro broadphase using flag, two-level exclusive
  scan, and canonical scatter without global append atomics;
- a deterministic one-thread `O(n²)` collision correctness kernel for
  sphere/sphere, sphere/plane, capsule/plane, box/plane, cylinder/plane,
  sphere/capsule, capsule/capsule, and sphere/box;
- a correctness-first generic fixed/floating articulation operator for body
  poses, point Jacobians, mass, `Jᵀp`, and factor-solved `M⁻¹Jᵀp`, exercised
  on actual 35-velocity G1 with authoritative armature;
- the fixed-budget contact PGS block.

These kernels have CPU/Metal parity probes, but they are not yet assembled
into a batched generic GPU world. GPU manifold persistence/reduction,
segmented LBVH, parallel narrowphase, and a parallel articulated tree
implementation are open work. The first generic articulation kernel executes
its dense correctness path in lane zero and must not be represented as a
throughput result. The current micro broadphase has an explicit
65,536-logical-pair scan bound.

The forward architecture is operator-first: every constraint type compiles to
one semantic program, every dynamics backend exposes a free-motion solve
action, collision stages are deterministic streams, and numerical methods are
selected per island while sharing one residual oracle. The concrete decisions
and their implementation boundary are in
[V04_TRANSACTIONAL_ARCHITECTURE](V04_TRANSACTIONAL_ARCHITECTURE.md).

The first `ConstraintIR` CPU reference is executable. Its ABI-v2 records are
fixed-layout, 16-byte aligned, pointer-free, stable-key sorted, and
range-validated. An adapter converts ABI-v1 contact blocks transactionally.
One evaluator owns stabilization, restitution, compliance/dissipation
regularization, and static/dynamic friction selection; quality and throughput
views reference the same evaluated buffers and semantic fingerprint. A common
residual implements scalar KKT and exact capped elliptic
normal/tangent/torsion projection. The composed articulated quality path now
consumes the fingerprinted evaluated contact subset and ends in this residual.
Other production solvers and non-contact constraint compilers are not yet
driven from the IR; unimplemented rolling/adhesion semantics fail explicitly.

## Collision and contact boundary

The CPU collision reference implements deterministic sweep-and-prune,
sphere/sphere, sphere/plane, capsule/plane, box/plane, cylinder/plane,
sphere/capsule, capsule/capsule, and sphere/box witnesses, stable features,
and persistent four-point manifold reduction. The Metal collision kernel
implements the same eight pair classes and matches CPU witness geometry in
its focused probe. Other cylinder pairs, general capsule/box pairs, convex
GJK/EPA/MPR, triangle mesh, heightfield, SDF, and deformable geometry are not
executable production paths.

There is no continuous collision detection. Neither conservative advancement,
time-of-impact island stepping, nor speculative CCD is implemented.

The throughput PGS kernel has a hard capacity of 128 contacts per dispatch.
The composed CPU world partitions independent connected constraint islands
before dispatch, making 128 contacts the current limit for any one connected
island. A larger connected island returns explicit capacity overflow; contacts
are not silently dropped. Metal dispatch construction must provide the same
island partition. Size buckets and spill/replay remain future work.

The solver portfolio also includes an independent FP64 projected-gradient
exact-cone oracle and a globalized FP64 semismooth-Newton quality solver. The
throughput path remains PGS, not TGS.

## API, memory, and synchronization

- `metalrobo::Runtime` and `c_api.h` expose the original Franka runtime.
- Canonical engine, collision, solver, world, G1, and articulated-reference
  APIs are currently C++ interfaces; they are not all surfaced through the C
  ABI.
- The Python package exposes stable read-only NumPy views and an MLX PPO
  learner.

All current Metal runtime buffers use Apple-silicon shared storage. This
avoids PCIe copies but not synchronization: the C API `step` completes its
command buffer before returning shared views. The PPO path then materializes
MLX arrays, so v0.4 is not a fused physics/learner command stream.

The generic articulated operator's public host API owns its compact buffer
table and preflights checked element/byte arithmetic, the shader's 32-bit
address ceiling, actual `MTLBuffer.length`, per-buffer device limits, and the
aggregate recommended working set before encoding. Its result and typed GPU
status stream publish atomically. This correctness wrapper is synchronous and
recreates resources per call; it is not the persistent rollout scheduler.

Allocation is preflighted against
`MTLDevice.recommendedMaxWorkingSetSize` and `MTLDevice.maxBufferLength`.
Training must leave unified-memory headroom for MLX parameters, rollout
storage, macOS, and a future viewer.
