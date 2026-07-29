# Validation record

Snapshot: 2026-07-28, v0.4 development milestone plus MetalWorld ABI v1.

## Evidence status

The v0.4 numbers below are from a fresh out-of-tree Release build on an Apple
M4. C++, Objective-C++, and all Metal shaders compiled cleanly; C++ and
Objective-C++ warnings were errors. All 21 probe executables passed from the
same source state. A second full out-of-tree build with AddressSanitizer and
UndefinedBehaviorSanitizer also compiled and passed all 21 probes.

The later device-resident contact/MLX tranche built 38 native probes from one
source state; all 38 passed, followed by the MLX extension probe. Its new
result is recorded separately below; historical v0.4 component numbers are
retained rather than being rewritten. The earlier sanitizer statement applies
to the v0.4 source state only and is not silently extended to this tranche.
A fresh out-of-tree Release build with C++ and Objective-C++ warnings promoted
to errors then passed all 38 native probes.

A local pass establishes only the stated executable contract. It does not
establish external-simulator agreement, long-horizon robot stability, contact
differentiability, or performance superiority.

## Machine and toolchain

- MacBook Air `Mac16,12`
- Apple M4: 10 CPU cores, 10 GPU cores, 24 GB unified memory
- macOS 26.6, Metal 4
- Apple clang 21.0.0
- MLX 0.32.0 on `Device(gpu, 0)`
- NumPy 2.2.5
- CMake 4.0.1 Release configuration

## Persistent MetalWorld ABI v1

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target metalrobo_metal_world_probe -j 8
./build/bin/metalrobo_metal_world_probe
```

The probe compiles the canonical Franka model, advances seven control steps
with three physics substeps and per-environment resets, and compares every
captured q/v/acceleration against the FP64 generalized dynamics oracle. It
then verifies same-device bitwise replay, asynchronous input snapshotting,
the one-in-flight arena gate, transactional host rejection, grow-only
capacity reuse, and an injected ABA factorization failure that must restore
the entire control-step checkpoint.

The performance case uses 4,096 Franka environments and a 16-control-step
horizon with four physics substeps. Five measured samples follow one warmup.
Device time is taken from Metal's command-buffer GPU start/end timestamps;
wall time also includes host validation, input copies and encoding, final
wait, result allocation, and output copies. The hard gate applies to both.

```text
metal_world=metal
device="Apple M4"
abi=1 graph=free_motion_aba
environments=4 control_steps=7 physics_substeps=3
q_error=5.7526e-07
v_error=4.74513e-08
acceleration_scaled_error=1.98164e-06
throughput_batch=4096 throughput_horizon=16
gpu_control_steps_per_s=242100
wall_control_steps_per_s=239771
gpu_p50_ms=270.531 gpu_p95_ms=271.123
wall_p50_ms=273.413 wall_p95_ms=273.683 thermal=nominal
replay=bitwise async=pass input_snapshot=pass busy_gate=pass
reset=pass rollback=pass g1_free_motion=pass
capacity_bucket_equivalence=bitwise grow_only=pass host_transaction=pass
no_host_sync_between_control_steps=yes contact_graph=deferred status=ok
```

The floating-base G1 canary (two environments, four control steps, two
substeps) had q/v/scaled-acceleration errors of `1.3234e-7`, `8.04595e-8`,
and `1.28661e-5`. This passes the declared 150,000 control-steps/s Franka
free-space gate on this machine. It does not exercise a free object, contacts,
rewards, or MLX zero-copy interoperability; those stages are explicitly
absent from ABI v1.

## Final clean Release gate

The completed clean gate started outside every existing build directory:

```sh
metalrobo_check_dir=$(mktemp -d /tmp/metalrobo-v04-clean.XXXXXX)
cmake -S . -B "$metalrobo_check_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-Werror \
  -DCMAKE_OBJCXX_FLAGS=-Werror
cmake --build "$metalrobo_check_dir" -j 8

for probe in "$metalrobo_check_dir"/bin/metalrobo_*_probe; do
  "$probe"
done
```

The native/Python version boundary also returned
`python=0.4.0 native=0.4.0`.

### Install/consumer gate

`cmake --install` was exercised into a fresh temporary prefix. An external
C++23 consumer compiled against only the installed headers and dylib, then ran
the checked G1 Metal host probe successfully. This verifies that the public
operator discovers the co-installed `lib/metalrobo/MetalRobo.metallib`
instead of depending on its build-tree path. The installed
`metalrobo_bench` also launched through its relative install RPATH.

### Sanitizer gate

The second clean build used
`-fsanitize=address,undefined -fno-omit-frame-pointer -Werror` for C++ and
Objective-C++, plus sanitizer linker flags. Every probe passed with
`halt_on_error=1`. LeakSanitizer is not supported by this Apple sanitizer
runtime, so this result covers AddressSanitizer memory safety and undefined
behavior, not leak detection.

## Generalized articulated dynamics and armature

```sh
./build/bin/metalrobo_articulated_dynamics_probe
```

The generalized CPU reference uses FP64 intermediates, world-coordinate CRBA,
checked Cholesky factorization, and recursive Newton-Euler inverse/bias
dynamics. It supports fixed or floating trees made from revolute, continuous,
and fixed joints. Floating state is deliberately `nq != nv`: root COM
position plus quaternion in `q`, and world linear plus angular COM velocity
in `v`.

ABI v2 supplies one canonical `MRDofPropertiesGPU` record per generalized
velocity. Armature is treated as physical generalized inertia:

- CRBA adds it to the mass diagonal;
- RNEA adds `armature * qdd`;
- kinetic-energy evaluation includes it;
- forward dynamics, contact response, and impulse response factor the same
  operator;
- the generic Metal operator consumes the same per-DoF stream.

Clean Release result:

```text
articulated=cpu_fp64_crba_rnea
free_body_error=4.440892e-16
pendulum_error=5.329071e-15
forward_inverse_error=8.881784e-16
mass_symmetry_error=0
min_mass_pivot=3.025828e-01
g1_com_anchors=yes g1_nq=36 g1_nv=35
g1_forward_inverse_error=3.526346e-14
armature_mass_error=1.110223e-16
armature_inverse_error=1.764403e-15
armature_energy_error=1.776357e-14
armature_samples=12
energy_drift=5.569560e-11
linear_momentum_drift=8.940619e-10
angular_momentum_drift=2.019079e-09
q_norm_error=0
midpoint_iterations=2 midpoint_residual=5.532438e-17
limit_rejection=transactional nonfinite_rejection=transactional
finite=yes
```

This establishes analytical checks, forward/inverse consistency, conservation
checks, and armature propagation through the actual G1 topology. It does not
establish agreement with an external engine. Authored joint limits are
validated model data; they are not yet compiled into limit impulses.

## Transactional articulated actuation

```sh
./build/bin/metalrobo_articulated_actuation_probe
```

`evaluateArticulatedActuation` supports four explicit per-DoF modes:
`disabled`, `modelPD`, `customPD`, and `effort`. It evaluates feed-forward plus
PD, uses the shortest signed angle for continuous joints, clamps actuator
effort before passive dry friction, forbids floating-root actuation, rejects
semantically active fields in inactive modes, and publishes only after the
entire command stream is valid.

Clean Release result:

```text
articulated_actuation=cpu_fp64
model_pd=12.25 custom_pd=1.65 effort_clamped=88
saturation_excess=264 moving_friction=-3
stiction_generalized=0 continuous_seam=0.2
offset_q=7 offset_v=6
replay=bitwise transaction=pass status=ok
```

The near-zero dry-friction behavior is intentionally a local controller
approximation. It cancels only the local actuator load up to the authored
friction magnitude. Gravity, bias, external wrench, and contact loads are not
available to this evaluator, so this is not coupled set-valued stiction. The
evaluator can supply the composed world's generalized-force input; implicit
drives and coupled stiction are not embedded in the world step.

## Analytic articulated contact and exact frames

```sh
./build/bin/metalrobo_articulated_contact_probe
```

The FP64 contact operator evaluates body COM poses/twists and analytic point
Jacobians, retains the checked mass Cholesky factor, and applies impulses as
`J' -> solve(M) -> J`. The production contact-space path never forms or
multiplies a dense generalized inverse. A dense inverse remains an explicit,
checked compatibility artifact for the independent reference oracle only.

Each articulated contact carries a complete right-handed
`normal/tangentU/tangentV` frame. The validator checks finite, near-unit, and
near-orthogonal input, but deliberately does not normalize or Gram-Schmidt it.
The accepted frame is consumed verbatim by Jacobian construction, evaluated
targets and impulses, semantic fingerprinting, and residual evaluation. This
prevents hidden coordinate changes between compiler, solver, cache, and
acceptance gate.

Clean Release result:

```text
articulated_contact=analytic_fp64
free_jv_error=2.775558e-17
free_delassus_error=1.110223e-16
free_impulse_error=5.551115e-17
pendulum_jacobian_error=1.778273e-08
pendulum_delassus_error=7.793282e-09
armature_delassus_error=8.910624e-09
armature_impulse_ratio_error=6.512499e-09
g1_pose_error=1.734723e-18
g1_jv_error=5.551115e-17
g1_base_column_error=0
g1_tree_sparsity_error=0
g1_delassus_symmetry=0
g1_dense_adapter_residual=1.674285e-14
g1_reciprocity=0
g1_jacobian_adjoint=5.637851e-18
g1_solver_velocity_error=8.881784e-15
g1_quality_kkt=7.664994e-16
frame_semantics=verbatim transactionality=yes status=ok
```

The pendulum tolerance is limited by FP32 canonical model constants; the
operator and oracle computations are FP64.

## Evaluated ConstraintIR semantics

```sh
./build/bin/metalrobo_constraint_ir_probe
```

ConstraintIR ABI v2 validates fixed-layout streams, stable-key ordering,
endpoint and range ownership, exact contact frames, cones, and warm starts
before publication. Its evaluator is the only timestep-dependent semantic
authority. It selects:

- stabilization and restitution targets;
- static versus dynamic friction;
- compliance, dissipation, and discrete regularization;
- feasible warm-start projection;
- cone and row activation;
- a fingerprint over the evaluated program.

Quality and throughput views reference the same evaluated row/cone buffers and
fingerprint. The articulated adapter consumes that evaluated view rather than
re-deriving material or target rules. The common residual is evaluated from
the same program after impulse application.

Clean Release result:

```text
constraint_ir=abi_v2
blocks=2 rows=7 endpoints=4 cones=2
fingerprint=603214807485541855
shared_buffers=yes
restitution_target=1.149999976
regularization=0.2299999893
equilibrium_residual=0
adversarial_cone_violation=0.1499999985
coupled_torsion_residual=2.309401127
projected_warm_normal=1.200000048
projected_warm_tangent=0.6000000238
adversarial_checks=16 transactional=yes status=ok
```

The nonzero adversarial cone and torsion values are intentional analytical
violation regressions, not accepted-equilibrium residuals. The current
three-row articulated exact-cone adapter rejects rolling, torsion, adhesion,
anisotropy, hard impulse caps, and zero-regularization blocks rather than
silently approximating them. The ABI can describe more than this adapter can
currently execute.

## Exact-cone contact-space quality solve and physical PSD gate

```sh
./build/bin/metalrobo_quality_contact_probe
```

The quality solver accepts the physical contact-space Delassus operator
`W = J M^-1 J'`, free contact velocity, evaluated targets, warm impulses, and
strictly positive per-row regularization. It solves exact circular Coulomb
cones with a safeguarded FP64 semismooth Newton method. A four-entry
Grippo-Lampariello-Lucidi merit window avoids microscopic monotone-Armijo
steps while cone active sets change. Direct Newton gets twelve line-search
attempts; if it is not accepted, the regularized Gauss-Newton direction gets
the full search budget before the projected-gradient safety step.

The physical `W` is checked independently for symmetry and positive
semidefiniteness before material regularization is added. This prevents a
negative physical mode from being hidden by an SPD regularized Hessian.
Exactly zero and nonzero rank-deficient PSD operators are accepted. A
scale-aware eigenvalue tolerance accepts a mode inside the contract and
rejects one outside it, including a dense tiny-indefinite Schur tail.

Residual and certificate norms use scaled sum-of-squares, and natural-map
normalization combines magnitudes through binary exponents. A frictionless
scalar regression with finite `1e200` velocity and regularization therefore
returns the analytic unit impulse instead of overflowing a denominator and
falsely accepting zero impulse.

## Heterogeneous multi-articulation Metal contact graph

```sh
./build/bin/metalrobo_metal_multi_articulated_contact_probe
```

This device probe composes two floating articulations and one independent
dynamic body, then solves two coupled contacts as one 18-DoF exact-cone
island. Point Jacobian emission, global row assembly, four inverse-ABA
packets, free-body inverse response, Delassus construction, the quality solve
and candidate publication execute in one Metal command buffer.

On the Apple M4 the normal Delassus block is `[2, -1; -1, 2]`, both normal
impulses are `0.99999`, the largest post-solve speed is `1.02e-5`, and the
maximum error against the FP64 heterogeneous oracle is `9.2e-6`. A second
submission is bitwise identical. An invalid quaternion injected into one
environment rolls back only that environment while the other three publish
normally; a rejected pre-dispatch input leaves the previously accepted host
result unchanged.

The canonical heterogeneous surgical probe also compiles the two PSMs and
dynamic needle once, then reuses that immutable program for a 34-DoF Metal
contact solve:

```text
contact_nv=34
needle_contact_impulses=0.0002592/0.0002592
equality_residual=1.38778e-17
metal_equality_residual=7.45058e-09
metal_contact_error=6.89394e-08
metal_kkt=2.9199e-07
cold_start_rejected=yes
```

The persistent-manifold warm start agrees with the FP64 generalized velocity
and impulse payload. With warm starts explicitly disabled, this deliberately
ill-conditioned free-base contact does not satisfy the FP32 KKT certificate;
the quality stage reports `DID_NOT_CONVERGE` and the environment velocity is
rolled back. Before the KKT certificate was added, the step-size-scaled
natural residual could incorrectly accept a zero impulse on this case.

Twelve PSM floating-base lock rows and two jaw gear rows are Schur-eliminated
into the two jaw/needle contact cones on both FP64 and Metal. The Metal graph
appends these rows to the same parallel inverse-ABA RHS packets, performs the
small deterministic Cholesky/null-space projection before its cone solve, and
reconstructs equality impulses afterward. The result closes every regularized
row below `7.5e-9` and matches the constrained FP64 payload within `6.9e-8`.

Clean Release result:

```text
solver=quality_fp64_semismooth_newton
contacts=4 iterations=6 reference_iterations=386
newton_steps=6 gauss_newton_fallbacks=0
projected_gradient_fallbacks=0
kkt=1.513167e-12
primal_cone_violation=9.464532e-18
dual_cone_violation=3.768652e-12
contact_complementarity=1.665117e-12
frictionless_normal_complementarity=0
reference_impulse_error=4.741880e-10
contact_space_impulse_error=7.560244e-15
contact_space_velocity_error=2.396976e-15
delassus_psd_gate=yes
rank_deficient_psd=yes
nonzero_rank_deficient_psd=yes
psd_tolerance_contract=yes
permutation_impulse_error=5.958796e-16
warm_iterations=0 explicit_failure=yes extreme_scale=yes finite=yes
```

The quality path materializes the small contact Hessian used by direct Newton.
It does not materialize `M^-1`. Large quality islands still require a
matrix-free Newton-PCG implementation.

## Collision-generated G1 contact adapter

```sh
./build/bin/metalrobo_g1_collision_contact_probe
```

This boundary probe starts with actual G1 body state, executes deterministic
collision and persistent-manifold assembly against a z-up plane, compiles and
evaluates ConstraintIR, adapts its fingerprinted rows, and solves through the
factor-backed generalized operator:

```text
g1_collision_contact=cpu_to_generalized_fp64
root_lowering=1.629781e-02
raw_contacts=8 manifolds=8 constraints=8 adapted=8
penetration_max=5.000198e-04
target_normal_max=1.920095e-02
solved_normal_min=1.919900e-02
operator_velocity_error=1.873501e-15
reconstructed_point_error=1.862645e-09
free_jv_parity_error=0
quality_kkt=8.428597e-16
endpoint_swap_error=0
target_rule_error=2.775558e-17
compliance_regularization_error=0
kinematic_compensation_error=2.682209e-09
capacity_transactional=yes tiny_timestep_rejected=yes
cross_articulation_rejected=yes
unbound_dynamic_rejected=yes
strict_constraint_flags=yes status=ok
```

This proves the geometry-to-generalized boundary with real collision contacts,
not hand-authored sole rows. The complete configuration transaction is covered
by the next probe.

## Composed transactional ArticulatedWorld

```sh
./build/bin/metalrobo_articulated_world_probe
```

`stepArticulatedWorldCpu` now composes one complete symplectic generalized
timestep:

```text
validate
  -> FP64 free dynamics
  -> articulated body-state projection
  -> collision and private persistent-manifold update
  -> canonical ConstraintIR compilation
  -> one evaluated semantic program
  -> fingerprinted articulated-contact adaptation
  -> J, checked mass factor, and physical W
  -> exact circular-Coulomb solve
  -> factor-backed impulse application
  -> independent solver-velocity agreement check
  -> common ConstraintIR residual
  -> configuration integration
  -> atomic state/cache publication
```

Any failure leaves `q`, `v`, manifold cache, impulse cache, and cache step
unchanged. The quality path does not materialize a dense generalized inverse.
The projected-gradient reference path may build its explicitly checked dense
compatibility adapter for differential validation.

Clean Release result:

```text
articulated_world=cpu_transactional model=g1
free_q_error=0 free_v_error=0
contacts=8 max_normal_impulse=1.063735408
velocity_correction=1.893145902
factor_contact_velocity_error=2.184953887e-16
semantic_fingerprint=3121490046967162423
common_residual=8.184883882e-10
quality_reference_q_error=5.551115123e-17
quality_reference_v_error=5.906462385e-15
kinematic_compensation=0.200000003
deterministic=yes late_rollback=yes
overflow_rollback=yes status=ok
```

The probe includes free-flight parity, real G1 foot contact, quality/reference
differential comparison, moving-kinematic-ground compensation, deterministic
replay, late integration rollback, and pair-capacity rollback.

This world currently admits exactly one executable articulation plus static or
kinematic environment bodies. It is not a multi-articulation island solver
and does not admit dynamic free environment objects. It does include active
position-stop impulses in the same solve as contact, but still has no
self-collision, implicit drives, or fully composed Metal timestep.

## Transactional mixed PSM scene and supported needle pickup

```sh
./build/bin/metalrobo_articulated_rigid_collision_probe
./build/bin/metalrobo_articulated_rigid_world_probe
./build/bin/metalrobo_supported_needle_pickup_probe
```

This focused CPU FP64 path uses the actual nine-body PSM and procedural
GS-21-scale needle. The generic mixed endpoint operator compacts only dynamic
scene bodies into six-velocity blocks; static and kinematic point velocities
remain prescribed. One collision stream emits
articulation-dynamic, articulation-prescribed, dynamic-dynamic, and
dynamic-prescribed contacts. Contact plus active PSM stops enter one block
inverse-mass exact-cone solve, and state/manifold/contact/limit/grasp caches
publish only with the integrated state. Warm contact impulses are stored in
world coordinates on canonical endpoint B and reprojected into each refreshed
frame.

The collision probe keeps the legacy PSM/needle compatibility path and adds a
full-scene case containing a dynamic-dynamic sphere pair and a
dynamic-moving-kinematic pair. Both pair classes enter the generic solve, the
kinematic output velocity is preserved, and both warm starts rematch.

The trajectory probe closes both physical jaw coordinates around a grasp-zone
needle segment, holds for 100 steps, and commands the prismatic insertion axis
for 200 steps. No weld or attachment is created. Grasp status is derived from
two compressive jaw impulses, opposing normals, friction, post-solve slip, and
three-step dwell. A separate actual-PSM case activates an authored position
stop during two-jaw needle contact, proving contact and limit impulses coexist
in the monolithic solve and that the scalar limit warm start rematches on the
next step.

The supported pickup probe is deliberately stronger. A six-button cradle uses
three independently owned static support pairs at needle segments 6, 9, and
25; their triangle contains the needle COM with positive barycentric margin.
The PSM starts 4 mm away with open jaws, approaches without premature contact,
and closes four 0.35 mm teeth through legal computed-torque control on
authored grasp segment 17. The two tooth rows on each jaw are separated by
0.8 mm along the jaw axis; their actual load-bearing needle contact points are
about 0.449 mm apart.

Segment 17 is 4.833 mm from the needle COM. Its lift-start world orientation
produces an 8.407 µN·m gravity moment, so this is an observed off-COM load
case rather than a nominal placement claim. Success requires bilateral
load-bearing contacts on segment 17 in all 2,000 lift steps, distributed
two-contact-per-jaw evidence with at least 0.4 mm span in at least 90% of lift
steps and a continuous run covering at least half the lift, support unloading,
a long support-free interval, geometric fixture clearance, no robot/fixture
collision, needle motion along the jaw trajectory, bounded lift-relative
rotation, exact-cone KKT quality, and byte-for-byte rollback of all
state/cache streams after an injected non-finite force. There is no weld,
attachment, teleport, or collision filter between the robot and cradle.

Apple M4 Release result:

```text
articulated_rigid_collision
  articulated_shapes=20 rigid_shapes=32 contacts=4
  warm_matches=4 penetration=0.00103302716552
  normal_impulse=1.84515996103e-05
  island_contacts=2 dynamic_dynamic=1 dynamic_prescribed=1
  island_warm_matches=2 status=ok

articulated_rigid_world
  steps=300 contacts_max=3
  mixed_limit_impulse=0.00725238423535
  warm_matches_max=3 normal_impulse_max=0.00215094446817
  grasp_frames=298
  needle_displacement_mm=1.86475768885
  needle_lift_mm=0.395209689216
  needle_dz_mm=0.407338142395
  jaw_travel_mm=2.71564290541
  kkt_max=9.82853245044e-06
  grasp_slip_max=0.0332524969239
  grasp_identity_reset=pass grasp_dwell_saturation=pass
  rollback=pass status=ok

supported_needle_pickup
  steps=3030 support_buttons=6
  grasp_offset_mm=4.83317104204 gravity_moment_unm=8.40712343562
  support_triangle_margin=0.27617782522 support_contacts_max=6
  supports_at_lift=1 support_free_run=1979
  fixture_clearance_mm=8.0316551143
  art_dynamic_max=4 art_prescribed_max=0
  dynamic_prescribed_max=7 warm_matches_max=7
  solver_iterations_max=1491 solver_gauss_newton_max=339
  qualified_target_shape_frames=2269
  grasped_lift_frames=2000 target_shape_lift_frames=2000
  distributed_patch_lift_frames=1859
  distributed_patch_lift_run=1395
  jaw_a_contacts_max=2 jaw_b_contacts_max=2
  jaw_a_span_max_mm=0.449077938205
  jaw_b_span_max_mm=0.449132972612
  jaw_a_tooth_shapes=15,18 jaw_b_tooth_shapes=17,19
  grasped_lift_run=2000 final_grasp=1
  needle_lift_mm=8.06483626366 jaw_travel_mm=8.07909451418
  follow_ratio=0.998283892038
  orientation_drift_rad=0.00674117703419
  kkt_max=2.00466672669e-05
  observed_load_bearing_shape=17
  controller=computed_torque no_weld=yes
  ccd=conservative_discrete rollback=pass status=ok
```

The general world preserves all assembled witnesses by default. A probe may
explicitly cap a canonical endpoint-body pair: reduction retains the deepest
witness first, then fills the cap with deterministic maximin world-space
separation and stable-key tie breaking. The trajectory probe uses one witness
per pair; the supported pickup uses two so both longitudinal tooth rows remain
available. Replacing one rigid-shape generation after a qualified grasp proves
dwell resets instead of transferring to the replacement.

This milestone validates conservative-discrete, distributed contact pickup
for this segment-17 off-COM gravity load, not high-speed approach CCD or
generic six-dimensional force closure. Each retained witness still has only
normal plus two exact Coulomb tangent rows; no rolling or torsional friction
is synthesized. The tooth geometry is an explicit research default, not a
calibrated Large Needle Driver surface. Arbitrary needle orientations,
puncture, tissue, thread, cutting, biomechanics, clinical claims, multiple
articulations, and a batched device-resident Metal composition remain outside
this result.

## Generic Metal articulated operator

```sh
./build/bin/metalrobo_articulated_operator_gpu_probe
```

The Metal correctness operator consumes canonical articulation, body, joint,
and per-DoF records plus environment-major configurations and point-impulse
queries. For fixed or floating revolute trees within its declared capacity it
computes body COM poses, queried world points, analytic point Jacobians, a
diagnostic dense mass matrix, `J' p`, and factor-backed `M^-1 J' p`.

Clean Release Apple M4 result:

```text
floating 6-DoF analytic:
  pose=0 orientation=1.789394e-09 point=1.902070e-08
  mass=1.697124e-09 J=1.078162e-08 JTp=3.518265e-08
  dv=2.662635e-07 equation_residual=2.662635e-08

fixed 1-DoF analytic:
  pose=5.936634e-09 orientation=1.289274e-08
  point=3.878424e-09 mass=1.035631e-08
  J=3.878424e-09 JTp=6.517712e-09
  dv=3.059345e-09 equation_residual=2.615740e-09

Unitree G1 3-env x 35-DoF:
  pose=5.426786e-07 orientation=1.139133e-07
  point=5.587429e-07 mass=5.103648e-06
  J=5.401164e-07 JTp=2.294779e-06
  dv=3.682954e-05 equation_residual=1.044466e-05
  status_backward_error=4.794357e-11

G1 topology:
  bodies=30 joints=29 nq=36 nv=35 point_impulses=4
  deterministic_replay=yes transactional_failure=yes
  armature_canary=yes inertia_validation_parity=yes
```

This first implementation deliberately executes a lane-zero correctness path.
Its diagnostic dense `M` is not a production throughput representation, and
its timing is not a benchmark. It establishes real Metal execution of the G1
operator, not a device-resident dynamics/collision/solve/integration loop.

### Checked public host API

`runMetalArticulatedOperator` now provides typed inputs/results and derives the
raw kernel layout internally. It checks the canonical model, topology,
dimensions, arithmetic, the shader's 32-bit element-address ceiling,
capacities, finite values, point ownership, each actual Metal buffer length,
the device's per-buffer and aggregate working-set guidance, pipeline creation,
command completion, and per-environment GPU status. It owns all 15 compact
buffers, so callers cannot introduce aliases, offsets, or short raw bindings.
Predispatch failure leaves the caller's result bit-for-bit unchanged; after
dispatch, a batch and its status stream publish together.

```sh
./build/bin/metalrobo_articulated_operator_host_probe
```

Clean Release result:

```text
articulated_operator_host=metal device="Apple M4"
environments=2 q_elements=72 point_elements=4
mass_elements=2450 jacobian_elements=420
allocated_bytes=26008
replay=bitwise offset_articulation=pass predispatch_canaries=7
empty_buffers=pass gpu_status_publication=pass status=ok
```

The offset differential duplicates G1 as articulation 1 and proves that
nonzero articulation, body, joint, `q`, and `v` offsets produce byte-identical
physics payloads to the zero-offset model.

The seven predispatch canaries cover one-short packed `q` and point spans,
host-size overflow, shader-address overflow, declared point capacity, an
invalid canonical model, and invalid point ownership without dispatch or
result mutation. The zero-point/mass-disabled case proves that logically
empty streams receive typed dummy bindings. A deliberately non-finite derived
GPU result proves that completed environment failures publish the typed status
stream while leaving failed payload slots zeroed. Host latency is
intentionally omitted because this synchronous correctness wrapper recreates
its Metal context and allocations; it is not a throughput benchmark.

## Collision, manifolds, and the fifth pair class

```sh
./build/bin/metalrobo_collision_probe
./build/bin/metalrobo_collision_gpu_probe
./build/bin/metalrobo_deterministic_broadphase_probe
```

CPU FP64 and Metal FP32 now share ten supported analytic/SAT pair classes:

1. sphere/sphere;
2. sphere/plane;
3. capsule/plane;
4. box/plane;
5. cylinder/plane;
6. sphere/capsule;
7. capsule/capsule;
8. sphere/box;
9. capsule/box;
10. box/box.

Oriented cylinder AABBs and cylinder/plane witnesses are implemented on both
backends. Deterministic feature selection emits a four-point cap ring for a
parallel supporting cap, two ordered side-rim endpoints for a plane-parallel
axis, and one extremal rim point for the general tilted case. Collider-order
reversal, stable features, outward AABBs, exact capacity behavior, malformed
input rejection, and CPU/Metal witness parity are executable gates.

Clean Release CPU collision result:

```text
collision=cpu_fp64 pairs=6 raw_contacts=13 manifolds=6
sap_corpus_pairs=29 false_negatives=0 pair_classes=10
sphere_capsule_order=yes capsule_capsule_adversarial=yes
sphere_box_adversarial=yes capsule_box_adversarial=yes box_box_sat=yes
stable_ids=yes persistent_refresh=yes manifold_reduction=8_to_4
canonical_filters=yes overflow_transactional=yes finite=yes
```

Clean Release Metal narrowphase result:

```text
device="Apple M4" broadphase=metal_o_n2_baseline
shapes=17 pairs=9 raw_contacts=16
capsule_endpoint_contacts=2
box_raw_contacts=8 box_manifold_contacts=4
max_witness_error=8.195639e-08
cylinder_max_witness_error=9.536743e-07
capsule_pair_max_witness_error=8.940697e-08
sphere_box_max_witness_error=1.788139e-07
capsule_box_box_max_witness_error=9.536743e-07
pair_classes=10
sphere_capsule_order=yes capsule_capsule_adversarial=yes
sphere_box_adversarial=yes capsule_box_adversarial=yes box_box_sat=yes
cylinder_cap_side_rim=yes
cylinder_endpoint_order=yes
cylinder_aabb_tight=yes cylinder_adversarial=yes
canonical_filters=yes stable_features=yes
deterministic_replay=yes overflow_transactional=yes
finite_validation=yes strict_shape_flags=yes
strict_exclusions=yes strict_body_stream=yes
error_precedence=yes derived_transform_validation=yes
bounded_collision_domain=yes subnormal_policy=yes
quaternion_boundary_parity=yes zero_shape_world=yes
disabled_unsupported_skipped=yes status=ok
```

Clean Release deterministic flag/scan/scatter broadphase result:

```text
device=Apple M4 broadphase=metal_parallel_flag_scan_scatter
shapes=50 logical_pairs=1225 scan_blocks=5 candidate_pairs=47
cpu_parity=yes cylinder_plane=yes deterministic=yes
exact_capacity=yes zero_pair_worlds=yes
overflow_transactional=yes nonfinite_transactional=yes
shape_validation_transactional=yes strict_body_stream=yes
error_precedence=yes derived_transform_validation=yes
bounded_collision_domain=yes subnormal_policy=yes
quaternion_boundary_parity=yes unsupported_transactional=yes
global_append_atomics=none status=ok
```

This is still cylinder/plane support, not generic cylinder collision.
Cylinder/sphere, cylinder/capsule, cylinder/box, cylinder/cylinder, convex,
mesh, heightfield, and CCD remain unsupported. Box/box uses deterministic SAT
plus vertex/support witnesses; complete clipped face manifolds remain open.
The G1 shoulder cylinders stay simulation-disabled because the other required
pair and self-collision semantics do not exist yet—not because
cylinder/plane is missing.

The `O(n²)` Metal pair enumeration remains the generic correctness baseline.
The parallel scan probe is a micro-broadphase component, not a completed LBVH,
GPU manifold-persistence path, or throughput result.

## Pinned G1 compilation

```sh
./build/bin/metalrobo_g1_model_probe
```

Clean Release result:

```text
model="unitree_g1_29dof_rev_1_0"
mode_machine=5 mode_pr=0
bodies=30 joints=29 nq=36 nv=35
root_com_z=0.72396994 mass_kg=33.34114204
primitive_shapes=12 executable_shapes=8 foot_spheres=8
imus=2 max_inverse_error=1.02104631e-07 status=ok
```

The model retains factual URDF limits separately from the named Unitree RL Lab
drive/armature preset. Eight foot spheres are executable. Four shoulder
cylinders remain present but simulation-disabled for the collision-coverage
reason stated above.

## Additional clean component evidence

### Original Franka Metal runtime

The clean probe initialized the FP64 CPU reference and fixed-base Franka Metal
ABA from the same state and advanced one control step:

```text
max_q_error_rad=9.313226e-10
max_qd_error_rad_s=3.278255e-07
```

This is evidence for the original fixed-base Franka environment, not the
generic G1 world.

### Generic free-body dynamics

The FP64 implicit-midpoint oracle advanced a torque-free anisotropic body for
10,000 steps at 1 kHz:

```text
q_norm_error=5.115205e-09
angular_momentum_drift=3.065610e-06
max_newton_iterations=2
max_residual=8.295525e-16
```

The Apple M4 Metal probe covered static, kinematic, and anisotropic dynamic
bodies under implicit-midpoint and symplectic integrators:

```text
implicit_max_error=2.384186e-07
symplectic_max_error=2.384186e-07
energy_drift=4.951812e-08
angular_momentum_drift=2.890093e-08
midpoint_iterations=2
motion_contract=yes range_preflight=yes
```

### Throughput PGS and maximal-coordinate world

The clean fixed-budget CPU/Metal PGS probe agreed on a coupled two-contact
stack:

```text
max_linear_error=7.450581e-08
max_angular_error=1.192093e-07
max_impulse_error=2.980232e-08
max_cone_violation=0
momentum_xy_error=2.215217e-07
arithmetic_rollback=yes
```

The PGS block uses radial friction projection. It is not TGS and is not the
exact-cone effective-mass quality solve. Its hard limit remains 128 contacts
per connected island/dispatch.

The transactional maximal-coordinate CPU world held a two-sphere stack for
1,200 steps and ran a 240-step semismooth quality path:

```text
penetration_max=0
bottom_y=0.5 top_y=1.5
quality_kkt_max=4.812123e-17
manifold_constraints=4
rest_offsets=yes island_batched_contacts=129
quality_friction_rejection=transactional
implicit_midpoint_split=unsupported
overflow_transactional=yes
```

That world is independent-body evidence; it does not replace the new
generalized `ArticulatedWorld`. Constrained split worlds currently accept
symplectic Euler only; implicit midpoint is explicitly rejected because its
configuration increment cannot be reconstructed exactly from the post-contact
endpoint velocity alone.

### Original native runtime throughput

| Environments | Environment control-steps/s | Last GPU control step |
| ---: | ---: | ---: |
| 1,024 | 216,313 | 4.382 ms |

The clean run executed 1,000 control steps. Each environment control step
included four Franka ABA/contact/integration substeps plus observation, reward,
termination, and pose work. This is a local original-runtime result, not a
cross-engine benchmark and not generic G1 throughput.

### Device-resident contact world and MLX active encoder

The current Release probes are:

```text
metal_world_contact=ok environments=4 steps=12
retained_manifolds=1 franka_cube_contacts=2
isolated_overflow_required_raw=2
large_pair_stream=66049 large_pair_tail_contacts=1
mesh_candidates=2 mesh_ccd_events=1
cylinder_convex_ccd_events=1
tile33=33 tile96=96 tile257=257 tile513=513
tile257_spill_rows=675 tile513_spill_rows=1443
throughput_envs=1024
throughput_gpu_steps_per_s=33297.9
throughput_wall_steps_per_s=32148.7
gpu_batch_step_p50_ms=30.8742 gpu_batch_step_p95_ms=31.1237
wall_batch_step_p50_ms=31.6722 wall_batch_step_p95_ms=32.9228
throughput_active_contacts=2
wave_cohort=8
high_water_pairs=3 high_water_raw=2 high_water_manifolds=2
high_water_constraints=2 high_water_rows=6 high_water_islands=1
high_water_spill=0 retained_bytes=246061852 thermal=nominal
release_gate_40k=open

metal_world=metal device="Apple M4" abi=3
throughput_batch=4096 throughput_horizon=16
gpu_control_steps_per_s=244191 wall_control_steps_per_s=240281
gpu_p50_ms=268.074 gpu_p95_ms=269.537
wall_p50_ms=272.975 wall_p95_ms=274.349 thermal=nominal
replay=bitwise rollback=pass contact_graph=device_resident status=ok

{"mlx_world":"ok","mlx_version":"0.32.0",
"franka_environments":8,"g1_environments":4,
"compiled_policy_physics_reward":true,"deterministic_replays":100,
"isolated_failure_code":2,"fp64_max_q_error":8.003553e-11,
"fp64_max_v_error":7.450581e-09,"rollout_shape":[8,8,14],
"ppo_updates":2,"numpy_step_conversions":0,
"autodiff_rejected":true,"contact_world_supported":true,
"contact_blocks":[2,2,2,2],"contact_cache_explicit":true,
"persistent_wave32_packets":1}
```

The contact performance case uses an explicit 32-contact capacity class and
starts every environment with the same dynamic 1 kg cube touching Franka; the
observed high water is two active contacts. It is therefore real mixed-contact
execution, but not a 32-active-contact saturation result. Both measured rates
remain below the 40,000 control-steps/s release gate, which stays open. The
246 MB figure is the persistent Metal arena retained by the benchmark
context, not total process or MLX memory. The MLX probe now proves the
active-encoder contact graph, explicit manifold/pair-cache state, immutable
Wave32 packet generation, fixed-grid persistent packet pulling, and optimizer
integration. It does not prove policy convergence.

Hybrid CCD currently performs deterministic conservative advancement, orders
the event prefix, clusters simultaneous impacts, and reports explicit
speculative-remainder use through ABI-v3 event cursors. It still solves that
certified remainder as one speculative TGS interval. Literal repeated
TOI advance/impact-solve/continue splitting is therefore not yet release
evidence.

## What remains unvalidated

- Trajectory/contact comparison against pinned MuJoCo, Genesis, or another
  independent simulator
- A measured 32-active-contact saturation result and the 40,000 control-step/s
  Franka-plus-object gate on M4
- Per-complexity solver queues instead of the current homogeneous cohort
  selection used by cloned RL batches
- Long-horizon controlled G1 contact stability, locomotion learning, and RL
  throughput
- Multi-articulation islands and long-horizon/large-island mixed-scene
  stability beyond the focused dynamic-dynamic and supported-pickup cases
- GPU ConstraintIR joint limits, implicit drives, equality/loop blocks, patch
  rolling/torsional resistance, and force/torque sensors
- Articulated self-collision, loop constraints, and unsupported pair classes
- Calibrated surgical jaw surfaces, rolling/torsional resistance, and generic
  grasp-wrench/force-closure certification beyond the tested segment-17 load
- Long-horizon dual-PSM needle/thread transfer with thread self/tool collision;
  the landed swage coupling currently proves deterministic two-way impulse
  transfer and transactional state publication, not knot tying
- Production segmented LBVH and heightfield collision
- Matrix-free Newton-PCG for large exact-cone quality islands
- Quality/throughput task-level closure for the landed temporal TGS
- Qualified derivatives through impact, friction-regime, and active-set
  changes
- Cross-machine performance/reproducibility and any superiority claim over
  MuJoCo, Genesis, or NVIDIA simulators
