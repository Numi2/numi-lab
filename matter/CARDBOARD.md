# Cardboard from manufactured structure

The native Matter instrument represents two paper liners, a corrugated paper
medium, and finite starch-glue regions as distinct three-dimensional FEM
materials. It uses published inputs and keeps missing measurements explicit.
The purpose is to derive folding from material response and contact, rather than
assigning a weak hinge to a box-shaped elastic object.

## Starting material

The default `--preset literature2009` builds a 31.6 × 20 × 4.210 mm C-flute coupon with
7.9 mm pitch, 0.277 mm liners, and a 0.191 mm medium. The sinusoidal medium has
constant normal thickness. Separate upper and lower glue bridges connect the
paper with conforming shared interfaces; there are no direct liner-to-medium
bonds. The lower glue region has a nominal 0.8 mm horizontal footprint and
0.1 mm minimum gap; the upper uses 0.6 mm and 0.03 mm. End patches are clipped
to the coupon. The manifest records their actual width and area. Its
`medium_centerline_height_m` denotes peak-to-trough rise, not an absolute
vertical coordinate.

This is a **cross-source recipe**, not an exact reconstruction of a measured
board lot. [The manufacturing sources](examples/cardboard/manufacturing-sources.md)
and [machine-readable manifest](examples/cardboard/manufacturing-source-manifest.json)
identify each source value, proxy, derived quantity and numerical assumption.
In particular, the paper calipers/densities and finished-board caliper are
borrowed from separately reported specimens. Glue density is assumed.

The initial condition is preformed, bonded, conditioned and stress-free.
Paper forming through corrugating rolls, residual manufacturing stresses,
starch gelatinization/curing, drying and moisture transport are not simulated.
These omissions cannot be hidden by calling the starting geometry a simulated
manufacturing process.

## Paper and adhesive mechanics

The Haj-Ali 2009 paper assets use direction-dependent elastic constants in the
machine, cross-machine and thickness axes. The medium's material frame follows
the flute tangent. Additive plastic Green strain and an associated Hill yield
surface provide irreversible material memory through Matter's implicit-state
constitutive solver. The constants describe a compression-calibrated,
ideal-plastic approximation: the source's bilinear hardening slopes were not
tabulated, and a zero slope is an explicit modeling choice. The symmetric Hill
surface does not reproduce the source's distinct tension/compression strengths.
It is not a complete paper failure or crease law.

The six `ep` components are plastic strain; algebraic stress variables are
scratch state and must not be counted as permanent deformation. Source-bound
CPU and Metal material-point checks compare stresses, return states, yield,
work and tangents. They verify the implemented law, not agreement with a real
folded box. Matching elastic-control assets hold plastic strain at zero while
preserving the elastic constants and density.

Finite glue uses the reported 400 MPa modulus with an assumed Poisson ratio
0.3, density 1500 kg/m³ and a Neo-Hookean finite-rotation extension. It resolves
bond geometry and elastic compliance. Its bonded interfaces cannot debond;
there is no calibrated adhesive strength, fracture energy or curing law.
Deformable self-contact is enabled for finite-glue geometry; its friction is
currently zero. `--without-self-contact` is a numerical diagnostic control.

The original `--preset nagasawa2013` remains available for the retained elastic
line-tie experiments. Its source-arc profile check preserves an inconsistency:
interpreting the reported 1.7 mm radius as a centerline radius with the stated
pitch and angle gives approximately 5.728 mm total caliper, not 5 mm. The check
does not silently rescale those inputs. The executable geometry is explicitly
a sinusoidal approximation.

## Build and verify

```sh
cmake -S matter -B build-cardboard -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-cardboard --target numi-matter-cardboard-probe \
  numi-matter-cardboard-profile-check numi-matter-cardboard-glue-mesh-check \
  numi-matter-cardboard-paper-material-check numi-matter-cardboard-paper-gpu-check \
  numi-matter-cardboard-material-check numi-matter-cardboard-gpu-check \
  numi-matter-cardboard-tooling-check numi-matter-cardboard-tooling-native-check \
  numi-matter-cardboard-barrier-scale-check \
  numi-matter-fem-kinematic-release-check numi-matter-fem-reference-check \
  numi-matter-cardboard-box-blank-check numi-matter-cardboard-delamination-reference-check \
  numi-matter-cardboard-delamination-gpu-check numi-matter-cardboard-preconditioner-gpu-check -j4
ctest --test-dir build-cardboard -R '^matter\.(compiler|metal)\.cardboard_' --output-on-failure
build-cardboard/numi-matter-cardboard-probe --help
```

Compile and inspect the structure without advancing physics:

```sh
build-cardboard/numi-matter-cardboard-probe --preset literature2009 \
  --compile-only --nx 8 --ny 1 --thickness-slices 1 --output /tmp/cardboard-geometry
python3 matter/tools/audit_cardboard.py /tmp/cardboard-geometry/mesh.json \
  /tmp/cardboard-geometry/initial.obj /tmp/cardboard-geometry/audit.json
```

Output directories must be new or empty. Mesh JSON stores authored FP64
coordinates while OBJ snapshots contain the native FP32 positions; small
initial differences include quantization. The audit requires NumPy; the
renderer additionally requires Matplotlib. Rendering uses exported geometry
and does not manufacture or amplify deformation.

## Loading protocol and evidence

The literature preset defaults to a coarse diagnostic mesh (8 subdivisions
per pitch, 1 across width and 1 through paper thickness), a 0.25-degree grip
rotation, 25 µs timesteps, and 32 loading + 16 hold + 32 unloading + 64 constrained
settling steps. Its total physical duration is 3.6 ms. These are working
numerical instrument settings, not a mesh-converged folding experiment.

The left grip is fixed. The right grip rotates about its section center with
translation constrained, so this is a combined bend/shear test, not pure-moment
or four-point bending. `--steps`, `--hold-steps`, `--unload-steps` and
`--relax-steps` specify a continuous loading cycle. Unloading rotates the
right grip back to zero, and relaxation keeps **both grips clamped** at that
angle. This is constrained settling; its residual shape cannot establish a
freely unloaded crease.

`--release-steps N` appends a native right-grip release; the left grip remains
fixed to remove rigid-body drift. Use `--unload-steps 0 --relax-steps 0` to
release directly from the loaded/held angle. Release changes the candidate
constraint tag and inverse mass, retaining accepted position and momentum
history; rejected steps roll it back. It does not reset plastic state.
`right_grip_constrained`, `current_fixed_nodes`, and
`measured_right_grip_angle_deg` expose the actual accepted boundary and angle.
The latter is the orientation of a least-squares end-section thickness vector,
not a prescribed fold angle or a local crease measurement. During release,
`target_angle_deg` is inactive. A released section can still be moving.

Example: `--steps 32 --hold-steps 16 --unload-steps 0 --relax-steps 0 --release-steps 64`. The probe runs an unloaded
reference in a separate environment, exports phase-end geometry and material
history, and records accepted-state solver certificates on every step.
A short dynamic cycle does not establish a quasi-static fold.

`--newton-iterations` controls the global nonlinear budget (7 legacy, 14
literature preset). `--fgmres-iterations` controls the global Krylov budget; `--material-iterations`
controls local constitutive Newton iterations in [1,16]. Legacy defaults remain
10 and 8 respectively; the literature recipe selects 32 and 16. These iteration budgets do not
change acceptance tolerances. `--residual-tolerance` explicitly tightens the
nonlinear threshold (it cannot loosen the default `1e-4`). The material budget is bound into the compiled
world fingerprint. A state with both `update` and `implicit` uses `update` as
the local Newton starting guess and `implicit` as the equation to solve; an
initializer never replaces the constitutive residual. Raw status diagnostic slots have status-dependent meanings;
use the separately exported certificate residual rather than a successful
status slot that happens to be zero. The certificate raw accepted flag can
remain one on a rejected global step in this runtime; `step_accepted` therefore
requires successful status as well as finite, in-tolerance residuals.

A lasting crease requires more than a residual displacement: examine the six
plastic-strain components, unloaded-reference drift, residual speed/kinetic
energy, reaction forces, and sensitivity to mesh/time resolution. Elastic
material may retain visible motion during relaxation. Do not label that motion
plasticity. An unsuccessful return map or nonlinear step remains failed
numerical evidence even if a smaller load later passes.

The final default cycle and a half-angle cycle each accepted 144 native steps,
retaining nonzero plastic strain while the unloaded references retained zero.
An otherwise matching elastic control also completed 144 steps. Nine targeted
compiler/Metal checks passed. These are implementation checks, not physical
folding validation.

See [retained evidence](examples/cardboard/evidence/README.md) for exact source
and binary identities, accepted runs, failed trials and unresolved gates.
Published crease curves, adhesive failure, physical creasing-tool response and
full-box folding/assembly remain separate validation gates. This instrument does not
claim physical qualification of those behaviors.


## Moving creasing tool

`--crease` runs a rounded capsule punch against a finite rigid backing anvil.
The board is the same explicit paper/flute/glue mesh; both end grips remain at
zero rotation. Only boundary-face nodes participate in rigid contact, excluding
internal tetrahedral nodes and conforming material interfaces. The punch is a
native kinematic body: each step supplies its endpoint pose and consistent
velocity for preDynamics, followed by that realized endpoint for native
postCommit contact certification and rollback. Board displacement is solved through contact, not imposed as an
indentation constraint. A second environment keeps the punch stationary.

```sh
build-cardboard/numi-matter-cardboard-probe --crease --indent-mm 0.020 \
  --reference-displacement-gradient --prescribed-tool-predictor \
  --residual-tolerance 1e-6 --newton-iterations 28 --fgmres-iterations 64 \
  --steps 32 --hold-steps 16 --unload-steps 32 --relax-steps 16 \
  --output /tmp/cardboard-tool-trial
```

The default 0.750 mm nose radius and 0.010 mm initial clearances are exploratory
numerical choices, not a reproduction of a published scoring tool. Both
clearances must exceed the contact slop. `--indent-mm` denotes punch travel from
its initially separated pose, including the initial clearance. Release mode and
crease mode cannot be combined. The punch starts at the coupon center and spans
its width. Tool speed follows the load/hold/unload phases and becomes zero in
settling. No friction coefficient or damage law has been calibrated.

`tool-observations.csv` records commanded travel, native reaction impulse on the punch divided
by timestep, contact counts, and node-sampled signed gaps against the commanded
end pose and fixed anvil. Gap samples are not a continuous surface collision
proof. The probe stops if a commanded endpoint penetrates beyond an FP32
coordinate-roundoff allowance, even when the native step itself accepted. The initial caller failed that independent check. The corrected caller
now supplies the actual endpoint to native postCommit certification and rollback.
The earlier endpoint-only caller completed a 10 µm shallow cycle and rejected
the 20 µm attempt. With the native feasible predictor and persistent
displacement enabled, the 20 µm cycle now completes all 96 steps. Its minimum
sampled tool gap is 80.03 nm and peak punch reaction is 0.7313 N; the stationary
control has zero punch force. These values use the exploratory `1e-4` residual
threshold and are not calibrated scoring forces. At default barrier stiffness, a further 50 µm trial rejects after 15 accepted
steps. Opt-in stiffness factors 2 and 4 each complete that 96-step cycle with
minimum sampled gaps of 53.46 and 80.16 nm. Their peak punch forces differ by
15.64% (2.316 versus 2.745 N), so these completions do not establish force
accuracy or physical crease formation. At `1e-6`, factors 2 and 4 also complete all 96 steps, and their peak punch
forces differ by 0.746% (0.3878 versus 0.3907 N). Full withdrawal force curves
still differ by 22.60% relative RMS (0.00213 N absolute RMS). At the same tighter
tolerance, factor 1 accepts loading and hold but rejects the first withdrawal
step; tighter tolerance alone does not recover its complete cycle. These results
do not identify physical damage or certify mesh convergence. The default
remains factor 1.
Failed-step diagnostics are retained and must not be interpreted as
accepted force or geometry at the attempted tool position. The arm names are
`indented` and `stationary_tool_reference`.

## Numerical reliability gates

`--reference-displacement-gradient` selects an immutable-reference FEM path
that stores displacement separately from absolute FP32 position. It evaluates
`F = I + grad(u)` from that persistent displacement and materializes `X + u`
for contact and observation. This avoids losing small displacement increments
when subtracting two large absolute coordinates. The reference nodes and their
identity are bound into the compiled world. Candidate updates, snapshots,
rollback, reset and physical-state hashes include the displacement state.
Matter ABI 40 increases the FEM node-state stride from 64 to 80 bytes; old
compiled packages must be rebuilt. The generic default arithmetic is unchanged.

The new path completed a 48-step loading/hold trial at residual tolerance
`1e-6`, Newton budget 28 and FGMRES budget 64. Its stationary control retained
exactly zero force and motion. The previous absolute-position path rejected
its first loaded step at that tolerance. A further `1e-8` trial still rejected
the first loaded step, with residual `2.31384e-7`; the precision repair does
not imply arbitrary accuracy or mesh/time convergence. The half-timestep arm
completed 96 loading/hold steps; endpoint moments changed by at most 3.474%
and paper plastic-strain RMS by 0.887%. This passes the narrow 10% endpoint
screen while retaining unequal equivalent force bounds (0.04 versus 0.08 N).
The length-refined mesh still rejected its first loaded step at this tolerance,
and the thickness-refined mesh rejected after two accepted steps. Increasing
the Krylov budget to 256 recovers all 48 increments of the thickness-refined
arm, but its matching coarse control rejects after 29 increments. Length
refinement still rejects its first increment even with 256 columns. These
results do not establish mesh convergence.

A subsequent same-build pair also halves the residual tolerance with timestep,
matching the tolerance/timestep ratio. Both complete. Loading/hold endpoint
moment differences are below 0.93% and paper plastic-strain RMS differences
below 0.23%; reaction-force curve relative-RMS differences are at most 6.22%.
The measured maximum free-force imbalance is below 0.04 N in both. This is
improved time sensitivity for the loading/hold instrument, not mesh or
full-cycle physical convergence.

`--fgmres-preconditioner regional-tangent-diagonal` is an experimental pure-FEM
option using regional constitutive tangents and current contact curvature.
A separate direct Metal oracle checks its diagonal against force finite
differences. Both board trials at `1e-6` and `1e-8` nevertheless rejected their
first step; there is no demonstrated robustness or performance improvement.
The default remains `scalar`.

`NM_FEM_NEWTON_TRACE_ROOT=N` enables a bounded, read-only diagnostic for one
control step in a single-object, static pure-FEM world. It snapshots the stored
Newton-start residual and four step-fraction boundaries, followed by the
terminal certificate before commit. The next iteration's reassembled norm is
the preceding step's residual; the Arnoldi estimate is not a direct `b-Ax`
measurement. Trace support is limited to one physics substep, at most four
environments, 4,096 FEM nodes and 32 Newton iterations. Unsupported scopes
emit explicit skip records. Decode a captured stderr log with:

```sh
python3 matter/tools/cardboard_newton_trace.py trace.log --root 0 \
  --near-full-alpha .99 --growth-limit 1.1 --output newton-analysis.json
```

The decoder binds the log and its own source by SHA-256, refuses to overwrite
an existing report, and returns failure for incomplete, skipped or invalid
captures. A raw failed status overrides the certificate's informational flag;
status zero at this capture does not establish a later accepted commit. This
instrument changes no numerical acceptance threshold or globalization rule.

`--prescribed-tool-predictor` opts a translating capsule into a native feasible
Newton initializer and node/capsule swept-path certificate. It supplies only a
candidate starting guess; the usual residual, material, volume and contact
gates still decide acceptance. Native checks reject inconsistent successive
poses, rotation, nonzero inverse mass/inertia and a changed postCommit endpoint.
`--prescribed-tool-barrier-scale N` applies a bounded `[1,100]` numerical
stiffness multiplier to the prescribed punch. The default 1 takes the existing
arithmetic path. Impulse, tangent curvature and line-search potential scale
together; activation distance, collision clearance and acceptance gates stay
fixed. The effective FP32 factor is recorded; a requested double is retained
separately when rounding changes it. With nonzero Coulomb friction, scaling the
normal impulse also scales friction capacity; the current cardboard contact
uses zero friction. This is a conditioning experiment motivated by
[IPC's barrier-stiffness discussion](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf),
not a material parameter or implementation of that paper's adaptive algorithm.
It requires the prescribed tool path for nondefault values.

Failures restore the previous native rigid pose and FEM state. After reset,
the first start pose is inferred from the supplied endpoint and velocity.
This is a nodal swept test, not continuous triangle-surface collision coverage.

## Box blank and source-calibrated material coupons

`--box-blank --bend-angle 0` builds a slotted FEFCO 0201 flat blank with the
same separate liner, medium and finite-glue regions. It exports panel dimensions,
score centerlines and an ordered assembly-target schedule in `box-layout.json`.
The scores remain metadata and all assembly targets remain unexecuted. The
native probe currently holds the blank at rest; it does not physically score,
fold, fasten or close a box. See the [blank geometry and source boundaries](examples/cardboard/cardboard-box-blank.md).

The [Tryding/Biel bleached-paperboard coupon](materials/tryding2023_bleached_paperboard_modeI_coupon.md)
implements a separately sourced positive-opening Mode I delamination law.
It passed 11,894 direct native material-evaluator checks on Apple M4, including
loading work, unloading, tangent comparison and 1,000 holds without damage
creep. These are constitutive evaluations with zero physical FEM steps.
Its single-ply bleached, coated paperboard is a different material from this
corrugated recipe. It must not be installed as a calibrated starch-glue law or
as a general bulk cardboard material. Native interface traction, mixed-mode
debonding and grade-matched crease damage remain separate implementation gates.

The full 4,308-node, 10,356-tetrahedron flat blank also completed two native
zero-load steps with exactly zero motion and force imbalance. That is a rest
and integration check, not physical assembly. New trials and retained failures
are in [precision and scoring evidence](examples/cardboard/evidence/native-precision-and-scoring/README.md).

## Earlier resolution evidence

[Release and resolution evidence](examples/cardboard/evidence/release-and-resolution/README.md)
records the sealed hypotheses, failures and follow-up experiments. The initial
10% phase-end sensitivity screen was contradicted by time and length refinement;
the thickness-refined run failed. A passing status alone is insufficient.

The final unpreconditioned mechanical residual is now exported as
`free_force_imbalance_l2_N` and `free_force_imbalance_max_N` after division by the
actual timestep. This is force minus inertial response on free nodes, including
contact; it is not the grip reaction or a static equilibrium claim. Values on
rejected steps are diagnostics only. The general solver certificate normalizes
its coupled residual by a norm floored at 1; for this small FEM-only coupon the
`1e-4` threshold acts as an absolute impulse floor. Halving the timestep therefore
loosens the equivalent force bound. Tighter tolerances require their own accepted
runs and sensitivity checks; they are not automatically better-validated physics.

`cardboard_convergence.py` compares complete accepted bend/release runs, binds
input files by SHA-256, and reports tensor plastic-strain metrics with the
correct shear weighting. Optional thresholds are descriptive numerical screens.
It refuses incomplete trajectories and crease-mode data rather than assigning
a fake endpoint. `--force-matched-tolerance` is an explicit temporal-study
option: only the residual tolerance may differ, and its ratio to timestep must
match within one part per million. The report includes measured free-force
imbalances and keeps the absolute force gate unknown when its normalization
scale is not exported. `plot_cardboard_release.py` plots actual accepted angle and
velocity, without magnifying geometry. See the [reference gap map](examples/cardboard/crease-fold-gap-priorities.md)
for published scoring/folding candidates and missing material information.
