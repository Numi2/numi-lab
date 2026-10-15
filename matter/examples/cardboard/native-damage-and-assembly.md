# Native damage, interface and assembly follow-up

This follow-up extends the native Matter owner. It distinguishes a published
single-ply paperboard constitutive benchmark from the existing corrugated
liner/medium/starch recipe. Parameters do not transfer between those grades.
The working evidence is retained under
`/Users/home/cardboard-followup-20261013`. The curated records are in the
[native evidence bundle](evidence/native-damage-and-assembly-20261013/README.md);
the bundle carries a source-bound [summary](evidence/native-damage-and-assembly-20261013/supplemental/bulk/evidence-summary.json)
and [frontier](evidence/native-damage-and-assembly-20261013/supplemental/bulk/evidence-frontier.json).
Original working cases remain in the external evidence workspace.

## Material implementation

`matter/tools/robertsson2023_material.hpp` implements a 27-state native
constitutive slice derived from the finite-strain elastoplastic and staggered
shear-dissipation damage model in [Robertsson et al. (2023)](https://doi.org/10.1002/pts.2774).
Its substrate is
0.39 mm single-ply softwood paperboard, density 788 kg/m³. It is not a local
damage calibration for the 205 g/m² kraft liner, 126 g/m² NSSC medium and
separately sourced dried-starch modulus used by the corrugated probe.

The implementation uses 27 persistent scalars: nine plastic-deformation
gradient components, twelve subsurface hardening variables, an accumulated
plastic multiplier, four critical dissipation values frozen at damage onset,
and scalar damage. The material ABI is 41, with a capacity of 32 scalars and
64-bit state-transfer masks. Expression caching is opt-in for the large
constitutive DAG; ordinary material programs retain their uncached bytecode.
Cache and state-capacity tests are implementation checks, not physical
calibration.

Damage uses only the source's four out-of-plane shear subsurfaces, with
reference beta `[2.85,2.85,3.98,3.98] MPa^-1` and c `[8,8,9,9]`. The source's
projected-brick-area scaling is an empirical mesh correction, not a proof of
mesh objectivity. A tetrahedral adaptation has not been physically validated.
There are two additional source-mapping limits. The material builder defaults
both MD and CD reference areas to the paper's `0.0011 mm^2` brick reference
area, giving unit area ratios; it does not compute each tetrahedron's
material-direction projected areas from source Eq. 26. This is not the
paper's element-level mesh correction, and no tetrahedral mesh-regularization
claim follows. The authored ZD-compression branch differs by the additive
constant `E6*(1-1/E7)=0.333260869565 MPa` from printed Eq. 18. This leaves
fixed-damage stress unchanged but changes damage-conjugate `A`. The source also
states `Psi0 >= 0`, which conflicts with the printed Eq. 18 compression
branch; whether the equation has a typo is unresolved. Exact source energy and
damage-force reproduction is therefore not claimed. The transcription and
limits are documented in
[`robertsson2023_bulk_damage_architecture.md`](../../materials/robertsson2023_bulk_damage_architecture.md).
The current material-point benchmark uses the reference coefficients and the
line-creasing damage cap 0.96. It executes no FEM steps; this statement applies
to the material-point benchmark only. A separate, fully prescribed one-tetrahedron
FEM integration-parity experiment is summarized below. That manufactured test
checks native FEM/point coupling, not physical validation.

The production constitutive evaluator supplies diagnostic flags, the scaled
local residual, and both accepted and partial candidate states on failure.
End-to-end algorithmic-response validity includes projection, stress, and
implicit tangent evaluation. A rejected candidate is never promoted to the
accepted history. In the initial high-shear `source-point-003` trial, 119
material-point responses were accepted through gamma 0.59, reaching damage
0.348892; gamma 0.595 was rejected before unloading. Its older instrument did
not preserve the failed candidate state or identify the terminal subtype.
`source-point-006` replayed the same 0.005-increment loading path without
changing constitutive parameters or tolerances. It again accepted gamma 0.59
and rejected gamma 0.595 after 16 local projection iterations: residual
infinity norm `4.8307702e-5` exceeded the FP32 convergence threshold
`2e-5 * (1 + state_inf) = 4.3015134e-5` by `5.292568e-6`. The terminal subtype
was `projection_iteration_exhausted`; the last committed damage was
`0.34889245`, while the rejected candidate damage `0.35250464` was not
committed. This diagnoses a local return-map convergence limit in the authored
27-state material-point path, not a global FEM failure or a physical failure
criterion. Captured prediction, result, all 27-state histories and the verified
input bindings are in
`/Users/home/cardboard-followup-20261013/bulk/source-point-006/`.

The seeded high-shear continuations separately resolve the specific
single-increment rejection: one increment reproduces the source-point-006
failure exactly, while two and four subincrements accept from the exact
accepted `gamma=0.59` state through `0.595` and return only to `0.59`. At
`0.595`, their maximum state difference is `3.5465e-6` (state 21), `P13`
differs by 7 Pa and damage by `3.8743e-7`; after the short return to `0.59`,
`P13` differs by 2.5 Pa. All 362 sealed inputs per run verify. This is local
increment-refinement evidence under the unchanged material program; the small
nonzero differences do not establish convergence, and the path does not unload
to zero shear. Results and source bindings are retained in
`/Users/home/cardboard-followup-20261013/bulk/source-point-seeded-{1,2,4}/`,
with aggregate `bulk/seeded-sensitivity.json`.

The separate diagnostic `0 -> 0.3 -> 0` cycle accepts all 121 material points
at 0.005 shear increments. Damage starts at shear 0.17, reaches 0.119373 at
peak shear and 0.245010 after reverse plastic flow on unloading. The final
P13 stress is -4.974485 MPa at zero commanded shear. No state is reset between
increments. The maximum local Newton count is ten. This shorter low-shear
cycle is not the high-shear continuation or its unload-to-zero.

## Fully prescribed FEM integration-parity experiment

A separate `fem-coupling-001` run integrated the same authored 27-state
Robertsson program through production FEM and point-material paths. One
four-node tetrahedron of thickness 0.00039 m had all four nodes prescribed to
`x = X + gamma*Z`; it had no free mesh degrees of freedom. The path was
`gamma: 0 -> 0.3 -> 0` with 60 increments on each leg, 121 matched rows
including the initial state, and a 0.005 s timestep. All FEM steps returned
status zero, and all point responses were accepted.

Against the same-step point response, maximum absolute state difference was
`1.63912773e-7` (state 17, step 47). The unregularized relative error reached
1.074 near a zero-valued state, so the sealed comparison used its mixed
absolute-relative tolerance; maximum utilization was 0.0006781. The maximum
reaction `P13` difference was `5.31275666 Pa` at step 47, with maximum
mixed-tolerance utilization 0.0004088. Final damage matched at `0.245009735`;
final FEM and point reaction stresses were `-4.97448550 MPa` and
`-4.97448500 MPa`. The test also exercised negative-J rejection, selective
reset and snapshot restore. The negative-J assertion preserved the accepted
FEM history and held control, although the fixture log did not print the exact
failure status code.

This is manufactured numerical coupling verification: it shows state and
reaction-history parity when the FEM deformation is fully prescribed. It is
not an independent constitutive implementation, free-mesh response or
convergence study, experiment, or physical validation. Sealed artifacts are
in `/Users/home/cardboard-followup-20261013/bulk/fem-coupling-001/`.

## Interface coupling

An explicitly authored paired-face path now assembles positive Mode I forces,
the algorithmic and geometric tangent, reference-area-weighted work, and
transactional interface history in FEM. It uses the separate
[Tryding/Biel source coupon](../../materials/tryding2023_paired_modeI_native_coupling.md).
The `cohesive/native-003` test passes 4,004 checks: the interface changes the accepted opening,
preserves history on rejected closure, and agrees with independent traction,
damage and reference-area work calculations. Equal-area coarse and refined
faces each complete a 56-step load, unload, reload and reopen path. Native
reaction finite differences reproduce the published unloading modulus with
relative errors 3.173e-5 and 3.169e-5, respectively. In the corrected
`native-tests-009/cohesive` repeat, all 4,004 checks pass, with measured
unloading-tangent errors 3.17262e-5 and 6.77552e-4. Both measurements are
retained; the smaller original error is not substituted for the repeat.
The corrected `native-tests-011/cohesive-reset` fixture passes 4,072 native
checks, including episode history reset/isolation and rejected-reset
checkpoint verification. It remains a separate single-ply bleached-paperboard
coupon; these software checks do not qualify the corrugated board or its starch
interface.
`native-tests-012/cohesive-reset` repeats the 4,072-check reset and coupon suite
successfully. Its `native-tests-012/capsule-box` sibling also passes the
capsule/BOX Armijo, endpoint rejection, rollback, temporary-grip and static
support checks. These remain software regressions on the separate bleached
paperboard coupon and synthetic tooling fixtures.
The `native-tests-013` cohesive-reset and capsule/BOX runs pass again under the
published runtime-policy revision: 4,072 cohesive checks, plus native endpoint,
rollback, grip and Armijo checks. These repeated software regressions do not
calibrate the corrugated starch interface or qualify physical assembly.
This is an operator proxy,
not a direct matrix-free tangent comparison. This bounded result does not qualify mixed mode,
closure contact, crack propagation, or a corrugated starch bond.

The source law approaches complete damage asymptotically. No invented opening
cutoff turns it into a finite rupture rule. Published pin-adhesion or peel
forces alone do not uniquely identify local starch traction, fracture energy,
unloading, and mixed-mode behavior. The current corrugated glue remains
elastic and tied.

## Mesh and solver instrumentation

The frozen-028 `solver/mesh-armijo-004` matched paths remain incomplete at all
three tested resolutions. Nx8 accepted 43 steps before rejecting step 43
(status 10, 1,015 FGMRES iterations; base/trial residuals
`1.0010345e-6`/`1.0015222e-6`). Nx16 accepted 19 steps before rejecting step
19 (status 10, 390 iterations; base/trial residuals
`7.9471856e-6`/`7.9685842e-6`). Nx32 timed out at 1,200.027 s after 17 accepted
rows without a terminal native rejection record. These are incomplete path
coverage results, not mesh convergence.

The selected no-contact nx16 production operator-versus-residual comparison
uses the actual Newton correction at root 19, iteration 3. The selector-off
frozen-031 baseline and all three diagnostic captures accept 19 steps and
reject step 19, with byte-identical observations, result JSON, compiled
material pack and all 41 exported mesh hashes. All recorded inputs verify:
the scale-1 input map is identical to the baseline, with 360 files each.
Scale-8 and scale-16 each bind 361 files, including changed host diagnostic
code and added documentation. A separate
cohesive test fixture fixes its per-environment kinematic-target count, and
the production `NumiMatter.metallib` is byte-identical. The complete source
and runtime identities are retained with the case records.

| Diagnostic epsilon | Relative operator/FD difference | Action/FD cosine |
| ---: | ---: | ---: |
| 1/32 | 1.022397 | -0.067131 |
| 1/64 | 1.024829 | 0.001745 |
| 0.125 | 0.946589 | 0.399366 |
| 0.25 (scale 8) | 0.920136 | 0.431603 |
| 0.25 (scale 16) | 0.920136 | 0.431603 |
| 0.5 | 0.898466 | 0.456066 |

The epsilon-0.25 measurement is exactly repeatable across host scales 8 and 16.
All nonzero pairs had valid stage status and positive minimum J. The additional
environment had a zero correction direction, so its zero error is a vacuous
control. The measured discrepancy varies with epsilon but persists; it does
not identify its cause or certify the production Jacobian. A hash-bound audit
found some tetrahedra near the Hill yield surface at accepted step 18, but has
neither the iteration-3 candidate nor its projected perturbation states and
does not prove yield-transition crossing. The frozen-034 `operator-fd-scope`
software case accepts its single eligible step in 4.496 s; that exercises the
finite-positive-mass scope admission only.
The corresponding frozen-036 scope regression also accepts its single
eligible finite-positive-mass step; runtime-policy revision 2 versions the
host execution contract, with the production Metal library unchanged.

An isolated `solver/local-projection-accuracy-035` alternate Metal build
changes a local projection residual threshold from `2e-5` to `2e-6`; it leaves
the production checkout/default, local iteration budget 16, global tolerance
`1e-6`, host executable and physical parameters unchanged. Its isolated native
run exited 2 after two accepted steps; the next attempt was rejected with
status code 5 at object 0, element 1430, before any microstep completed or
FGMRES iterations ran. This experiment did not complete the target path. The
status is retained as reported; it does not identify the cause. The tighter
threshold remains unadopted and the production default remains `2e-5`.

An opt-in frozen-operator measurement recomputes the actual linear `b-Ax`
after each FGMRES restart. On the rejected first nx16 grip increment, 728
measurements differed from the Arnoldi estimate by at most 0.01073% relative.
The diagnosis-only run preserved CSV observations and all accepted/failure
meshes byte for byte. Linear residual drift is therefore not supported as
the principal explanation for this case's nonlinear failure.

The legacy explicit-FEM line search checks admissible deformation but has no
free-DOF residual descent test when no mixed-volume rows exist. An isolated,
opt-in contact-free experiment evaluates at most eight candidate corrections
in private arenas against the production mechanical residual and the existing
Armijo coefficient. It leaves accepted material history unchanged and retains
the final production certificate. In a matched first-step experiment, the legacy path rejected the nx16 step
while residual Armijo accepted it at residual 6.900e-7 and minimum J 0.999539.
Newton corrections at iterations four to six selected alpha 0.25, 0.5 and
0.5; the previous rising residual was avoided. The remaining iterations
were certified zero-correction no-ops. This one-step result is not mesh
convergence. `NM_FEM_RESIDUAL_ARMIJO_ROOT=all` selects each control step
within the same bounded scope; contact and cohesive transactions remain
outside it. No acceptance tolerance is relaxed.

A separate selector governs the prescribed-capsule plus static-BOX Armijo
path: `NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT=<control-step|all>`. A matching
numeric step or `all` selects the same transaction for both the BOX endpoint
certificate and the optional residual-Armijo trial. Before solving, the native
runtime checks body-backed BOX endpoints against the supplied current body
pose and the captured prior projection; at post-commit it checks the realized
BOX projection again. Body-backed BOX bodies must be static or kinematic, have
zero linear/angular velocity and inverse mass/inertia, and preserve the same
projected transform across that transaction (with quaternion sign equivalence). `all` applies these checks at
every control step; an unset selector leaves the legacy path unchanged.

This is a fail-closed endpoint contract. If a matching selector names a moved
or otherwise invalid BOX endpoint, that environment is rejected and its FEM,
material/contact history, and projected BOX state are rolled back. The endpoint
certificate remains active even if the residual-Armijo trial later logs
`skipped` because its solver scope is unsupported or conflicts with another
Armijo mode. When the endpoint is valid but the trial is skipped, the step
continues without BOX residual Armijo. The native capsule-Armijo regression fixture uses `all` on a body-backed
kinematic support BOX, changes its transform both before solve and at
post-commit, and asserts rejection with exact accepted-state restoration while
the separated control environment succeeds. This is a bounded software
regression, not physical box qualification.

## Scored-tool trial and native invocation audit

The `scoring/armijo-001` 50 micrometre prescribed-punch trial used frozen
build 027. Its accepted indentation history recorded peak punch reaction
0.389654 N at step 47 with two punch contacts; these values remain native
diagnostic observations. The stationary-tool control reported no punch
contacts or force, but its endpoint result is `SCOPE_NOT_QUALIFIED` because the
same build has BOX dispatch selectors that bypass the stationary-endpoint
gate. Sixty steps were accepted through indentation and hold. The first
withdrawal attempt was rejected (status 10 after 512 FGMRES iterations, zero
completed microsteps), with reported base and trial residuals `6.8061954e-6`
and `6.8579789e-6`. The gap (`2.2695e-6 m`) and zero punch-pair fields on that
rejected row reflect the rolled-back accepted snapshot, not the proposed
withdrawal endpoint. They are not evidence of realized withdrawal contact or
separation. This run does not complete the prescribed path or qualify a crease;
self-contact was disabled.

The repaired frozen-028 `scoring/armijo-002` and `scoring/armijo-003` runs
compare two prescribed withdrawal schedules. They use the same runtime,
compiled-world fingerprint and sealed input hashes; every CSV cell across the
60 accepted loading/hold rows is identical. The 24-step withdrawal in 002
rejects step 60 after 60 accepted steps (status 10, 1,024 FGMRES iterations,
residual metric from `6.8061954e-6` to `6.8700779e-6`). With only the withdrawal
step count changed from 24 to 96, 003 completes 168/168 native steps: 48 load,
12 hold, 96 unload and 12 relaxation steps. It records a peak punch force of
`0.389654 N` at step 47 with two contacts, then ends with no punch contacts.
The longer schedule changes path duration at fixed `25 μs` timestep, so this
is prescribed-path sensitivity rather than temporal convergence. Self-contact
is disabled and the material law is not calibrated to an irreversible crease.

The `native-tests-008` capsule-box, grip and mutable-snapshot siblings remain
software-regression records. Their frozen-027 stationary-endpoint observation
is not moved-body endpoint proof under the BOX selector-gate bypass. Four 008
regional, reference, frame and cohesive invocations omitted required material
`.nmatter` arguments and exited after printing usage; these are
`UNEXECUTED_INVALID_INVOCATION`, not mechanics failures. In `native-tests-009`,
all five corrected capsule-box, regional, reference, frame and cohesive
invocations exited zero and all 360 sealed input hashes per invocation
verified. The capsule test exercises moved-body endpoint and rollback checks.
The cohesive invocation passed 4,004 native checks for a separate single-ply
bleached clay-coated paperboard Mode-I coupon; it does not calibrate corrugated
starch. The 009 records supersede the old endpoint-guard claim without erasing
the 008 records. Predictions, receipts, logs and invocation audit are retained
under `/Users/home/cardboard-followup-20261013/native-tests-008/` and
`/Users/home/cardboard-followup-20261013/native-tests-009/`.

## Full-box driver

The native driver compiles the complete FEFCO 0201 blank with twelve score
axes and fifteen assembly actions. It commands temporary boundary grips and
rigid tools; panel interiors remain FEM unknowns. Accepted geometry determines
the next fold command and the recorded angle. The current compiled default
blank contains 4,308 nodes and 10,356 tetrahedra.

The `box/native-003` full-blank attempt accepted 32 native steps, then rejected
step 32 during `score_die_contact_approach`: status 10 after 512 FGMRES
iterations, with residual `1.24495e-6` against a `1e-6` tolerance. Later fold
actions and flap closure were not attempted. This records an early native
contact-solver failure on the full mesh, not a completed score, fold, or
assembly. The prediction, receipt, result, and full action/step logs are in
`/Users/home/cardboard-followup-20261013/box/native-003/`.

The frozen-027 `box/native-004` instrumented residual-gap run ended naturally
with exit code 2 after 502.526 seconds; it was not interrupted. It accepted 32
steps, then rejected native step 32 during score-die contact approach with
status 10, 512 FGMRES iterations, and residual `1.63902e-5`. Later fold actions
and closure were not attempted. The all-selector BOX audit found dispatch paths
that bypass the stationary-endpoint gate, so this capture remains
`SCOPE_NOT_QUALIFIED` for endpoint-gap admission or box mechanics. Its failure
and logs are preserved as native diagnostic evidence, not a gate or physical
qualification. A corrected frozen-028 replay is separate and must be judged
from its own prediction and receipt.

The frozen-028 `box/native-005` replay includes default self-contact and a
1,024-iteration FGMRES budget. It assembled the same 4,308-node, 10,356-tet
blank, accepted 32 steps, then rejected native step 32 during
`score_die_contact_approach` (status 10, residual `1.25349e-6` against the
`1e-6` tolerance). No later fold action or closure was attempted. This corrected
run remains an early native contact-solver failure, not a completed score, fold
or assembly.

The initial die stroke reaches contact without a calibrated irreversible
crease-making law. The manufacturer's joint is an explicit calibration gate.
An optional external-fixture experiment holds the tab/seam and end-flap edges;
it is externally held, unbonded and unsealed. It cannot establish fixture-free
assembly or adhesive closure. Every rejected native attempt and unattempted
action remains in the ledgers. See the
[driver contract](../../tools/cardboard_box_native_driver.md).

The remaining qualification targets are complete accepted mesh/time histories,
a calibrated irreversible crease law with self-contact and validated
withdrawal, source-matched single-ply bulk damage in free-deformation FEM with
justified tetrahedral regularization, grade-matched adhesive debonding, and
physically solved folding and closure. The 50 μm prescribed punch path completes
with the longer withdrawal schedule, self-contact disabled and no calibrated
crease law. This schedule sensitivity does not establish time convergence. The
isolated `2e-6` local-projection experiment failed after two accepted steps and
does not change the production default. Manufactured coupling and software
fixtures do not close these physical targets.
