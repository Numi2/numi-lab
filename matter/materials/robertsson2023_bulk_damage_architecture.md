# Robertsson 2023 bulk paperboard damage: implementation boundary

**Historical baseline:** the assessment below records the implementation
boundary before the October 13 native follow-up. The current implementation
has ABI 41, 32-scalar capacity, a complete 27-state source-grade material
instrument, and a separate positive Mode I paired-face FEM path. The native
source-grade `0 -> 0.3 -> 0` shear cycle accepts 121 material points; it does
not qualify resolved scoring or corrugated glue debonding. See the
[current follow-up](../examples/cardboard/native-damage-and-assembly.md).
References to the "current" 16-scalar cap and absent face forces below describe
that historical baseline, not the updated checkout.

The source is a bulk damage model for a
particular single-ply paperboard, not a glue-interface law and not a parameter
set for the current corrugated recipe.

## What the source model represents

Robertsson et al. combine isotropic scalar damage with an anisotropic finite
strain elasto-plastic paperboard model. The total free energy and Kirchhoff
stress are reduced by the factor `1 - alpha`. Damage is driven by the
undamaged plastic dissipation in the four out-of-plane shear subsurfaces
`nu = 9..12`; the paper explicitly omits the other possible damage couplings
for this study. The damage threshold uses KKT conditions and accumulated
subsurface dissipation, and the paper caps Gauss-point damage at `alpha =
0.96`.

The constitutive update is staggered: first solve the Euler-backward
elasto-plastic equations for the plastic deformation gradient and 12
hardening variables, then update damage from the resulting out-of-plane shear
plastic dissipation. Damage is not a direct function of opening, crease angle,
or the cohesive opening used by the separate Tryding/Biel coupon.

The paper's damage calibration targets line-creasing and subsequent
line-folding measurements in machine-direction and cross-direction setups.
Its substrate is a single-ply, softwood-fibre paperboard, 0.39 mm thick and
788 kg/m³. The current corrugated geometry instead combines separately sourced
liners, flute medium, and assumed finite glue. Its paper assets
([liner](hajali2009_liner_hill_ideal.nmatter),
[medium](hajali2009_medium_hill_ideal.nmatter)) do not identify the
Robertsson calibration grade.

## State capacity is an ABI blocker

A direct scalar representation requires 22 persistent values:

| Source quantity | Scalar count |
| --- | ---: |
| Full plastic deformation gradient `Fp` | 9 |
| Subsurface hardening variables `kappa^(nu)`, `nu = 1..12` | 12 |
| Isotropic damage `alpha` | 1 |
| **Total** | **22** |

Matter currently declares `NM_MAX_MATERIAL_STATE = 16` in
[`shared.h`](../include/numi/matter/shared.h), and the compiler rejects a
material whose internal-state count exceeds that fixed GPU capacity in
[`constitutive_impl.cpp`](../src/constitutive_impl.cpp). A direct 22-value
asset therefore cannot compile. This work does not increase the global state
cap or pack away state under an undocumented constraint. A future native
implementation needs either a versioned, material-specific extended-state
path through compiler, package, GPU evaluation, accepted-state storage and
rollback, or a proven exact reduced state representation with equivalence
checks against the published equations.

The state extension must preserve the source's update order: candidate `Fp`
and all `kappa` values first, plastic shear dissipation next, `alpha` last.
The accepted state must remain unchanged on local or global rejection. The
algorithmic stress tangent must include the staggered damage contribution
without turning the four shear drivers into normal-opening damage.

## Mesh length is part of the calibrated law

The paper explicitly reports that local softening damage is mesh-dependent.
It uses reference projected area `A_r^e = 0.0011 mm²` in the critical crease
zone and scales both damage coefficients `beta_n` and `c_n` with the
undeformed element's material-direction projected area:

- For out-of-plane shear subsurfaces `9,10` (MD), scale with
  `A_md^e / A_r^e`.
- For subsurfaces `11,12` (CD), scale with `A_cd^e / A_r^e`.

For the paper's eight-node bricks, the projected areas are derived from the
six undeformed face areas and face normals. The reference discretization uses
approximately 0.07 mm in-plane by 0.0156 mm through-thickness near the crease.
The reported empirical scaling gives close response agreement across the
paper's tested mesh range, but the finer mesh still predicts more delamination;
it is not a proof of full mesh objectivity. A Matter implementation must bind
material axes, projected-area calculation, reference area, and the mesh
calibration to the exact discretization. Alternatively, a nonlocal or
regularized damage model would need its own characteristic-length
identification and validation; it would not be the unchanged published local
law.

## Current source-equivalence limits in the native authoring path

The material builder accepts machine- and cross-direction projected areas, but
its defaults are both `0.0011 mm²`, equal to the paper's reference area. The
current point and FEM fixtures call `makeMaterial()` with those defaults, so
the resulting two area ratios are `1.0`; the FEM path does not compute and pass
per-tetrahedron material-direction projected areas. The native implementation
therefore does not reproduce the paper's elementwise projected-area scaling
or its mesh calibration. The one-tetrahedron coupling check verifies
state/force parity for this authored program, not the paper's mesh-regularized
crease response.

The authored ZD compression energy also subtracts an additive constant so
`Psi(I)=0`. This preserves the fixed-damage stress because the deformation
derivative of a constant is zero, but it changes the absolute energy level and
can change any damage-conjugate term evaluated from the unshifted energy. The
printed equation and the source implementation's energy normalization have not
been reconciled; the current tests do not establish exact source-energy-level
reproduction. Treat the program as the documented stress/plasticity/damage
slice with this explicit normalization, rather than claiming identity of the
full source free energy.

The source lists reference damage coefficients `c_n^r = [8, 8, 9, 9]` and
`beta_n^r = [2.85, 2.85, 3.98, 3.98] MPa^-1` for subsurfaces 9–12. These values
remain source-specific to that paperboard, reference mesh and crease/fold
calibration. Copying only this table into the current corrugated board would
not transfer its behavior.

## Starch peel data do not identify a cohesive law

Iwahashi and Nagasawa tested three stacked 0.31 mm virgin-pulp, non-coated
paperboards with adhesive between the sheets. Their corn-starch condition
reports a peel line force of `0.22 ± 0.009 N/mm`; the stack had three sheets,
two glue lines, and a measured total thickness of `1.03 ± 0.015 mm`. The
specimens were conditioned at 293 K and 50% RH, with more than 48 hours after
gluing. Their two-dimensional punch test reports a whole-stack first peak of
`83.2 N/mm` for this condition. They also fit an across-condition correlation
between whole-stack punch peak and peel line force.

Those observations constrain a test fixture and bonded stack. They do not
uniquely determine a local normal or shear peak traction, a traction–separation
curve, mode-I or mode-II fracture energy, critical opening/sliding, mixed-mode
interaction, cured glue thickness/modulus, or a debonding evolution law. The
punch peak includes paperboard cutting, friction/sliding, layer deformation,
and adhesive effects together. It cannot be substituted for a cohesive
strength or `Gc` without additional geometry and constitutive measurements.

The finite elastic glue currently used in Matter resolves bridge geometry and
compliance; it has no debonding. The [Tryding/Biel material coupon](tryding2023_bleached_paperboard_modeI_coupon.md)
uses a different single-ply bleached, clay-coated paperboard and only a pure
Mode-I opening specialization. Neither source is a calibrated starch bond for
the corrugated recipe.

A further [bounded starch-interface literature search](starch_interface_literature.md)
found useful pin-adhesion and peel studies, but no complete local calibration
for this recipe. Those data can support a separately identified inverse
problem; they cannot supply missing cohesive parameters by direct substitution.

## Native cohesive faces are topology records, not a traction law

Matter has a useful topology foundation, but the current path does not apply
cohesive forces. When `MutationPolicySource::cohesiveFracture` is enabled, the
[compiler](../src/compiler_impl.cpp) enumerates every two-tetrahedron internal
face of that FEM object and creates an [`NMCohesiveFaceGPU`](../include/numi/matter/shared.h)
record. The record carries rest-face geometry and slots labelled damage,
maximum opening, traction, and fracture work. It has no authored interface
material/region selector, so this inventory cannot presently target only the
liner–glue and flute–glue boundaries while excluding other internal faces.
Faces between separate FEM objects are not generated by this per-object
adjacency pass.

The [`NM_MUTATION_COHESIVE_SEPARATION` branch](../src/metal/topology_mutation.metalinc)
handles a scheduled separation command: it duplicates the three face nodes,
marks the face separated, and sets `face.state.x` to one. It does not evaluate
opening, traction, fracture energy, or a failure criterion. The mutation adds
the rest-face area to `topology.accounting.w`; that quantity is area in this
path, not fracture energy. The shared-structure comment now names the area
correctly; the runtime accounting is unchanged.
I found no cohesive-face force, energy, tangent, or state-update path in
[`fem.metalinc`](../src/metal/fem.metalinc) or
[`contact.metalinc`](../src/metal/contact.metalinc). Likewise,
`MixedMaterialSource::cohesiveStrength` and `fractureEnergy` are packed into
`NMMixedMaterialGPU`, but are not consumed by those force kernels. The
[runtime](../src/runtime.mm) already mirrors cohesive-face buffers as accepted,
candidate, and checkpoint state; that supplies transaction plumbing, not the
missing constitutive update.

A real face law therefore needs an authored, source-bound interface assignment;
native opening/sliding potential, traction and tangent contributions in FEM
assembly; candidate face-state updates that commit only with an accepted global
step and roll back on rejection; and a criterion that issues separation only
after the source law reaches its stated failure condition. The existing
topology mutation can then perform node splitting, but it cannot substitute
for that constitutive work.

At the historical baseline recorded above, the nearest source-matched
interface route was an unimplemented pure-Mode-I law for the separate single-ply
bleached, clay-coated paperboard in the
[Tryding/Biel coupon](tryding2023_bleached_paperboard_modeI_coupon.md). The
October 13 follow-up has since connected that grade-specific law to an
explicitly authored paired-face FEM path, with native opening traction,
algorithmic/geometric tangent, reference-area work and transactional face
history. Its native tests do not transfer parameters to starch or the current
corrugated-board recipe; the corrugated glue remains elastic and tied.

## Evidence needed before implementation can support a creased board

1. Select a published board grade with source-reported MD, CD, out-of-plane
compression/tension and shear response, plus line creasing and folding data
for that same grade. Reuse published curves, supplements or author-released
data; missing inputs stay unidentified rather than being filled from a
different grade. This project does not require collecting new physical data.
2. Determine either a compatible 22-scalar state route or a source-equivalent
reduction without changing global state capacity. Verify the plastic
return-map and staggered damage update against an independent reference.
3. Record the exact mesh geometry and material orientation; compute the
source projected areas and preserve its reference-area calibration, or
independently identify a different regularization length.
4. Locate published cured-starch geometry and direct mode-I/mode-II interface
response for representative flute/liner bonds. A peel line force alone is
insufficient. Bind any available unload/reload, rate, moisture and mixed-mode
data to their reported specimens; unsupported responses remain uncalibrated.
5. Compare accepted native tool force–travel, residual crease geometry,
folding moment–angle, and mesh/time refinement against the matching published
experiments. Bulk paper damage and glue debonding remain distinct mechanisms.

Until those inputs and the state-ABI work exist, the evidence supports only
the explicit elastic finite-glue geometry and the separate material-point
coupons. It does not support bulk crease damage or starch debonding in the
corrugated board.

## October 13 follow-up evidence

The full Robertsson single-ply softwood material now has a 27-state native
point implementation. `source-point-003` accepted 119 material-point responses
through shear `gamma=0.59` and rejected `gamma=0.595`; that earlier instrument
zeroed the failed candidate and did not identify the terminal subtype.
`source-point-005` completed a separate `0 -> 0.3 -> 0` loading/unloading path
with 121 accepted material-point responses, damage `0.245009735` at the end,
and final `P13=-4.974485 MPa`. It did not test the earlier `0.595` failure.

`source-point-006` replayed the high-shear loading at `0.005` increments, with
no constitutive parameter or tolerance changes. It retained 119 accepted
responses through `0.59`, then rejected `0.595` as
`projection_iteration_exhausted` at the 16-iteration local projection limit.
The final residual infinity norm was `4.8307702e-5`, above the source-code
FP32 criterion `2e-5*(1+state_inf) = 4.3015134e-5`. The accepted history ended
at damage `0.34889245`; rejected candidate damage `0.35250464` was not
committed. Thus the local iteration limit is identified for this tested path;
it is not evidence of a physical failure threshold and is not a global FEM
failure. The captured point-test results and all 360 sealed input hashes are
at `/Users/home/cardboard-followup-20261013/bulk/source-point-006/`.

A seeded continuation imported the exact 27-scalar accepted state at `gamma=0.59`
from rejected row 119; FP32 values match the source row bit-for-bit. The
one-increment continuation reproduced the rejected `gamma=0.595` candidate
bit-for-bit, with the same `projection_iteration_exhausted` subtype, residual
`4.8307701945304871e-5`, state norm `1.1507568359375`, and 16 iterations.
Two- and four-subincrement continuations accepted to `0.595` and then returned
only to `0.59`. At `0.595`, the two partitions differed by at most
`3.5465e-6` in state 21, 7 Pa in `P13`, and `3.8743e-7` in damage; after the
short return to `0.59`, `P13` differed by 2.5 Pa. All 362 sealed inputs per
continuation verified. This resolves the specific one-increment local
projection failure for these tested subdivisions under the unchanged law, but
the nonzero partition difference is not a convergence result. The run does not
complete the high-shear unload to zero. Per-case results, source bindings, and
the aggregate are at
`/Users/home/cardboard-followup-20261013/bulk/source-point-seeded-{1,2,4}/`
and `bulk/seeded-sensitivity.json`.

A distinct `fem-coupling-001` experiment checks the integration path. It used
the same authored 27-state material in production FEM and the point evaluator
for one tetrahedron, with all four nodes prescribed to the homogeneous
`F13` path `x=X+gamma*Z`, `gamma: 0 -> 0.3 -> 0`, 60 increments per leg,
121 matched rows including the initial state, and a 0.005 s timestep. All FEM
steps returned status zero. Maximum absolute state difference was
`1.63912773e-7` at state 17, step 47; a relative error of 1.074 near zero was
not used alone, and maximum mixed-tolerance utilization was 0.0006781.
Maximum reaction `P13` difference was `5.31275666 Pa` at step 47, with
0.0004088 maximum mixed-tolerance utilization. Final damage matched at
`0.245009735`, and final reaction stresses were `-4.97448550 MPa` (FEM) and
`-4.97448500 MPa` (point path). A deliberately negative-J candidate was
rejected while the accepted FEM history and held control remained unchanged;
the captured fixture log does not print that failure's exact status code.

This is manufactured numerical coupling parity under a fully prescribed
kinematic path. It does not demonstrate a free-mesh solution, mesh convergence,
experimental validation, or an independent implementation of the constitutive
law. It also does not validate the tetrahedral adaptation of the paper's
projected-area damage regularization. The captured run and comparator are at
`/Users/home/cardboard-followup-20261013/bulk/fem-coupling-001/`.

## References

- K. Robertsson, E. Jacobsson, M. Wallin, E. Borgqvist, M. Ristinmaa, and
  J. Tryding, “A continuum damage model for creasing and folding of
  paperboard,” *Packaging Technology and Science* 36(12), 1037–1050 (2023),
  [DOI 10.1002/pts.2774](https://doi.org/10.1002/pts.2774),
  [publisher full text](https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774).
- T. Iwahashi and S. Nagasawa, “Breaking behavior of laminated-glued three
  pieces of paperboard subjected to a shearing of punching tools,”
  *Transactions on GIGAKU* 9(2), 09018/1–10 (2021),
  [full text](https://www.jstage.jst.go.jp/article/gigaku/9/2/9_09018-1/_pdf/-char/en).
