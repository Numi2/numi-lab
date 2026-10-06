# Robertsson 2023 bulk paperboard damage: implementation boundary

**Status:** source-audited architecture assessment only. No Robertsson damage
law is implemented or calibrated in Matter. This is a bulk damage model for a
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

The nearest currently implementable source-matched interface slice is a
**separate, unimplemented** pure-Mode-I face law for the single-ply bleached,
clay-coated paperboard in the
[Tryding/Biel coupon](tryding2023_bleached_paperboard_modeI_coupon.md). Its
native material-evaluator coupon verifies the constitutive response but ran no
physical FEM steps and is not yet connected to cohesive faces. That route could
advance a paperboard delamination benchmark using its own reported normal
parameters and loading/unloading path. It does not identify the starch bond or
transfer to the current corrugated-board recipe.

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
