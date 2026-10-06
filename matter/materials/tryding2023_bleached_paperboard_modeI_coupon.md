# Tryding 2023 bleached-paperboard Mode I coupon

This material asset implements a narrow native constitutive coupon for the
positive-opening, pure Mode I specialization of the interface model in Tryding
et al. (2023). It is a source-bound development check for Matter's material
evaluator. It is not a calibrated starch adhesive, a corrugated-cardboard
liner/flute model, a zero-thickness cohesive-face law, or a qualified creasing
or folding model.

The paperboard sample in Tryding et al. is single-ply, bleached, and
clay-coated, with basis weight 319 g/m² and thickness 0.410 mm. The cohesive
parameters are attributed to tests from another production batch of the same
paperboard quality reported by Biel et al. They therefore describe that
paperboard quality, not the Nagasawa corrugated-board grades in the other
material assets here.

## Source parameters and interpretation

The normal cohesive envelope uses the source values `Tn = 0.397 MPa`,
`deltaN = 0.111 mm`, and `c = 0.863`. Its source form is

`t(delta) = Tn / (1 + (delta / (xc * deltaN))^(1/c))`,

where `xc` is derived from Eq. (24) and equals 0.7031799504329036. The
normal elastic unloading/reloading modulus is Eq. (43), with
`Ec = 10.9 MPa/mm` and `m = 1.39`. The source's scalar free-energy
specialization is `Psi = 0.5 * E(kappa) * (delta - deltaP)^2` per unit
interface area. The material maps `delta = h * (F33 - 1)` and converts the
surface energy to a volumetric energy density by multiplying by the finite
evaluation thickness's reciprocal `1/h = 2439.0243902439024 m^-1`
(`h = 0.410 mm`). The reciprocal is algebraically exact; storing it
as a parameter avoids an unsimplified `h^4` denominator in generated FP32
tangent bytecode, which falls below Matter's scalar division floor.

The source relates the normal damage variable to the uniaxial cohesive
traction by `kappa = 1 - t/Tn`. Combining that relation with the envelope gives
`deltaMax = deltaN * xc * (kappa/(1-kappa))^c`. The plastic opening `deltaP`
is derived so Eq. (2)'s energy gives the source traction at `deltaMax`. The
history update takes the maximum between prior damage and the damage implied
by current positive opening.

The residual uses a dimensionless Fischer–Burmeister smoothing value
`history_fb_epsilon = 1e-8` because the native symbolic tangent generator does
not differentiate the `max` primitive. The state initializer remains the exact
source `max` update. Matter's local state projection accepts residuals up to
`2e-5 * (1 + |state|)`, so at an exact history junction its residual
`sqrt(2) * epsilon = 1.41421356e-8` is accepted at the max seed, about 1,428
times below the tolerance. This is not a claim that the smoothed equation was
solved to `epsilon`; preserving the max seed prevents artificial damage creep
on repeated holds. Its source-state difference is zero in the tested scalar
FP32 bytecode path; the production Metal hold check is included in the native
coupon test.

At `kappa = 0.01`, solving the smoothed complementarity exactly would move the
state by at most `epsilon` at the junction. The independent source equations
map that increment to a maximum-opening shift of `1.28983e-9 mm`, a
`-2.84135 Pa` traction change at the old opening, and a
`-4.65106e-7 J/m²` stored-energy change. The native state projection accepts
the exact max seed instead, so these are quantified bounds for the exact
regularized root, not observed native state changes. At the nonsmooth junction
the source has different loading and unloading tangents. The generated
regularized tangent selects their midpoint; it is a numerical generalized
derivative, not a unique source tangent. CPU evaluation of the compiled FP32
bytecode measured `450340608 Pa/F33`, between the source one-sided values
`-1261948.21` and `901943314.97 Pa/F33`, with a `75.38 Pa/F33` midpoint error.
The native Metal evaluator measured the midpoint at a second representable
junction: `394364064 Pa/F33` versus the source midpoint
`394364055.2 Pa/F33`, between loading `-1278203.9` and unloading
`790006314.2 Pa/F33`.

The source states that its finite-element simulations were initialized with
damage `kappa = 0.01`. This asset retains that explicit value. The coupon test
sets its initial stretch to the matching positive opening, approximately
`0.00147964075578 mm`; it does not regularize the singular zero-damage limit
or claim a stress-free undeformed reference.

The paper's Table 1 also reports shear strength/length and friction inputs.
The scalar coupon does not use them. The source notes that its plastic-potential
friction value `muBar = 0.10` lacks experimental support; it is retained only
in the separate CPU reference's provenance record, not consumed here.

## Implemented and excluded behavior

The native evaluator check exercises the production material state projection,
first-Piola stress, and directional tangent on a positive-opening uniaxial
path. It checks the source normal envelope, irreversible scalar damage,
monotonic external work, the source elastic unload/reload branch, recoverable
energy from stress/tangent on the unloading branch, and a failed
nonpositive-opening, shear, or in-plane-strain evaluation that leaves the
accepted input state unchanged. The `valid` gate requires `F33 > 1`, other
diagonal entries within `2e-6` of one, and off-diagonal entries within `2e-6`
of zero; this tolerance is a numerical admission bound, not a material
parameter or physical shear allowance.
The native direct evaluator does not own or commit a Runtime state transition;
it is not evidence of transaction rollback in a complete FEM step.

This slice omits the source model's mixed-mode law, in-plane response,
compression/contact, plastic shear flow, hysteresis details, and its full
finite-element damage driver. The normal envelope approaches zero traction
asymptotically and this asset supplies no finite complete-separation criterion
or cohesive-face topology. The material's kinematic validity gate intentionally
prevents use as a general bulk paperboard material. It does not identify any
adhesive strength, fracture energy, or debonding law for corrugated-board
starch glue.

The tangent comparison excludes neither loading nor unloading away from the
junction; only the exact max-history junction uses the midpoint convention
above. Tests compare a coarse and refined native opening path for work
convergence, repeated constant-opening holds for zero history creep, and the
load/unload/reload branch against the independent source oracle.

## Native evaluator result

The 2026-10-06 native Metal coupon check passed on Apple M4 with material
fingerprint `5570626462714653447`. It evaluated the production constitutive
program and tangent on a one-tetra instrument fixture; it advanced zero FEM
steps. It is not a simulated or experimentally qualified physical specimen.
The native 512-step opening path produced `26.10949601 J/m²` against
`26.10950156 J/m²` from the source oracle. The work error decreased from
`1.4711e-3 J/m²` at 32 steps to `5.55e-6 J/m²` at 512 steps. The reported
total includes the source-oracle work accumulated before the positive opening
seed; native stress integration covers only the seed-to-0.1 mm interval. Both
path resolutions use the same source seed-work offset. Maximum loading
tangent relative error was `2.08e-6`; maximum centered stress-FD error was
`1.1234%` under a 4% check limit. One thousand native constant-opening holds
at a representable history junction changed damage by exactly zero.

The positive seed stretch is quantized to FP32. Its native seed traction was
`392978.53 Pa`, `51.47 Pa` below the exact source seed traction of
`393030 Pa`; that difference remains reported rather than absorbed into a
fit. The unload check preserved damage and passed its source free-energy and
dissipation checks. The complete CTest output, executable/metallib/source
hashes, and load/unload/reload/new-maximum source-oracle plot are preserved in
[`delamination-native-coupon-002`](/Users/home/cardboard-closure-evidence-20261006/delamination-native-coupon-002/receipt.md).

The [Robertsson bulk-damage assessment](robertsson2023_bulk_damage_architecture.md)
describes a separate constitutive path and its current 22-versus-16 state
capacity blocker. Neither that damage model nor this bleached-paperboard
interface coupon is a calibrated starch-adhesive law for the corrugated board.

## References

- J. Tryding, M. Ristinmaa, and collaborators, “Delamination of cellulose-based
  materials during loading–unloading conditions: Interface model and
  experimental observations,” *International Journal of Solids and Structures*
  279 (2023), 112365. DOI: [10.1016/j.ijsolstr.2023.112365](https://doi.org/10.1016/j.ijsolstr.2023.112365).
  [Open full text](https://www.diva-portal.org/smash/get/diva2%3A1774521/FULLTEXT01.pdf).
- S. Biel et al., “Experimental evaluation of normal and shear delamination in
  cellulose-based materials using a cohesive zone model,” *International
  Journal of Solids and Structures* 252 (2022), 111755. DOI:
  [10.1016/j.ijsolstr.2022.111755](https://doi.org/10.1016/j.ijsolstr.2022.111755).
  [Lund University record](https://lup.lub.lu.se/search/publication/db96df6b-ed7f-4c7c-bc3e-8a9a68b67a2a).
