# Robertsson et al. (2023), Equation 18 source extract

Source: Robertsson, K. et al., “A continuum damage model for creasing and
folding of paperboard,” *Packaging Technology and Science* 36(12), 1037–1050
(2023), DOI [10.1002/pts.2774](https://doi.org/10.1002/pts.2774).
The official Wiley article page labels the article Open Access and exposes the
full HTML text: <https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774>.
Equation 18 is in Section 2.3 (publisher HTML equation node
`pts2774-disp-0018`; accessed 2026-10-14). The equation text below is a
transcription of that primary source, not an independently authored
constitutive law:

```text
Psi_0^e = E1 (I11 + I11^-1 - 2)
        + E2 (I12 + I12^-1 - 2)
        + E4 (I11 + I12 + I23^-1 - 3)
        + H E3 (I13 + I13^-1 - 2)
        + (1-H) E6 [ I13^2 + (1/E7) exp(-E7 (I13^2 - 1)) - 2 ]
        + E5 (I11 I12 I13 - Je).
```

The same article states immediately after its damage evolution equations that
`Psi_0 >= 0` (Eq. 17 discussion). With the printed compression branch and
`E6 = 0.365 MPa`, `E7 = 11.5` (Table C1), the one-sided value of the ZD
compression term at `I13 -> 1-` is
`E6 (1/E7 - 1) = -0.3332608695652174 MPa`, whereas the tension-side value at
`I13 = 1` is zero because `H=1` there. The printed branch therefore has a
constant offset and a discontinuity at the branch switch; the nonnegative
energy statement and printed formula do not agree on this branch.

The constitutive reference is the undeformed state `F=I`, `Fp=I`, with all
hardening variables `kappa=0` and `alpha=0`. At that state
`I11=I12=I13=I23=Je=1`; the Heaviside convention selects `H=1`, and the
elastic and plastic parts of `Psi_0` are zero. This reference-state check does
not expose the compressed-side discrepancy because the compression branch is
inactive at equality.

The native builder now exposes two explicit conventions. `paper_absolute`
uses the literal printed `-2`, whose compressed-side limit is
`-0.3332608695652174 MPa`. `recentered` uses `-1 - 1/E7`, adding
`E6 (1 - 1/E7) = +0.3332608695652174 MPa` to the compression branch so that
its limit at `I13 -> 1-` is zero. There is no implicit convention default:
the own-grade verification selects `paper_absolute`, while the historical
regression uses the named `makeLegacyRecenteredMaterial()` entry point. The
material identity includes the selected convention.

The primary article prints Eq. 18 and reports that its simulations used
commercial code; no author implementation source or alternate executable
constant was identified in the available primary-source materials. Therefore
`recentered` is an explicitly named comparison/legacy regression convention,
not a verified source-author implementation. The primary evidence supports
the literal printed expression only; its inconsistency with the stated
nonnegative-energy condition remains unresolved.

Both choices have the same deformation derivative within the compression
branch, but they do not have the same absolute energy. The builder forms the
full selected `Psi_0` (elastic plus plastic contribution) and multiplies it by
`(1-alpha)` in the energy root. Therefore the damage-conjugate force
`A = -dPsi/dalpha = Psi_0` uses the selected convention; stress parity does
not make the two energy conventions interchangeable. The source extract does
not decide whether Eq. 18 contains a publication typo, so source runs must
identify which convention they use.

## Source-reported plastic strength and hardening table

The official Wiley full text presents Table C2 with the initial yield
distances `K0^(nu)` (MPa), hardening amplitudes `a_nu` (MPa), and hardening
rates `b_nu` (dimensionless). Equation 27 defines
`K^(nu)=a_nu*ln(b_nu*kappa^(nu)+1)` for `nu={1,2,3,6}`,
`K^(7)=a_7*kappa^(7)`, and zero hardening for the remaining subsurfaces.
The table's em dashes mean “no hardening parameter reported” for those
subsurfaces; the implementation encodes them as zero amplitude/rate only to
evaluate this published equation. They are not defaults for the Kraft/NSSC
recipe.

| nu | K0 MPa | a MPa | b | source location |
|---:|---:|---:|---:|---|
| 1 | 20.4 | 12.2 | 690 | Table C2, Eq. 27 |
| 2 | 11.7 | 5.10 | 435 | Table C2, Eq. 27 |
| 3 | 9.91 | 6.63 | 334 | Table C2, Eq. 27 |
| 4 | 22.0 | — | — | Table C2, Eq. 27 |
| 5 | 16.7 | — | — | Table C2, Eq. 27 |
| 6 | 9.91 | 6.63 | 334 | Table C2, Eq. 27 |
| 7 | 1.00 | 49.0 | — | Table C2, Eq. 27; linear branch |
| 8 | 0.40 | — | — | Table C2, Eq. 27 |
| 9 | 2.51 | — | — | Table C2, Eq. 27 |
| 10 | 2.51 | — | — | Table C2, Eq. 27 |
| 11 | 2.51 | — | — | Table C2, Eq. 27 |
| 12 | 2.51 | — | — | Table C2, Eq. 27 |

These parameters describe the separate single-ply softwood paperboard in the
article. They do not identify an equivalent 12-subsurface set for the current
205 gsm kraft liner or 126 gsm NSSC medium. The Haj-Ali fitted MD/CD tension
lines provide only four uniaxial bilinear slopes, not a mapping to all
multiaxial `K0`, `a`, and `b` values.

The article's Figure 4 is a plotted experimental folding response for the
separate single-ply softwood-paperboard grade. No raw curve table is included
in the accessible article text. The official Wiley HTML exposes a PowerPoint
download link for Figure 4, but the figure fetch returned HTTP 400 with the
unsupported `application/vnd.ms-powerpoint` content type in the web reader. A
browser-rendered article capture attempt returned HTTP 403, and the in-app
browser blocked the direct figure-download endpoint. No source pixels were
captured; no values are inferred here. Figure 4 remains eligible for a future
digitization only if its pixels can be obtained from an accessible official
publisher asset and archived with a source hash and visual uncertainty.

## Provenance and integrity

- Primary source URL: <https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774>
- DOI: `10.1002/pts.2774`
- Source location: Eq. 18, Section 2.3; Eqs. 7, 27; Tables C1–C3
- Reference state: `F=Fp=I`, `kappa=0`, `alpha=0`; `Psi_0=0`
- Material scope: single-ply softwood paperboard, not corrugated kraft/NSSC
- Source access: official open-access publisher HTML, checked 2026-10-14
- Figure asset attempt: <https://onlinelibrary.wiley.com/action/downloadFigures?doi=10.1002%2Fpts.2774&id=pts2774-fig-0004&partId=>; web reader rejected the response MIME type and headed browser access was blocked.
- No publisher pixels or PDF were archived. The SHA below binds this equation transcription only, not publisher HTML, rendered pixels, or PDF.
