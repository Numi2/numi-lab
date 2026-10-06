# Corrugated-board creasing source data

This directory records published inputs for an executable Matter instrument. It contains no new measurements, simulations, or model qualification. The preferred geometry/process source is the 2013 A-flute paper; the 2024 C-flute paper provides an independent homogenized-board comparison. Keep the two board grades separate.

## Preferred baseline: Nagasawa et al. (2013), A-flute rotary indentation

[Article DOI](https://doi.org/10.1299/jamdsm.7.103) · [Open full-text PDF](https://www.jstage.jst.go.jp/article/jamdsm/7/2/7_103/_pdf/-char/en) · [Local research copy outside the repository](/Users/home/cardboard-research-20261006/nagasawa-2013-rotary-creasing.pdf)

The board is JIS-Z1516 A-flute, nominal caliper 5.0 mm, pitch `lambda = 9.0 mm`, with two 0.25 mm liners (210 g/m² each) and 0.25 mm corrugated medium (180 g/m²). The medium cross-section is specified by radius 1.7 mm and inclination 64.8° (printed pp.104, 107; Fig. 8). The paper reports component-paper grammages, not finished-board grammage; flute take-up and adhesive mass are not supplied.

The model coupon is 20 mm wide by 36 mm long (`4 lambda`) and uses 3D eight-node solid-shell elements. Its crease tool has 140 mm working diameter, 120° bite, and 3.2 mm bite width; the flat anvil is 8 mm wide and 7 mm thick. The experimental anvil was urethane rubber, 90° hardness on JIS K 6253. The modeled main-creaser gap is 1.5 mm (`cC/t = 0.3`), with the bite centered over the flute valley (`X/lambda = 0`). The article reports fixed lateral liner edges as the default and also analyzes free edges. For the plotted frictionless case, tool-sheet friction is `mu_CL = 0`; the sheet is advanced with an imposed feed constraint because frictionless rolls do not entrain it. These inputs are in Section 3, printed pp.107–108, and Section 4.1, printed p.109.

The recommended first input set is source model **(a)**: both liners and corrugated medium use the orthotropic elastic constants in Table 3 (printed p.105). Axes are M = machine, C = cross-machine, and T = thickness:

| Material | `E_M`, `E_C`, `E_T` (MPa) | `nu_MT`, `nu_TC`, `nu_CM` | `G_MT`, `G_TC`, `G_CM` (MPa) |
|---|---:|---:|---:|
| Liner | 5020, 1884, 20 | 0.20, 0.01, 0.20 | 20, 20, 1175 |
| Medium | 4727, 2046, 16 | 0.20, 0.01, 0.20 | 16, 16, 1250 |

The paper prints three Poisson ratios and does not list their reciprocal values separately. Its raw-sheet test tables (Table 1, p.105) report liner tensile strength 44.6 MPa in MD and 21.2 MPa in CD, with breaking true strains 0.014 and 0.048; medium strengths are 38.7 MPa in MD and 21.5 MPa in CD. Table 1 prints the MD/CD moduli as 5.02 and 1.88 under an MPa heading, while the FE constants in Table 3 are 5020 and 1884 MPa. Treat the raw-table modulus scale as internally inconsistent; use Table 3 for the stated FE baseline. Through-thickness compression tests at 6.7×10⁻² s⁻¹ give 20.6 MPa for liner and 16.0 MPa for medium at true strain −0.3% (Table 2, p.105).

For this baseline, tied interfaces are an explicit implementation assumption. Nagasawa et al. do not report adhesive geometry, glue properties, or an interface law. This assumption must remain visible in model identity and any report. The source material is not itself a bond calibration.

The physical post-crease shape is the most direct public observable: for samples scored at `cC/t = 0.3`, the crushed flute width is 5–7 mm, the total inner-liner tangent angle is 124–132°, and the deflected zone spans 3–3.3 pitches (27–29.7 mm), Section 2.3 and Fig. 7 (p.107). At the model comparison state `s/t = −1.6`, option (a) gives 5.0 mm width and 163° angle; the angle therefore misses the measured interval substantially. The paper's option (c), which keeps elastic orthotropic liners but substitutes an RCT-equivalent isotropic elastoplastic medium, gives width 6.6 mm, angle 146°, and span 3 pitches; this is closer, but still not within the observed angle range (Section 4.1, Fig. 15, p.110). Here `s/t` is the source's moving-sheet state coordinate, not a physical crease-depth setting.

Keep option (c) as a separately named alternate: Table 4 (p.108) gives RCT-equivalent liner `E = 372 MPa, sigma_y = 7.7 MPa` and medium `E = 189 MPa, sigma_y = 4.3 MPa`; option (c) uses the latter for the medium. The paper does not supply enough of the RCT plastic hardening law for strict reproduction. Because the present geometry explicitly resolves flute buckling, substituting an RCT-effective medium can count that crush compliance twice. Do not mix the alternate with the default or silently use it as a material property of the paper sheet.

Option (a) is an elastic instrument baseline, not a validated crushing/folding model. The authors state that it has no fracture; its liner strain at the comparison state is 0.036, while the raw liner MD breaking strain is 0.014 (printed p.110). The source gives qualitative liner-failure locations but no measured stress field or quantitative failure threshold. Do not use its deep-crease result as large-deformation validation.

## Independent cross-check: Aduke, Venter & Coetzee (2024), C-flute board

[Open article](https://www.mdpi.com/2297-8747/29/4/70) · [Open PDF](https://res.mdpi.com/d_attachment/mca/mca-29-00070/article_deploy/mca-29-00070.pdf) · [Local research copy outside the repository](/Users/home/cardboard-research-20261006/aduke-venter-coetzee-2024.pdf)

This source uses `250KL/150SC/250KL`, not the A-flute material above: 250 g/m² virgin-kraft liner sheets and 150 g/m² semi-chemical fluting stock (Section 2.2, p.3). Those are flat-paper component grammages; the finished-board grammage is not stated and must not be set to 650 g/m² because flute take-up and adhesive contribution are absent.

Figure 6 (p.6) labels a 7.7 mm pitch, 0.251 mm flute sheet, 0.345 mm liner thicknesses, and a 3.861 mm distance between the inner liner faces. Interpreting that drawing geometrically gives a core midsurface peak-to-trough of 3.610 mm and centered sine amplitude 1.805 mm; total caliper is 4.551 mm. These are figure-derived interpretations, not separately tabulated centerline coordinates. Their homogenized model uses a different effective thickness, 4.4 mm (Table 1, p.7).

Structural-model elastic constants from Table 1 (p.7), in the paper's axes 1 = MD, 2 = CD, 3 = ZD:

| Layer | `E1`, `E2`, `E3` (MPa) | `nu12`, `nu23`, `nu13` | `G12`, `G13`, `G23` (MPa) |
|---|---:|---:|---:|
| Liner 250KL | 6695, 2310, 35 | 0.50, 0.01, 0.01 | 1522, 122, 66 |
| Fluting 150SC | 4709, 2918, 42 | 0.37, 0.01, 0.01 | 1435, 86, 83 |

`E1` and `E2` are from tensile tests; the other elastic constants are estimated using published relations. Table 3 (p.10) gives raw-sheet yield stresses in 0°/45°/90°: liner 26.11/17.44/17.44 MPa (average 19.61), fluting 32.65/14.02/14.02 MPa (average 18.68). The paper defines a Hill model but does not restate all measured strain ratios needed to reconstruct every Hill coefficient; a strict structural Hill-plastic reproduction therefore needs the referenced source data. The layers share nodes at flute contact locations, which is a perfect-tie idealization; the authors note that this rigid connection can overpredict stiffness compared with flexible starch bonds (p.15).

For a quantitative strip check, the creased four-point-bending specimen is 400×100 mm with its crease centered and along CD (Fig. 3, p.4; ten samples). The paper states bottom supports 200 mm apart, top loading anvils 340 mm apart, and support-to-loader distance `a = 70 mm`; displacement rate is 12.7 mm/min and the center deflection is measured. Equation (1), p.5, defines `Sb = (1/16)(P/Y)(L^3/b)(a/L)`, with `P` peak/failure load, `Y` maximum center deflection, `L = 200 mm`, `b = 100 mm`, and `a = 70 mm`. The stated pair labels/free-span wording should be checked against Fig. 4 before reproducing the fixture; do not swap the pairs by convention.

Table 5 (p.15) prints `N/m`, although the equation has dimensions of force×length (`N·m` in SI). Preserve the reported values and label, while flagging the dimensional inconsistency: MD experiment/structural FE/homogenized FE = 20.44/21.20/18.10; CD = 6.88/6.58/6.21. These are peak-load-over-maximum-deflection stiffness indices, not initial tangent stiffness. The raw force/displacement files are available from the authors on request; the article provides the table and plotted curves only.

The article's crease model is not a crease-forming simulation. It represents a 4 mm strip with reduced properties. For that strip Table 4 (p.15) gives vertical-crease liner `E1/E2/G12 = 3390/983/764 MPa` and homogenized core `51/116/11 MPa`; horizontal-crease liner `3985/1363/873 MPa` and core `60/140/12 MPa`. The paper reports 63% lower bending stiffness for a creased sample (Fig. 15, p.16). Its box simulation predicts 5.10 kN versus 5.09 kN experimental failure load after adding a 4 mm first-mode imperfection, but underpredicts in-plane displacement by 43%; it compares out-of-plane displacement with DIC within 10% (pp.20–21). This is a useful downstream comparison, not proof that the crease process or general constitutive response is reproduced.

## Scope boundary: Beex & Peerlings (2009)

[ORBilu record](https://orbilu.uni.lu/handle/10993/17431?locale=en) · [Open postprint PDF](https://publications.uni.lu/bitstream/10993/17431/1/BeexPeerlings2009.pdf)

This is a solid laminated paperboard creasing/folding study with cohesive delamination and plasticity. It is useful for mechanisms and for distinguishing crease damage in a laminated solid sheet, but it has no corrugated medium/flute geometry. Its geometry, material constants, and fold-moment targets are not corrugated-board inputs and should not be mixed into either source above.

## Evidence boundary

The first Matter deliverable should be an instrument that binds source inputs and reports observables against the matching published target. The explicit A-flute elastic case is only a baseline for numerical execution and source-output comparison. Its published physical-shape mismatch and absent fracture/bond data leave constitutive development open. Keep simulation outputs, published numerical results, and physical experimental observables distinct; no result from this literature pass is new physics evidence.
