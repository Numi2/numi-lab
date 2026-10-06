# Corrugated-board manufacturing and material sources

This note records a source-bound C-flute paperboard model recipe for the native Matter benchmark. The most complete nonlinear paper-property source is Haj-Ali et al. (2009). Its paper grade is identified by nominal component basis weights, but the paper-sheet calipers and densities are absent. Popil et al. (2007) supplies an adhesive-modulus assumption and a documented double-backer pilot procedure; it does not supply a starch cohesive or fracture law. Popil's 2017 book supplies a measured finished-board benchmark for a commercial board using the same nominal liner/medium weights, but its coupon/material records are not proved to be the same rolls as the 2009 paper-property measurements.

These sources therefore support an explicitly **literature-based starting model**, not an exact reconstruction of one manufactured board lot. The implemented `literature2009` starter intentionally combines Haj-Ali's constitutive/yield data with Popil's measured commercial-board caliper and separate paper-coupon calipers/densities, plus the selected Table 8 glue dimensions. Those paper coupons are not identified as the commercial board's exact rolls. Source values, derived quantities, and numerical assumptions are kept distinct below and in [`manufacturing-source-manifest.json`](manufacturing-source-manifest.json).

## Primary sources and where their evidence applies

1. **Haj-Ali, R., Choi, J., Wei, B.-S., Popil, R., and Schaepe, M. (2009), “Refined nonlinear finite element models for corrugated fiberboards,” Composite Structures 87(4), 321–333.** [DOI / publisher record](https://doi.org/10.1016/j.compstruct.2008.02.001); [author-posted full text](https://www.researchgate.net/publication/223896964_Refined_nonlinear_finite_element_models_for_corrugated_fiberboard). Source for 205 g/m² liner and 126 g/m² medium, orthotropic elastic constants, measured in-plane tension-yield points, assumed compression/out-of-plane/shear yield information, C-flute profile, and parametric explicit glue-solid geometry. Tables 1–5 are on printed pp. 324–326; the glue study and Table 8 are on printed pp. 331–332. This is the central paper constitutive source.

2. **Popil, R. E., Schaepe, M. K., Haj-Ali, R., Wei, B.-S., and Choi, J. (2007), “Adhesive level effect on corrugated board strength – experiment and FE modeling,” International Progress in Paper Physics Seminar report, 6 pp.** [Public PDF](https://research.gatech.edu/sites/default/files/rbi/pdfs/Adhesive%20level%20effect%20on%20corrugated%20board%20strength%20PPP%20Miami%20U%20final%20ver%203.pdf). The file identifies itself as an abstract submitted to the seminar. It reports dried-starch elastic modulus and a lab double-backer simulator, but is not a complete cure, moisture, bond-fracture, or corrugator operating recipe. Relevant printed pp. 3–6.

3. **Popil, R. E. (2017), *The Physical Testing of Paper*, first published 2017, printed pp. 108–109, 120–123.** [Georgia Tech repository PDF](https://repository.gatech.edu/bitstreams/e4749f61-7780-41d3-93c7-d51cf5127e3e/download). This book reports a commercial WC 4226C 42 board with nominal 205 g/m² liner and 126–127 g/m² medium, finished-board observations, and methods. The separate soft-caliper paper coupons in its Table 4.1 are not identified as the same rolls as the WC 4226C board.

The two local PDFs retained for audit are outside the repository, under `/Users/home/cardboard-research-20261006/`. Their SHA-256 values and retrieval URLs are in the manifest. The 2009 article was consulted through its author-posted full text and no local article PDF is represented as retained.

## Paper material model from Haj-Ali et al. (2009)

The nominal board constituents are **205 g/m² unbleached southern softwood kraft liner** and **126 g/m² NSSC medium** (printed p. 323, §2). Their paper axes are 1 = MD, 2 = CD, and 3 = ZD (the article's paper-axis sketch and table conventions). Table 2, printed p. 325, gives the following orthotropic elastic constants:

| Layer | E11 MD (MPa) | E22 CD (MPa) | E33 ZD (MPa) | G12 (MPa) | G13 (MPa) | G23 (MPa) | ν12 | ν13 | ν23 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Liner | 4775 | 2088 | 19.1 | 1222 | 166.28 | 137.28 | 0.18 | 0.01 | 0.01 |
| Medium | 4472 | 1612 | 17.9 | 1039 | 226.34 | 198.72 | 0.18 | 0.01 | 0.01 |

The paper states that in-plane properties were measured. The article also says `G12 = 0.387 √(E11 E22)` (Eq. 1) and estimates `E33 = E11/250`. Thus those two values are source-reported correlations/estimates, not direct measurements. Treat the small out-of-plane shear moduli and ZD Poisson ratios as model inputs whose measurement basis is not documented as a direct measurement; do not describe the entire Table 2 as experimentally measured. The article's Poisson-ratio reciprocity convention is not fully documented in a solver-ready stiffness matrix; use a reciprocal orthotropic compliance/stiffness construction that enforces symmetry, and record which of the listed directional ratios are treated as independent.

Table 1, printed p. 324, reports in-plane tension yield stress and strain:

| Layer | MD yield (MPa) | MD strain | CD yield (MPa) | CD strain |
|---|---:|---:|---:|---:|
| Liner | 38.2 | 0.0080 | 18.4 | 0.0090 |
| Medium | 15.7 | 0.0035 | 8.06 | 0.0050 |

The paper gives bilinear elastic–plastic stress–strain curves in Figs. 1–2. It does **not** tabulate post-yield slopes, so a numerical hardening modulus cannot be attributed to an exact table value. If a simulation requires one, digitize the relevant plotted curve and record the plot/page, digitizing points, fit interval, and resulting slope separately from the published tabular values. The tension-yield pairs above do not by themselves determine a hardening law.

The paper's compression-yield values are assumptions: it sets compression yield stress and strain to 60% of the corresponding tension values (p. 323; Table 1 p. 324). This gives liner MD 22.92 MPa at 0.0048, liner CD 11.02 MPa at 0.0054, medium MD 9.42 MPa at 0.0021, and medium CD 4.836 MPa at 0.0030. Table 3 instead prints 9.391 MPa for medium MD and 4.836 MPa for medium CD; use the explicit Table 3 value when reproducing its Hill surface and retain the small source-rounding mismatch as such. Compression values were not presented as direct compression measurements.

The yield surface is the six-parameter Hill quadratic (Eq. 2, pp. 324–325), with its coefficients computed from directional yield stresses using Eq. 3. Table 3 p. 325 uses liner/medium ZD compression yield stresses 7.52/3.23 MPa and ZD tension values 12.55/5.38 MPa; these are estimated by the cited Beldie relation, not directly tested values here. The shear yields used are τ12 = 2.1 MPa and τ13 = τ23 = 0.024 MPa (Eq. 4 and text p. 325); the article does not report these as measured for these sheets. A reproducible implementation should preserve these epistemic labels and compute F,G,H,L,M,N from the source yield stresses and the exact Hill convention, rather than silently substituting an isotropic yield stress.

The article also reports Tsai–Wu ultimate failure limits in Table 4 p. 325, but does not provide a damage evolution/softening law. These strength limits can flag initiation or a failure observable; they do not justify progressive paper fracture, crease cracking, or energy dissipation after failure. The Hill model and its ECT comparisons are the source-supported irreversible constitutive component. They do not validate crease formation.

## Flute profile and explicit glue evidence

Haj-Ali et al. Table 5 p. 326 reports typical C-flute pitch 7.1–8.3 mm and height 3.61 mm; the reported measured specimen pitch is 7.9 mm. Their FE idealization uses a sinusoid with pitch 8.50 mm and height 3.50 mm. The implemented starter below selects the **measured 7.9 mm pitch**, not the paper's 8.50 mm FE pitch, and derives a centerline wave height to fit Popil's finished-board caliper and the selected separate paper/adhesive thicknesses. That derived height is an implementation closure, not a published flute-height measurement. The paper does not state whether its tabulated flute height refers to centerline amplitude or outer-face-to-outer-face clearance; its reported 3.61 mm and FE 3.50 mm should therefore remain separately labeled.

The explicit glue-solid study (Haj-Ali et al., §5, pp. 331–332) reports a homogeneous isotropic glue material with **E = 400 MPa and ν = 0.3** and uses 3-D continuum glue solids. It does not report a cured glue density, moisture, cure kinetics, adhesive strength, traction–separation law, cohesive fracture energy, or debonding calibration. Table 8 p. 332 lists parametric rectangular glue zones, not a measured population of bondline sections:

| Source zone | Double-back width × thickness (mm) | Double-back area (mm²) | Single-face width × thickness (mm) | Single-face area (mm²) |
|---|---:|---:|---:|---:|
| Type I | 1.50 × 0.20 | 0.300 | 1.50 × 0.10 | 0.150 |
| Type II | 0.80 × 0.10 | 0.080 | 0.60 × 0.03 | 0.018 |
| Type III | 0.30 × 0.10 | 0.030 | 1.00 × 0.03 | 0.030 |

These are the closest source values for finite glue geometry. The implemented starter selects the **Type II** sizes: double-back glue width 0.800 mm and gap/thickness 0.100 mm; single-face glue width 0.600 mm and gap/thickness 0.030 mm. Its resolved curved bridges follow the paper surfaces; this is a geometry adaptation of the source's rectangular parametric zones, not the published geometry or an experimentally imaged bondline. Record the selected source width/thickness and the adapted bridge volume in implementation evidence.

The 400 MPa/0.3 glue values recur in the 2007 Popil et al. report (p. 3): the authors say they measured adhesive-film stress–strain slopes over 0.1–0.3% strain and use 400 MPa for fully dried starch; they **assume** ν = 0.3 for homogeneous adhesive in FE. The report does not identify a glue density. A density of 1500 kg/m³ is therefore a solver-side numerical assumption, not a published corrugating-starch value. It may have negligible effect in a quasi-static, no-inertia analysis, but it matters if density couples into transient/inertial response and must not be presented as calibrated.

## Manufacturing sequence and observables

Popil et al. (2007), printed pp. 5–6, describe an IPST double-backer simulator: preheat the liner and medium, meter adhesive from a heated gravure roll using a doctor gap, combine the single-face board with the liner, and pause over a pressure-actuated hotplate for a prescribed time. The adhesive bath is stated to be 37.8 °C; doctor gaps of 0.1, 0.2, and 0.4 mm led to applied adhesive levels of 14.6, 19.5, and 36.6 g/m², inferred from sample/component mass differences. The report lists high-shear starch, 60%-solids PVA, 40%-solids sodium silicate, 38%-solids acrylic dispersion, and modified starch formulations. It does not say which dry solids fraction applies to each listed mass result, and does not give a cured local bondline thickness, line speed, corrugating-roll temperature, hotplate pressure, hotplate dwell time, or moisture endpoint. In particular, the **doctor gap is a process setting**, not the cured glue thickness in the finite element model.

Popil (2017), printed p. 123, separately describes IPST pilot corrugating practice: a pilot single-facer uses conventional operating parameters; for the book's handsheet investigations, manually double-backing a single-face web used metered rolling-nip application of Stein–Hall starch to flute tips followed by a hotplate press to attach the liner. The phrase “conventional operating parameters” does not give numeric roll temperatures, speeds, nip pressure, dwell, or adhesive solids. It documents a process sequence, not a reproducible industrial corrugator settings card.

For measured structure-level comparisons, Popil (2017) §5.5.1 pp. 108–109 says commercial WC 4226C 42 C-flute board used nominal 205 g/m² liner and 126–127 g/m² medium; ECT tests were run at 12.5 mm/min, with resin-embedded test ends or a TAPPI T 839 clamp arrangement. At specimen heights up to 80 mm, resin-embedded tests were reported as approximately height-independent; the text states that 60 mm or less avoids beam-buckling influence for the tested board when resin-embedded or clamped. This is an ECT method, not a bending or creasing validation.

Table 5.4 printed p. 120 reports for commercial WC 4226C 42: finished board basis weight 605 g/m², board caliper 4,210 μm (= 4.210 mm), liner CD SCT 3.4 kN/m, medium CD SCT 2.1 kN/m, RMS Taber moment 31.4 gf·cm, and TAPPI T 839 ECT 7.9 kN/m. The table's rotated header visibly gives caliper units as μm. The companion lab-made board is 4,130 μm (= 4.130 mm), basis weight 646 g/m², and ECT 8.7 kN/m. The commercial ECT value is the safer first board-level target when the simulation reproduces that board type; the 2009 ECT predictions are model validation across flute classes but are not a crease test.

Popil (2017) Table 4.1 p. 61 gives separate paper samples: 42# kraft liner, 212 g/m², 277 μm soft-platen caliper, apparent density 766 kg/m³; and 26# neutral-sulfite semichemical medium, 130 g/m², 191 μm soft-platen caliper, 679 kg/m³. These are useful class-adjacent thickness/density references only. The book does not identify them as the actual WC 4226C 42 constituent rolls. Keep any use of those thicknesses/densities labeled as an alternate sample proxy. They must not be merged silently with Haj-Ali's 205/126 g/m² constitutive data and called the same exact stock.

## Implemented `literature2009` starting recipe

The current native starting geometry is a numerical coupon **31.6 mm long × 20 mm wide**, with four 7.9 mm flute pitches along its length. Length and width are implementation choices, not published coupon dimensions. The full-board caliper target is Popil (2017) Table 5.4's WC 4226C commercial-board value **4,210 μm = 4.210 mm**. For the constituent thicknesses, the recipe uses Popil (2017) Table 4.1's separate coupons: liner **277 μm** and medium **191 μm**, with apparent densities **766 kg/m³** and **679 kg/m³**, respectively. These are not known to be the WC 4226C constituent rolls or Haj-Ali's tested paper sheets; the assembled recipe is consequently a mixed-source proxy.

The selected double-back lower glue gap is 0.100 mm thick over 0.800 mm width; the selected single-face upper gap is 0.030 mm thick over 0.600 mm width, matching the source Table 8 Type II rectangular dimensions before adaptation to curved bridges. The core flute centerline's **peak-to-trough height** is derived to close the total caliper:

`h_centerline = 4.210 − 2(0.277) − 0.191 − 0.100 − 0.030 = 3.335 mm`.

For a sinusoid written `z(x) = A sin(2πx/7.9 mm)`, use half-height **A = 1.6675 mm**. This geometry calculation includes two liner sheets and one medium sheet, plus the two different glue gaps. It is a geometric packing assumption: actual paper compression, flute take-up, and bondline variation are not resolved by those nominal thicknesses. The resulting 3.335 mm centerline height is not the Haj-Ali 3.50 mm FE idealization or 3.61 mm typical reported height.

The glue is modeled as an isotropic elastic solid with **E = 400 MPa** (the reported fully dried starch value), **ν = 0.3** (assumed by the source), and **ρ = 1500 kg/m³** (numerical assumption; no source-matched glue density was found). The curved finite bridges are perfectly bonded and have no cohesive separation or fracture law.

The paper model uses the Haj-Ali orthotropic elastic constants and compression-calibrated Hill yield surface, implemented as **ideal plasticity with zero hardening modulus**, associated flow, and Green-strain kinematics. This is an implementation choice, not a full reproduction of Haj-Ali's plotted bilinear elastic–plastic curves: their post-yield slopes are not tabulated and this starter does not digitize them. It does not include Tsai–Wu failure or any paper failure/damage evolution. The Hill surface's compression calibration includes source-assumed compression data; it is not a direct reproduction of a measured compression test.

The initial state is a **preformed, conditioned, stress-free board geometry**. The simulation starts after the flute, liners, and glue bridges have been assembled. It does not simulate paper corrugation/forming, corrugator heating or pressure, adhesive curing, moisture change, or residual manufacturing stress. Physical folding, crease response, springback, and box assembly validation have not been demonstrated. Published ECT values remain possible future comparison targets, not evidence that this starter already matches a physical board.

## What the sources do not identify

- **Cured paper thickness and density for the 2009 constitutive sheets:** unreported. Basis weight alone does not determine caliper or bulk density. The nearby 2017 samples above are from separately described specimens.
- **Glue density:** unreported. A dried corn-starch bioplastic-film density or raw granule density would be the wrong material class for a corrugating Stein–Hall layer and is not used here.
- **Starch bond strength / fracture:** no source provides a matching-grade glue traction strength or fracture energy, or a validated cohesive damage law. Structural pin-adhesion/ECT numbers cannot identify local interface strength or Gc without bond area, failure locus, and a calibrated interface test. Use a perfectly bonded finite elastic bridge if that is the current model; do not call it debonding-capable.
- **Cure / moisture / thermal history:** the simulator reports bath temperature and sequence, but not a complete adhesive solids, gelation, drying, line-speed, pressure, or residual-moisture history. The available evidence does not parameterize manufacturing-rate or temperature-dependent bond setting.
- **Creasing:** these sources do not validate rotary indentation, crease damage, fold moment-angle, residual set, springback, or post-crease strength. The first source-faithful work can establish an instrumented uncreased board and benchmark published ECT or elastic bending. A controlled crease requires additional validated paper damage/plasticity and contact/crease data.
- **A stiffness-only/density assignment is not paperboard qualification.** Retain the distinction between a runnable constitutive proxy, a verified solver path, and an experimentally validated cardboard response.

## Suggested labels for the native example

Call the recipe `literature2009` and identify it as a **mixed-source literature proxy**. The current starter uses measured 7.9 mm pitch and caliper-closed 3.335 mm core centerline height, not the source's separate 8.50 mm/3.50 mm FE sinusoid. Declare its 31.6 × 20 mm coupon dimensions, separate-coupon paper thickness/density, 1500 kg/m³ glue density, ideal-plastic Hill implementation, and curved glue-bridge adaptation. Preserve the source's plotted hardening as unimplemented and do not claim same-lot reconstruction, manufacturing simulation, cohesive failure, physical fold, or box-assembly validation.
