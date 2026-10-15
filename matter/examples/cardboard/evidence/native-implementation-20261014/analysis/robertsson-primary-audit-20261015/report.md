# Robertsson et al. (2023): primary-source convention audit

Audit date: 2026-10-15. This is a read-only equation and provenance audit of the current Robertsson authoring path. It is not a material validation or a parameter-transfer proposal.

## Primary publication and access

Robertsson, Jacobsson, Wallin, Borgqvist, Ristinmaa, and Tryding, “A continuum damage model for creasing and folding of paperboard,” *Packaging Technology and Science* 36(12), 1037–1050 (2023), DOI [10.1002/pts.2774](https://doi.org/10.1002/pts.2774). The official Wiley article is open access. I inspected Wiley's full-text PDF reader at <https://onlinelibrary.wiley.com/doi/pdf/10.1002/pts.2774> and HTML at <https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774> on 2026-10-15. The PDF reader exposes 14 pages. Direct byte download of the official PDF returned HTTP 403 in this environment, so no original-PDF bytes or PDF SHA-256 are claimed here. The publisher PDF text and page layout were available through the reader; the publisher equation transcription already retained in the repository is separately SHA-bound below.

## Grade and reported material data

The paper says it reuses the paperboard from Borgqvist et al. (2016), and describes it as a single-ply structure made entirely of softwood fibers, 0.39 mm thick and 788 kg/m^3 in density; representative fiber length is 3.6 mm and fiber thickness is 10–25 micrometres (PDF pp. 4–5, printed pp. 1040–1041, Section 2.5). It does not give a commercial product/grade name. The single-ply statement means the model assumes uniform properties through thickness; it does not establish that this is a corrugated liner, NSSC medium, or glue grade.

Table C1 (PDF p. 13, printed p. 1049) lists E1..E6 = [1690, 292, 28.4, 1333, 78.0, 0.365] MPa and E7 = 11.5 dimensionless. Table C2 (PDF p. 14, printed p. 1050) gives K0 [MPa] and hardening coefficients a [MPa], b [dimensionless]: nu 1: 20.4, 12.2, 690; 2: 11.7, 5.10, 435; 3: 9.91, 6.63, 334; 4: 22.0, dash, dash; 5: 16.7, dash, dash; 6: 9.91, 6.63, 334; 7: 1.00, 49.0, dash; 8: 0.40, dash, dash; 9–12: 2.51 each, dash, dash. Equation 27 uses a*ln(b*kappa+1) for nu={1,2,3,6}, a*kappa for nu=7, and no hardening for the remaining modes. The dashes are absent parameters in the table, not measured zero values.

Table C3 gives c=[8,8,9,9] dimensionless and beta=[2.85,2.85,3.98,3.98] MPa^-1 for nu=9..12. The reference crease element projected area is 0.0011 mm^2 (Section 3.1). The article states E1, E2, E4, E5 come from MD/CD/45-degree tension stiffness comparison; E3 is estimated from initial out-of-plane tension stiffness; E6/E7 from out-of-plane compression unloading; nu12=0.46, nu21=0.19 are obtained using an in-plane Poisson product from prior paperboard measurements; K0 modes 9–12 are acknowledged as weakly observed/estimated from crease force response. These are reported parameter assignments with mixed provenance, not a raw same-grade dataset. The paper's data-availability statement says all data are in the article and simulations used commercial code; it provides plots, not raw curve tables or the authors' UMAT source.

## Stress, strain, energy, and evolution conventions

The paper's base variables are finite-strain quantities: F=Fe Fp; d=sym(Fdot F^-1); tau is Kirchhoff stress; Psi=(1-alpha)Psi0 and Psi0=Psi0e+Psi0p (Eq. 2). The elastic invariants in Eqs. 18–19 are elastic stretches/area measures (I11=|v1|, I12=|v2|, I13=|Fe n0^(3)|, I23=|n^(3)|), not engineering strains. Equation 20 defines undamaged Kirchhoff stress tau0; total stress is tau=(1-alpha)tau0. The yield surface and flow are based on tau0 (Eqs. 8–10, 21–23). Table C1 moduli and K0/a are in MPa, and the invariants/hardening b are dimensionless, so Psi has the dimensions of energy per volume (MPa = 10^6 J/m^3). The paper uses backward Euler and local Newton updates (Appendix B, Eqs. B1–B6).

## Equation 18 finding: a genuine source-internal conflict

The official PDF, Section 2.3, Eq. 18 (PDF p. 3, printed p. 1039) prints the ZD compression contribution as

`(1-H) E6 [ I13^2 + exp(-E7 (I13^2 - 1))/E7 - 2 ]`,

and defines H=1 for I13 >= 1 and H=0 otherwise. Table C1 supplies E6=0.365 MPa and E7=11.5. The same paper's explanation of Eq. 17 says its damage dissipation inequality holds because Psi0 >= 0 (PDF p. 2, printed p. 1038).

For a pure elastic thickness compression Fe=diag(1,1,s), Fp=I, kappa=0, the other energy terms vanish and I13=s. The literal Eq. 18 compression branch therefore has

`lim(s -> 1-) E6 [s^2 + exp(-E7 (s^2 - 1))/E7 - 2] = E6 (1/E7 - 1) = -0.3332608695652174 MPa`.

At s=0.99 it is -0.3323625426152471 MPa; at s=0.9 it is -0.1521685459147348 MPa. At s=1, the published Heaviside selects the tensile branch, whose energy is zero. Thus the printed law has a finite jump at the branch point and admits negative Psi0 for a purely compressed elastic state. Since kappa=0 here, the plastic free-energy term is zero and cannot resolve the inconsistency. This directly conflicts with the article's Psi0 >= 0 statement. This is not a units issue: the offset is in MPa, the same energy-density unit as every E coefficient.

The explicit `recentered` alternative replaces the bracket constant -2 by `-1 - 1/E7`; this adds `E6(1 - 1/E7)=+0.3332608695652174 MPa` only on I13<1 and makes the compressed-side limit zero. The additive shift leaves the within-branch deformation derivative unchanged, but it changes absolute Psi and therefore the damage conjugate A=-dPsi/dalpha=Psi0. The primary paper does not state this alternate constant or identify an executed source implementation that resolves the conflict. The correct source-faithful claim is therefore “literal printed equation, with an unresolved thermodynamic inconsistency”; the recentered form is an explicit engineering comparison, not author-verified behavior.

## Current code comparison

Current source is `matter/tools/robertsson2023_material.hpp` (SHA-256 `52dc8daf9e417e773ec25ef288561460e786d7aa1e5dce8f715f94ebc61da3f7`). It has explicit `paper_absolute` and `recentered` enum values; `paper_absolute` selects the exact printed -2, while the separately named legacy fixture uses `makeLegacyRecenteredMaterial()`. The own-grade reference builder requires an explicit convention and uses source area 0.0011 mm^2, damage cap 0.96, density 788, E1..E7, nu12/nu21, m=0.7, k=3, and Table C1–C3 values. It forms Psi=(1-alpha)(Psi0e+Psi0p), so convention choice affects A as the source equations require. The code represents the source stress/yield terms in numerical MPa and multiplies the stored energy by 1e6 to return Pa, i.e. J/m^3. Its Eqs. 18–23 terms and Table C parameter values match the printed equations/tables on inspection. No code correction is warranted from this audit.

The current compiled check `matter/tools/robertsson_fem_coupling_check.mm` (SHA-256 `755ff822c826553de655413f5ead270c071f99a36bad9988acacdd338fb75380`) and `compare_robertsson_fem_point.py` (SHA-256 `06b70eb62f42b7e575c7387e7cc74024b6d53cf8747f143a556ae45f1744d426`) are implementation/FEM state-force coupling checks, not an independent implementation, a digitized published-curve comparison, or a physical calibration. The retained source note and benchmark correctly preserve the compression discrepancy rather than hiding it.

## Reproducibility and limits

Decimal recomputation used the exact printed E6/E7 values at 40-digit precision; inputs are the source constants above. The repository's published transcription (`robertsson_2023_eq18_source_extract.md`) SHA-256 is `fd1ddbfb50d039232b359d58e0a2245151e70cd42755dc8fbfb08b0acf71bddc`; its compression-branch numeric CSV SHA-256 is `8392aef6412eda1467e787a265a6ae6094f3ce81368187e3937a0e061340608e`; its source catalog SHA-256 is `f812ca264c3f0cd15e98178aeae090332eb753d127a199906a9ebc800e3a62c1`. The original publisher PDF was not locally retained, so the PDF has no claimed byte hash. A fresh audit must return to the DOI/publisher URL and compare against Eq. 18, Eq. 17 discussion, and Tables C1–C3.

No coefficients from this single-ply softwood paperboard are to be transferred to the kraft liner, NSSC medium, or starch interface. Its damage law is calibrated to crease/fold setups and a reference brick mesh; the current source-equivalent point/FEM instrument does not reproduce those test geometries or provide physical corrugated-board qualification.
