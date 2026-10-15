# Cardboard published source-data package

This package records published measurements, source-reported assumptions,
digitized plot estimates, and unavailable inputs as different evidence classes.
It uses no new physical measurements. The approved `literature2009` recipe
remains a cross-source model recipe; the rows below do not make it a
same-roll reconstruction.

Original PDFs are kept outside the repository under
`/Users/home/cardboard-research-20261006/`. The catalog stores each retained
PDF's SHA-256. No original PDF or slide image is republished here. To check the
catalog and embedded CSV hashes, run:

```sh
python3 matter/tools/cardboard_source_data.py validate
python3 matter/tools/cardboard_source_data.py audit
python3 matter/tools/cardboard_source_data.py check-cohesive
python3 matter/tools/cardboard_source_data_test.py
```

`acquire-pdf <source_id> --output /external/archive/name.pdf` can retrieve a
catalog-listed public PDF when its URL is directly downloadable. It refuses
paywalled sources, requires an external destination, enforces a 100 MiB limit,
and verifies the registered hash when one exists. A blocked response is kept as
an access limitation; the tool does not work around it.

`check-cohesive` exits with code 3 on purpose: no available publication
identifies a local mixed-mode cohesive law for the current Stein-Hall starch
liner-to-medium interface. A nonzero code here is the expected negative
identifiability result, not a passing calibration.

## What is included

| Dataset | Status and permitted use |
|---|---|
| `haj_ali_2009_yield_anchors.csv` | Four Table 1 tensile yield anchors for 205 gsm kraft liner and 126 gsm NSSC medium are usable as source-grade yield anchors. Compression values are separately marked as the article's 60%-of-tension modeling assumption. They are not compression measurements. The retained author-posted page-4 render makes the four solid bilinear tension-fit lines in Figs. 1–2 readable. Those source-fit estimates are stored separately from the Table 1 anchors; raw experimental marker traces and specimen identities remain unavailable, and compression rows remain assumptions. |
| `haj_ali_2009_bilinear_tension_fit_nodes.csv` | Twelve linked nodes: origin, Table 1 tensile-yield point, and one visually read endpoint on each of the four source-labeled bilinear tension-fit lines in Figs. 1–2. Tangent intervals propagate ±2 px endpoint readout, ±1 px axis-line location, and Table 1 rounding. The CSV also derives a uniaxial plastic modulus `H=E*Et/(E-Et)` from each tangent and Table 2 E. This is an estimate of the authors' fitted curves, not raw experimental traces, individual tests, a full multiaxial fit, or a confidence interval. Compression curves are excluded. |
| `haj_ali_2009_experimental_marker_centers_v1.csv` | Selected visible marker-glyph centers from the same four Fig. 1–2 panels, transformed from integer raster pixels to strain/stress using the registered plot frames. The CSV includes a ±1.5 px glyph-center bound and ±1 px axis-line bound and one explicitly unresolved overplot region per panel. This is a partial visual extraction, not raw instrument data: hidden/overlapping marker counts and specimen/replicate identities remain unresolved. These points are from the same publication and are calibration inputs only, not an independent or held-out validation set. The reviewed pixel ledger is separately hash-bound. |
| `popil_2012_ect_table2.csv` | Published C-flute ECT/caliper/basis-weight rows. The commercial 42-26C-42 board and separate IPST lab board stay distinct. Use as structural analogue holdout only. |
| `popil_2012_ect_rate_fit.csv` | Derived evaluations of the article's regression `ECT = 0.239 ln(v) + 6.47 ± 0.01`, with `v` in mm/min. The source calls ±0.01 the mean regression error; it is not a confidence interval. These are equation evaluations, not observed points. The 12.5 mm/min evaluation (7.073612 kN/m) is distinct from the Table 2 commercial result (7.9 kN/m), so the series are not merged. |
| `johnson_popil_2016_pin_force_displacement_digitized.csv` | Sparse visual estimates directly digitized from the Georgia Tech authors' 2016 presentation slide 15: load in N against displacement in mm for two illustrative defect traces. The 2015 article is separately recorded as protocol/study context. Its article Figure 4 is not pixel-digitized here and exact curve-to-specimen identity is unresolved. The high-response trace is truncated to the visible branch; the annotated later branch is excluded. |
| `johnson_popil_2016_pin_strength_summary_digitized.csv` | Visual estimates of summary markers directly digitized from presentation slide 16. They are not raw per-specimen observations and are not mapped to the article's Figure 5 points or its six defect categories. The article narrative and slide 16 print the work unit `N-m`; integrating slide 15's `N` by `mm` curve gives `N·mm`. This conflict remains unresolved, so slide-16 work values are axis numbers only and are rejected for physical-work fitting. |
| `aduke_2024_fold_ratio.csv` | A reported rounded crease-to-flat stiffness ratio of 0.37 is admissible as a within-study dimensionless holdout. Absolute Table 5 stiffness stays unadmitted: the printed unit `N/m` conflicts with the printed equation's `N*m` dimensions. |
| `nagasawa_2012_fig9_scored_bend_envelope.csv` | A pooled outer-envelope estimate from the ten overlapping traces in Figure 9 of the open J-STAGE AB double-wall study. It is one complete test-condition group from two source specimens, sampled at 1° increments; the lower response is censored against zero for 1–34°. The ten traces are not individually identified and the rows are never split point-wise. This is a cross-grade bend analogue, not a fit to the current recipe. |
| `mrowczynski_2021_box_bct_holdout.csv` | Reported 300 x 200 x 200 mm box compression results for BC-780 and EB-965 grades. This supports a geometry/load-path holdout only; both are different double-wall materials. |
| `conde_2012_hma_joint_scope.csv` | Hot-melt corrugated double-lap joint analogue. `Sempacol-1 HM-577` is a hot melt, not the internal starch. The source's target was paper delamination and the hot melt remained elastic in its model; it supplies no starch cohesive strength or fracture energy. |
| `robertsson_2023_identity_energy_benchmark.csv` | Deterministic tension-side identity check for the separate single-ply softwood grade; not an experiment and not transferable to current liner/medium. It does not test the compression-side Eq. 18 energy discrepancy. |
| `robertsson_2023_eq18_compression_branch.csv` | Exact algebraic comparison of the literal `paper_absolute` Eq. 18 term and explicit `recentered` builder term. This is not an experiment or a physical-material benchmark. It exposes the additive energy difference that changes `A=Psi0` in the damage potential while preserving deformation derivatives within that branch. |
| `numi-matter-robertsson-fem-coupling-check --cpu-only` | Independent CPU evaluation of the Robertsson source energy-expression graph attached to the compiled world against Eq. 18 at the reference, compression, and tension states. It verifies the literal negative compressed-side energy and resulting damage conjugate; it is a source-equation/software check, not a Metal-kernel run, experiment, author-code reproduction, or corrugated-grade validation. |
| `robertsson_2023_strength_hardening.csv` | Table C2 strength/hardening parameters with Eq. 27 law labels for Robertsson et al.'s separate single-ply softwood grade. Source dashes remain blank; they are not missing-data zero defaults and are not transfer values for the Kraft/NSSC recipe. |
| `impact_data_availability.csv` | Explicit availability boundary. One dynamic study reports confidential raw data; the open drop-test analogue is a padded package with different construction and no raw synchronized response for the target box. No dynamic calibration is admitted. |

## Sources and extraction

The source links, paper identifiers, PDF hashes, specimen conditions, failure
observables, data split, digitization uncertainty, and known limits are in
[`catalog.json`](catalog.json). The public [Popil 2012 article and PDF](https://bioresources.cnr.ncsu.edu/resources/overview-of-recent-studies-at-ipst-on-corrugated-board-edge-compression-strength-testing-methods-and-effects-of-interflute-buckling/)
report ECT tests on commercial and laboratory C-flute boards; the package does
not combine their separate series.

The [Johnson–Popil 2015 article](https://doi.org/10.1016/j.ijadhadh.2015.02.003)
is available as author-posted accepted-manuscript text on its
[ResearchGate page](https://www.researchgate.net/publication/272753889_Corrugated_board_bonding_defect_visualization_and_characterization).
Section 2.2 reports the Emerson 06-00-55 C-flute fixture, Instron 1122,
50 × 160 mm specimen, 12.5 mm/min platen approach, and out-of-plane selected-
facing loading with alternating pins 1/3/5 and 2/4/6. It does not dimension
pin diameter, pitch, pin-to-flute registration, or the specimen axes relative
to MD/CD. The author-page PDF endpoints returned 404 in this review; a
hash-bound protocol record is retained in
[`johnson_popil_2015_pin_protocol_record.md`](johnson_popil_2015_pin_protocol_record.md).

The [2016 author presentation PDF](https://research.gatech.edu/sites/default/files/rbi/pdfs/Roman_CORBOTEC%20PDF%20for%20distribution.pdf)
is the direct pixel source for the checked-in slide-15 and slide-16 CSVs. Its
slides list the pilot study's 205 g/m² kraft liner and 122 g/m² medium. The
article and presentation are the same study context and may overlap in tested
specimens, but no point-level or Figure-4/Figure-5 identity is established.
Only the presentation appears in the digitized datasets' `source_ids`; the
article is an explicit protocol/context source and cannot be selected as a
second independent dataset. `catalog.json` records the figure-category,
strength-range, and work-unit conflicts separately.

The pin geometry input for a native forward prototype is registered in
[`pin_adhesion_fixture_profile.json`](pin_adhesion_fixture_profile.json).
Six alternating pins and their upper/lower series are described in the
article. The prototype's 1 mm pin diameter, 8 mm pitch, centered registration,
5 mm board caliper, 1 mm end overhang, and 0.05 mm initial clearance are
engineering choices, not Emerson fixture dimensions or TAPPI tolerances. The
profile and C++ authoring helper are hash-checked and tested. This only
constructs fixture inputs; it does not claim a completed native fixture run.

Digitization errors describe visual chart reading only. They do not include
specimen scatter or measurement-system error. The slide-16 work ordinate
remains an unqualified source-axis number because the printed `N-m` conflicts
with the `N` and `mm` axes used for the slide-15 integral. The source-data
calibration tool rejects converting it to `N·mm` or joules.

The [Haj-Ali et al. paper](https://doi.org/10.1016/j.compstruct.2008.02.001)
is the source for the recipe's 205/126 gsm constituent yield anchors and
orthotropic paper model. The author-posted article page exposes the tables and
rendered plots. The retained page-4 render is hash-bound outside the repository.
`cardboard_source_haj_ali_digitize.py` reads one endpoint from each solid
source-labeled bilinear tension-fit line and joins it to the Table 1 yield node.
This supplies an approximate source-model tangent for MD/CD liner and NSSC
medium. `cardboard_haj_ali_marker_digitize.py` converts selected, reviewed
pixel centers in `haj_ali_2009_experimental_marker_centers_v1_pixels.json`
from the same page render into plot-axis units. The marker rows are partial
plot estimates: the full raw experimental traces, exact count of overplotted
symbols, and specimen identities were not recovered. The marker and fit-node
datasets come from the same published panels, so they cannot be separated into
calibration and validation evidence. Neither digitization supplies specimen
scatter or compression data; the published compression response remains a
modeling assumption. The original PDF download returned 404 in this session.

The [Aduke et al. open article](https://doi.org/10.3390/mca29040070)
provides the crease-bend analogue. The ratio is kept separate from absolute
stiffness because the table's `N/m` caption conflicts with the dimensional
result of the printed equation. The [Mrowczynski et al. open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8467740/)
provides a 300 x 200 x 200 mm BCT geometry analogue, with its BC-780 and
EB-965 results held out from material calibration.

The open [Nagasawa and Komiyama rotary-scoring study](https://doi.org/10.1299/jamdsm.6.1263)
provides a scored-bend response for a separate AB double-wall grade. Figure 9's
embedded raster was retained and hash-bound outside the repository; the CSV is
the pooled outer pixel envelope of its ten overlapping traces, with explicit
pixel-to-axis mapping and ±2-pixel readout bounds. Its Figure 9 condition is
`cP/t=0.38`, `cC/t=0.09`, and `V=2.92 m/s`. The separate `cC/t=0.09–0.40`
sweep in the same article means main-creaser roll gap divided by board
caliper; it is not a punch indentation ratio. The ordinate is retained as
moment per unit width, `N·m/m`, with `M=(324-y_pixel)×5/(324-21)`; its
dimension reduces to newtons, but the normalization remains part of the
observable. See the catalog and external
[source-access audit](/Users/home/cardboard-research-20261006/primary-access/primary-source-access-audit-20261014.md)
for the extracted-image and PDF hashes, the reproducible extraction command,
and the curve-identity and zero-axis-censoring limits.

The [official Wiley Robertsson article](https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774)
reports its twelve initial yield distances and hardening parameters in Table C2,
with the law in Eq. 27. These are model inputs for that single-ply softwood
paperboard, not an experimentally tabulated raw stress-strain curve and not a
matched set for the corrugated Kraft/NSSC recipe. The source extract and CSV
preserve em dashes as blank; the source-law adapter's zero branches implement
the paper's stated Eq. 27 structure only for that grade.

The [Conde et al. HMA joint article](https://doi.org/10.1016/j.ijadhadh.2012.05.003)
and the [Robertsson et al. paperboard source law](https://doi.org/10.1002/pts.2774)
are kept as distinct analogues. Neither identifies the current corrugating
starch interface. Cáceres-Naranjo et al.'s [dynamic study](https://doi.org/10.1016/j.rineng.2023.101072)
states that its experimental data are confidential; the package does not
reproduce or reverse-engineer them. Liu and Laundrie's [1987 edge- and
corner-drop report](https://www.fpl.fs.usda.gov/documnts/pdf1987/liu87a.pdf)
is an analogue involving folded corrugated cushioning pads and a dummy load.
The official PDF returned HTTP 403 in this session, so only indexed report text
was reviewed and no raw response was extracted.

The [Haj-Ali et al. (2009) author-posted article page](https://www.researchgate.net/publication/223896964_Refined_nonlinear_finite_element_models_for_corrugated_fiberboard)
is publicly readable. Its page-4 plot render is retained and hash-bound outside
the repository. The separate fit-node dataset estimates only the four drawn
bilinear tension fits; it is not raw test data. The direct PDF download returned
404 in this session. For Robertsson et al. (2023), the
[official Wiley open-access text](https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774)
provides Eq. 18 and source-grade fold figures. The exact printed-versus-recentered
energy-branch comparison is recorded in
[`robertsson_2023_eq18_source_extract.md`](robertsson_2023_eq18_source_extract.md).
The CPU check confirms the reference state and the printed-branch energy
offset; its limits are recorded in
[`robertsson_2023_absolute_energy_audit.md`](robertsson_2023_absolute_energy_audit.md).
Figure 4 is plotted rather than tabulated; no numeric points have been
digitized from it, so it is not yet an experimental curve dataset. The
publisher figure endpoint returned an unsupported PowerPoint MIME type to the
web reader; the headed-browser article capture attempt returned HTTP 403 and
the in-app browser blocked the direct figure endpoint. No source pixels were
captured. These access attempts are recorded in the source catalog and the
hash-bound Equation 18 extract.

## Identifiability and model-use boundary

The available pin test produces a peak load and a global load-displacement
response of an assembled board specimen. The manuscript protocol is known,
but the sparse presentation traces lack raw samples, point-to-specimen mapping,
and a visible zero-displacement origin. They cannot uniquely recover local
traction, separation, or fracture work at the corrugating starch bond because
paper tearing, delamination, pin contact, fixture compliance, specimen width,
and defect morphology contribute to the measurement. Conde's HMA joint is a
different adhesive and its modeled target was paper delamination. Robertsson's
bulk damage is a different single-ply paper grade. Neither is a substitute for
internal starch.

The following remain unidentified for the exact current recipe: cured bridge
thickness and density, normal/shear peak tractions, mode-I and mode-II fracture
energies, critical opening/sliding, mixed-mode interaction, rate dependence,
and moisture/conditioning dependence. They remain explicit uncertainty sets
or unresolved parameters. No confidence interval is reported for them.

`cardboard_source_data.py digitize` accepts manually selected pixels only when
the request binds to both the original PDF hash and a page-render SHA registered
in the catalog, and states page, units, axes, series, and positive reading uncertainty. The
`fit-pin-summary` command fits only a descriptive line through digitized
summary markers and emits no confidence interval; it is not a cohesive-law
calibration. The data package fails closed if someone tries to claim the
current interface is identified without a new same-grade published source
dataset.
For the Nagasawa Figure 9 envelope, the visible stroke edges remain included;
the separate ±2-pixel bound describes line-edge/axis reading uncertainty, not
sample scatter or a confidence interval.
