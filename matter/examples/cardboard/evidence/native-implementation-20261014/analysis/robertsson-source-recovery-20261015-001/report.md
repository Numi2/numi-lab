# Robertsson source recovery audit

Audit date: 2026-10-15. This is a bounded primary-source access audit of Figure 4 and the Equation 18 discrepancy. No repository files were edited, no figure values were digitized, and no source parameters were transferred.

## Result

No new hashable primary asset or author implementation was recovered. The official Wiley article remains readable in its HTML and PDF-reader views, and its article record labels it Open Access. The Wiley PDF reader exposes 14 pages of extracted text, including the paper's Figure 4 legend and discussion, but the accessible reader response did not yield retained PDF bytes or rendered page pixels in this environment. The original source PDF SHA remains unavailable.

Wiley's official Figure 4 PowerPoint link resolves to its `action/downloadFigures` endpoint for `pts2774-fig-0004`. The web reader reports HTTP 400 because the returned content type is `application/vnd.ms-powerpoint`, which it cannot present. A direct unauthenticated GET of the publisher PDF, `pdfdirect` variants, and the same figure endpoint returned HTTP 403. No attempt was made to evade access controls. The publisher article and the Lund University Publications record were accessible as pages, but the institutional record listed only the DOI and metadata, not an attached author manuscript or source package.

The publisher article's data-availability statement says that data are presented in the article and that simulations used commercial code. The article does not link an implementation or supplemental code. A bounded exact-DOI/title search of the official Wiley and Lund publication domains and an exact paper/title search on GitHub surfaced no corrigendum, erratum, or author implementation. This is a search result, not proof that no private or unindexed implementation exists. No evidence was found that resolves the printed Eq. 18 compression offset. The discrepancy must remain open; it is not valid to call the printed constant a typo or to choose the recentered convention as the authors' executed law.

## Figure 4 boundary

The official article identifies Figure 4 as the post-creasing folding curve for MD- and CD-aligned tests and describes the damage-model plateau near 20 degrees. The text does not contain digitized ordinates or a numeric curve table. Because no source pixels or downloadable figure file could be retained, Figure 4 remains un-digitized. No axes, curve values, samples, uncertainty, or train/holdout labels are inferred. The current single-ply softwood material remains a separate grade from the Kraft/NSSC recipe.

## Preserved Equation 18 finding

The existing source audit remains the current equation evidence: Eq. 18 prints the ZD compression bracket with `-2`; with Table C1 `E6=0.365 MPa`, `E7=11.5`, its compressed-side limit is `E6(1/E7-1) = -0.3332608695652174 MPa`, while the adjacent stated requirement is `Psi_0 >= 0`. The publisher page still exposes the same printed equation and parameter table, and no primary correction or author implementation was found in this bounded search. This is an unresolved internal source conflict, not an adjudication in favor of either convention.

## Reproduction and access log

The machine-readable `access-attempts.json` lists the primary URLs, retrieval routes, outcomes, and limits. `manifest.json` binds the existing repository audit inputs by SHA-256 and declares that no source bytes were acquired. Web-reader access is identified by the official Wiley URL and the audit date; no hash is claimed for the transient reader output.

Primary references:

- Wiley article and reader: https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774 and https://onlinelibrary.wiley.com/doi/pdf/10.1002/pts.2774
- Official Figure 4 download endpoint: https://onlinelibrary.wiley.com/action/downloadFigures?doi=10.1002%2Fpts.2774&id=pts2774-fig-0004&partId=
- Lund University Publications record: https://lup.lub.lu.se/search/publication/ff8ac441-219a-4ec1-8806-81fafe2cd72e
