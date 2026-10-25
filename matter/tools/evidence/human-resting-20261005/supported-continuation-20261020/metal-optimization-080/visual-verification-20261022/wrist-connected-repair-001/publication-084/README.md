# Connected skin/fat construction increment 084

This increment extends existing owners; it does not install a render-only skin or change material/contact/accepted-state physics.

The boundary-refinement owner now creates conforming shared edge midpoints and affected-face centroid fans, retaining parent-face barycentrics and explicit affine CSR source ancestry. Positive registered skin/fat depths use the same convex prolongation. The real 330-face crossing set refines to 10,145 points and 20,245 faces. Canonical points match the frozen construction bitwise; faces/ancestry/CSR match by value in emitted order (some integer storage widths differ). Depths differ by at most 1.74e-18 m due to summation order. Exact self audit finds zero pairs. Refinement does not remove crossings: complete all 185-bank checks find 380 generated vertices inside bone and 2,746 triangle pairs on the subdivided surface.

The registered-anchor owner now resolves coarse SOURCE IDs to unique typed output rows. The old construction runners incorrectly froze numeric source IDs as row indices. Of 237 anchors only 56 numeric IDs coincided; 65 true anchors moved, maximum 1.2143 mm. Their candidates remain unadmitted evidence. A corrected connected start restores only the true anchors and reinterpolates dependent generated points. Its complete exact self/185-bank point check has zero self pairs, 380 inside pairs and no ambiguous/invalid classifications.

The bone-audit owner adds a proposal-only closest-point plane transform using exact matrix inversion for positions and covector transformation for normals. Raw registered matrices need not be perfectly orthogonal. All 2,915 pinned outside-query planes reproduce the frozen arrays bitwise. A local plane is not a global clearance certificate. The route audit's construction-plane reporting policy remains unchanged.

The combined existing/new owner suites pass 66 tests on SSH Numi. The retained test predicate path is corrected to the existing repository evidence owner, rather than the historical scratch parent.

Earlier connected soft fitting reduced 1,868 to 1,638 triangle pairs before refinement, but its anchor-ID bug prevents promoting it as a boundary-preserving repair. These counts are not directly comparable with the subdivided 2,746 count.

The joint deep-domain remesh preserves the original outer facets, closed oriented boundary and volume (relative difference 3.35e-12), with positive emitted Float32 cells after orientation normalization. It reduces cells below q=1e-5 from 6,877 to 3,034, but minimum q=1.31e-10 still FAILS the actual skin/fat admission floor. Its 10,198 Steiner points also exceed the declared 10,000 construction target; both are retained, not hidden.

No candidate is installed. Zero native physical steps are accepted in this increment. Bone face clearance, matching final outer/inner domains, all material-cell quality/positivity, conservative mass/transfer/contact ancestry and native qualification remain incomplete. Temporary head caps remain construction only. Current physical and physiological owners, forces, source identities and unrelated checkout changes are preserved.
