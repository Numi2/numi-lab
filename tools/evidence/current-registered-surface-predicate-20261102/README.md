# Registered-surface predicate correction

The previous native surface checker reported 7,839 O pairs, 17,630 I pairs, and 30,951 D pairs on the canonical emitted Float32 source geometry. Reclassification by the retained exact binary64/integer predicate owner found zero erroneous intersections among all 56,420 witnesses. The old body-scale coplanarity and area tolerances exceeded the local feature scale. Geometry coordinates, triangles, identities, mass and material equations are unchanged.

The native admission checker now uses binary64 expansion signs for segment/triangle and coplanar tests. Indexed adjacency permits only intersection confined to the exact common feature; shared-edge folds are now checked too. The spatial broad phase remains conservative. The three complete emitted O/I/D surface audits pass with zero pairs, including shared edges.

Independent volume boundaries may contain multiple components only in the explicit volume-builder route, after exact facet incidence and registered boundary coverage. The default single-source-component policy is unchanged; manifold links, orientation, degenerate-face and intersection checks remain mandatory. Tests reject overlapping components and real folds while accepting shared indexed features and separated near-coplanar faces.

The new strict C++ regression and existing unpaired Skin/Fat volume owner tests pass. A stale assertion in the older runtime checkout expected a later rejection message; it was brought to the exact earlier-message assertion already present in published main. The original failed assertion and diagnostic are preserved. This control repair changes no admission logic.

These checks qualify source geometry and admission code only. Full native installation and accepted motion remain separately measured. Source/binary pins are retained in the build manifest; large geometry inputs remain on SSH Numi at the pinned paths.
