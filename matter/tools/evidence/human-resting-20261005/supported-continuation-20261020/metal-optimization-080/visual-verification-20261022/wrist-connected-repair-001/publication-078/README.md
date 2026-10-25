# Connected skin/fat reconstruction increment 078

This increment adds provenance-correct source anchors and conforming boundary refinement to the existing skin/fat construction tools. It does not admit a new physical skin or change per-step Metal physics.

## Implemented and tested

- Registered source anchors use the accepted coarse CSV source ID and the detailed source-ID map. Missing direct anchors use the exact recorded source face and barycentric coordinates.
- A one-hot displacement-transfer weight is not a coincident geometry anchor. The actual input audit finds nine coarse columns where choosing the minimum-index one-hot row is wrong by more than 1 micrometre, reaching 17.1108 mm.
- Source-to-coarse offsets are retained explicitly: 2,332 direct anchors reach 2.946 mm; 79 barycentric anchors reach 11.598 mm. An optional coincidence gate must be requested explicitly.
- Boundary refinement keeps typed parent ancestry, exact shared endpoints, orientation checks and the existing admission restrictions. Source-normal construction is an explicit diagnostic path, not a runtime shortcut.

All **210 canonical tests and 11 subtests pass on SSH Numi**. The registered-anchor helper adds eight tests, including nonidentity source IDs, uint32 faces, exact barycentric offsets and rejection of missing provenance.

## Geometry results and limits

Linear conforming subdivision preserves the existing surface and has zero PLC self-intersections, but whole-volume tetrahedralization still leaves four cells below the existing quality floor. One legal same-material connectivity repair reduces this to three. It does not repair the source boundary shape.

The curved-source proposal has 626 PLC self-intersection pairs and is rejected. Connected translations also fail the inner-layer surface checks. A bounded connected continuation removes surface self-intersections but leaves five below-floor tetrahedra and unresolved bone crossings. More sampled inside points are not interpreted as a change in penetration volume without a matched query domain.

The correct next construction is a connected source-patch replacement with the skin and fat sharing its topology. Active source-patch cutting and harmonic blending are deliberately excluded from this frozen increment. Their failures must not be promoted as accepted anatomy.

The previous conservative mass map cannot be reused after changing reference geometry. Layer volume, regional mass, material ancestry, detailed-surface transfer, surrounding anatomy and retained-pose/native qualification remain required. The head-height skin boundary loops also remain separately unresolved.

## Reproduction and ownership

Use canonical-tests.log, source-artifacts.json, canonical-manifest.json, verified-published-inputs.json and verify_retained_artifacts.py. Failed as-run geometry and anchor audits are retained, including the uint32 overflow and the invalid assumption that every direct source anchor is coincident with the current coarse point.

Mixed-source anatomy remains a reference assembly. No material law, contact owner, force owner, Brain state or accepted-state rollback has been replaced. No native scene was installed and no physical steps were accepted by this increment.
