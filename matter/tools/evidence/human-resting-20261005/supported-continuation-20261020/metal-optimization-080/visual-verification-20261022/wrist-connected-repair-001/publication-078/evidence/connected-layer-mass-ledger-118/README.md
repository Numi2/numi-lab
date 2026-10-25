# Connected layer mass ledger 118

Read-only accounting and topology checks for the frozen 115 capacity-shortening proposal and 116 whole-layer remesh. All execution used the SSH Numi host and its numi-human-prep-venv-20261005 Python environment. No input mesh, source checkout, or native app was modified.

The authoritative starting masses come from the active compiled FEM tet ledger in 068 compiled-skin-fem-tets.csv, filtered to active native material IDs 15 (skin) and 16 (fat). It records 14,316 cells per material, skin density 1000.0000610351562 kg/m³, fat density 900 kg/m³, and exact parent mass totals 3.452571158709057 kg skin and 10.487108579460903 kg fat. Density bounds used for the aggregate capacity check are skin 900–1100 kg/m³ and fat 850–950 kg/m³.

Relative to those parent totals, both 115 and 116 have approximately unchanged skin volume and fat volume smaller by 0.000414674030095 m³ (−3.558718%). At the original fat density this would reduce mass by 0.373206627 kg. Aggregate mass preservation would require mean fat density 933.210324 kg/m³, within the stated range. This establishes only aggregate capacity; no density was changed and no per-cell transfer was applied.

115 has exact row/material/UID identity with the 084 remesh, but not with the authoritative native parents; its shortened reference also has 63 nonpositive fat cells. 116 has new TetGen cells, no child-to-native-parent cell CSR, and no geometric parent-child overlap measure. Its node ancestry cannot substitute for cell mass ancestry. The helper conservative_parent_transfer accepts a parent-row by child-column matrix of geometric overlap measures in m³, checks parent coverage and material isolation, and computes child masses/densities. Synthetic tests cover closure and fail-closed validation. It was intentionally not called on 116.

Topology checks found 45 boundary edges on each O/I/D sheet by itself. The actual tissue-region shells close when side caps are included: O + reversed I + 90 O-I cap faces, and I + reversed D + 90 I-D cap faces. Both 084 and 116 composites have zero boundary edges, zero nonmanifold edges, and an orientable closed chain. The 114 source artifact has 146 midpoint proposals, each endpoint edge verified against exactly two O source faces; 114 is not an emitted/refined mesh, so these checks do not qualify its proposed geometry.

mass-and-boundary-audit.json contains exact input SHA-256 pins, per-material ledgers, row mapping status, layer edge counts, and transfer limitations. Geometry/native admission remains false.
