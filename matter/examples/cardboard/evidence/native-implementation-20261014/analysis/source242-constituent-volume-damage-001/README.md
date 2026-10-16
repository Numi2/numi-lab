# Source 242: constituent-volume damage observables

This CPU-only overlay exports and checks two distinct numerical damage summaries over accepted paper material history. It does not change the constitutive law, native solver, physical operator, or tolerances.

- Volume-weighted mean damage index: sum(Vref_tet * D_tet) / sum(Vref_tet), with each tet index computed as sum(Aref_plane * D_plane) / sum(Aref_plane).
- Thresholded reference-volume fraction: sum(Vref_tet * indicator(D_tet >= 0.5)) / sum(Vref_tet).

Across resolution pairs, both differences must independently be at most 0.02, and runs must match the mapped reference-volume denominator (relative tolerance 1e-6, absolute tolerance 1e-12 m^3), the 0.5 threshold, source/law identity, and mapped tet coverage. The threshold is inclusive. The existing area-weighted observable remains separate. Geometric crack/open area remains unavailable; neither new observable claims to be geometric crack area or opening. Fracture-energy and orientation checks remain separate and unqualified by this change.

The final source-bound CPU build succeeded for numi-matter-cardboard-probe and numi-matter-cardboard-constituent-volume-damage-check. The stage-study Python suite passed 33 tests, and CTest passed the helper check. No native GPU or physical run was performed.

Final frozen source SHA-256: 9423e95363ffc62e2d11f7e8e59720c6911bc6f1870565aa68803acb6550bbcc.
Final source-bound build manifest SHA-256: 4744f3cc05631f52c141b3e733f391b09b677a59676789bc33414b2edad4ac61.
