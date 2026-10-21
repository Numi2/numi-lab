# Rank-safe residual refinement
All execution: SSH numi. No visual verification.

The compiler can now promote an entire rank-deficient anatomical affine block to exact fine coordinates during explicitly requested residual refinement. The block is retired, source-node/tetrahedron ancestry remains intact, and the existing coordinate/memory/reduction budgets are checked again. Default construction still rejects deficient blocks. CPU tests verify default rejection, whole-region promotion, source coverage, mass/force adjoints, anchors and over-budget rejection.

The larger bounded witness selection retains all605 candidates from one captured Newton-cycle record (no cycle aggregation), rather than the old128 global cap. Source163 rebuilds successfully and both the matched full and refined hybrid Human accept one125-us root. Process times are25.3912/30.2862s, respectively: no speedup is established.

The exact-mass rank repair retires two affine regions (24 modes) and moves612 additional nodes to fine. The hybrid has5099 selected free-node rows,1848 q coordinates and26499 fine xyz coordinates:28347 total versus41796 full (1.474x). Source IDs, masses, connectivity and rest coordinates compare exactly.

Anchor-force error improves from the prior3.945% to1.167567% (reference0.84403759N, candidate0.85065748N, absolute L1 difference0.00985471N). It still fails the1% gate. Regional force/strain checks also fail; no reduced-model qualification is claimed. The comparison records which force channels and source-region labels are absent, and does not turn absent strain-energy evidence into a pass.

Source163's frozen source closure, exact plans, witness identity, compiler/native logs, endpoints and comparison accompany this increment. Further refinement must use a fresh valid single-cycle witness. The0.1x target remains unmet.
