# P149: offline construction checkpoint

P149 extends P148's offline construction record. P148 at `a17f689a8d25a5c6e8d7c09cfe9865548148ec9d` recorded the 2409-surface-pair candidate `eadcd3838ec5bf4f2d72d45ba5d19c775e9fef4361f92ebea6f49e18d4bfe3a5` with zero contained pairs, zero self pairs, and 185 complete banks. This publication adds a fresh exact 2397 checkpoint followed by a connectivity-only topology pass that reached 2395 surface pairs with zero contained and zero self pairs across all 185 banks.

## Latest checkpoint

The package's latest construction checkpoint is candidate `ec3abc4da2bb84620c84d23457202dbdd54bea27f7478ac543124b6ded6938e6`, with exact report `surface-clearance-546/topology/round-009-from2397-chain-025/final/report.json`. Its status is `COMPLETE_EXACT_CONSTRUCTION_CHECKS`: 2395 surface pairs, 0 contained, 0 self, 185 complete banks, and 237 typed anchor rows. The candidate passed the recorded arithmetic and exact construction checks. It keeps the prior 2397 candidate's point coordinates bitwise unchanged; the chain pass changed face connectivity and reduced two crossing pairs. Round 009 contains one accepted pair-reducing connected flip followed by a no-op.

Round 008 independently recorded candidate `ca5cffae1a942b00c72a11288d33904eacb562c264160e96951f06b6035fa391` at 2397 / 0 / 0 / 185. Its full check reports maximum source distance 11.995121 mm, maximum repair step 0.152362 mm, and a passing area gate (minimum fixed-reference area ratio 0.100093569 against a 0.1 floor). Round 009 adds no coordinate movement and preserves those coordinates.

`verification/coordinate-change-summary.json` records a direct NPZ array comparison against P148's 2409 candidate: 2,341 of 20,889 coordinate rows differ and 18,548 are bitwise unchanged. It also records the changed-row bounds and bitwise checks for the node-source ownership, source displacement, typed anchors, boundary path, and face source-ancestry arrays. `source_O_faces` differs as expected from the face flips. This bounded comparison describes stored construction arrays only.

## Retained refinement attempts

`surface-clearance-546/refined-point-001/` preserves the separate, unselected point-continuation branch and its failed or inconclusive trials. That branch began from a separate 5765 / 1371 / 0 affine checkpoint, not P148's selected candidate. Its best fully checked soft-field backtrack reached 5560 surface / 1259 contained / 0 self across 185 banks. The source-compatible and multi-parent target attempts each produced 1,371 target rows. The fixed-endpoint projection diagnosis remained infeasible for the quality owner; the four-round endpoint batch remained unresolved with 141 → 99 → 96 → 95 operations. The initial point LP and scaled LP remain construction proposals; the final scaled-LP backtrack reached 5693 / 1291 / 0. Soft-field weights 0.03, 0.10, 0.30, 1.00, 3.00, and 10.00 are retained. At weight 1.00 the first exact check failed with 6,672 self pairs; later backtracking reached the 5560 / 1259 / 0 result. None of this branch replaces or improves the published 2395 checkpoint.

The inert coherent-depth adapter package keeps the updated source and its 11-pass unit-test receipt. It records no actual target-generation run. The finalizer source and its 13-pass unit-test receipt are also retained. These test results cover software behavior; they are not geometry or physical qualification.

Two large stdout files that mirror structured JSON reports are recorded with exact path, size, and SHA-256 in `omitted-payloads.json`; their corresponding full reports and all distinct logs/evidence remain included. No caches are included.

## Limits

All work here is offline construction evidence. Native admission is false and accepted physical steps are zero. No physical skin or fat transfer, FEM admission, visual inspection, motion, or clinical/anatomical qualification is established. The 45 pre-existing boundary edges remain open; these construction checks do not establish a closed skin. The P149 candidate is a measured construction checkpoint, not a physical anatomy result.
