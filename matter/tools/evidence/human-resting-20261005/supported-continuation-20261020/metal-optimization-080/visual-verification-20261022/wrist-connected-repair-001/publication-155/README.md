# P155: offline construction evidence through the 558–561 work

P155 retains a completed, **unselected point-continuation branch** at **4,920 surface pairs, 1,184 contained point/bank pairs, 0 self pairs, 185 complete banks, and 237 typed anchors**. Candidate SHA-256: `ddb0c0a51f5518a3bef519824a9aca79d74ee85eb0640ab43187742e5936fefc`. The full 545 checker and existing finalizer both pass their construction-only checks. Its state explicitly records `candidate_selected: false`, `native_admission: false`, and `accepted_physical_steps: 0`.

The published selected main baseline remains P154 at **2,180 surface pairs / 0 contained / 0 self**, candidate `6cebc9e4d4d2cda4afbde1670f105b2b56b7e7f74de40cbb5f51b513e6bb4c7d` (P154 `daf88041dc138c75737ac3deb3147ae16b12e55b`). P155 does not replace or relabel that best state. The 558 refinement is a separate upstream experiment: its candidate and state are hash-matched at 2,188 reported crossing bank/triangle pairs; it is also unselected and construction-only. The 031 backtrack reduced its own baseline by 153 surface pairs and 64 contained pairs, but the result remains materially behind the selected 2,180 surface baseline.

## Retained trials and corrections

- The 558 constraints preparation retains the first refresh attempt that supplied the wrong ancestry parent (2.2306 mm mismatch) and the corrected refresh that uses the refinement's exact SHA-pinned immediate parent. Point-depth and multi-parent surface targets, their complete reports and target records, corrected guards, seeds, launch/execution receipts, and the unselected refinement are preserved.
- The 559 swept-preseed trial stopped at the 500,000 candidate-pair cap with 1,134,175 pairs unclassified, 0 classified eligible, and a 240-second classification budget. Its report explicitly says no exact intersection audit was run by that tool. Potential/unclassified pair arrays, command metadata, owner source snapshot and run log remain available.
- The 031 guarded backtrack and every retained pass result are included. Its endpoint is **4,920 / 1,184 / 0**, with maximum source distance 11.99653 mm under the recorded 12 mm cap, maximum step 1.01250 mm, and minimum edge slack 0.00174 mm. The exact 545 report is retained alongside point and surface audit outputs. No native or physical step was accepted.
- The 560 joint objective adapter's focused suite passed **10 tests**. The 561 prolongation mapper suites passed **8 (v1)** and **12 (v2)** tests. Both frozen mapping runs report 3,321 mapped and 129 unmapped witnesses out of 3,450, with 22,723 merged rows; these are partial construction seeds, not complete swept coverage. The additional swept-backtrack helper suite passed 6 tests.
- Round 032 remains a failed trial: the LP returned `LP_NO_PROPOSAL` after 21.77 s; component 7 exhausted its 6.8 s per-stage allocation under the fixed 120 s total. The v1 coordinator then attempted a checker with no candidate and failed before geometry checking. The v2 558 coordinator source with a stop-before-checker guard is included; P155 does not claim a separate round-032 v2 execution. The active 562/round033 work is outside this snapshot.

## Source and input identity

Launch, execution, input, candidate, target, and report hashes are retained in the original records. The exact 559 swept-preseed owner source is included, as are the 558/560 and v1/v2 mapper sources and frozen source snapshots. `source-pin-index.json` lists all discoverable Python source pins and their exact snapshots included in P155. The active budget-adapter folder and future round033 files are excluded.

## Limits

This is offline static construction and software/test evidence only. It does not establish physical skin/fat transfer, tissue motion, native admission, accepted physical steps, or anatomical/clinical completion. The zero-surface-crossing goal remains open. The retained 4,920 endpoint is not the selected main candidate and has nonzero contained pairs.
