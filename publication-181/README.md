# P181 draft: three-bank construction experiments from 663

Review draft only. The input is the previously published 663/0/0 construction checkpoint, pinned to P179 commit `d18de33e4939b60f66cc60b7023ce8bd419b4269` and included under `inputs/checkpoint-663/`.

## Targets and coverage

The target grouping has 359 exact pair identities across five edge-connected patches: 43/16 has 153, 65/99 has 86, and 109/114 has 120. Combined-fit coverage verifies all 359 identities, 1,077 guard corners, 502 positive movable soft-corner rows, and a positive soft row for every target pair. The unweighted selector is separate: it selected axes for 648 of 663 crossings and used the pinned 637 fallback for 15. These are row/objective coverage receipts, not clearance results.

## Fit, retention, and continuation results

- **Unweighted:** candidate `e0486cbc…eeba05d6` failed exact self checking with 15 pairs; its 545 check stopped before containment and surface checks. Self-safe retention kept 343 changed coordinates over 22 clean components, preserved the 663 non-point payload, and passed at 665 surface pairs, 0 contained, 0 self. It is +2 overall; all three requested banks have no identity changes. Not promoted.
- **Bank-balanced:** candidate `991bcffd…d4932cb` also had 15 exact self pairs. The 582 retention passed at 689 pairs, 0 contained, 0 self. Relative to 663, 43/16 is +16, 109/114 is +6, and 65/99 is unchanged; total is +26. Not promoted.
- **Zero-step seed union:** 15 new exact self witnesses were added to the earlier 32,125-pair seed set, producing 32,140 pairs without coordinate movement. The weighted retry failed before solving because the selector and prepared self-seed mappings disagreed.
- **661 small-fraction candidate:** a 0.03125 fraction of component 9106 changed 110 coordinates (maximum step 0.02418 mm). The full 545 report and absolute-path 130 audit report 661 surface pairs, 0 contained, 0 self, across 185 banks; strict 546 finalization references the same candidate. Against 663, 43/16 removed `(18990, 140)` and `(18995, 140)`; 65/99 and 109/114 identities are unchanged. This is a construction-only candidate, not physical or native acceptance. Reverse and extrapolation screens recorded no additional target-bank improvement.

Exact per-bank identity changes are in `evidence/three-bank-exact-pair-deltas.json` and `evidence/661-three-bank-exact-pair-deltas.json`. The 661 branch retains its screens, failed relative-path preparation attempt, corrected prepared-002 receipts, full 545 report, absolute 130 audit, and final-absolute state. P179's exact retention-report bytes are preserved as `.json.raw`; a separately labeled parseable copy and transformation note document its trailing literal backslash-n bytes.

## Limits

All evidence is construction-only: `native_admission=false` and `accepted_physical_steps=0`. No physical skin/fat/FEM admission, closed-skin claim, or completed-anatomy claim is made. The pre-existing 45 surface boundary edges and two loops are outside this work’s closure claim.
