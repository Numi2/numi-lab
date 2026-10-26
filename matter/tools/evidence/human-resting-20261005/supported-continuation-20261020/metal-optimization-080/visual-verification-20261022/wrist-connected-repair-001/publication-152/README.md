# P152: offline construction progress at 2,200 surface pairs

P152 records a static, offline construction checkpoint. The round-020 candidate passed the recorded exact construction checks at **2,200 surface pairs, 0 contained point/bank pairs, 0 self pairs, 185 complete banks, and 237 typed anchor rows**. Candidate SHA-256: `10754c967d1319051319db1d01472cb559cba4d2dafdcda24fc288ce56998366`. Its owner state keeps `candidate_selected: false`. The exact reports record `native_admission: false` and `accepted_physical_steps: 0`.

The previous published endpoint in P151 was 2,220/0/0. The bounded progression captured here begins with the separately checked round-016 baseline at 2,202/0/0. Coherent-exit round 017 reached 2,228 surface pairs but retained 1 contained pair. Coherent-witness round 018 passed at 2,226/0/0 but remained unselected. Pair-SAT round 019 reached 2,216 surface pairs with 1 contained pair. The retained-component round 020 then passed at 2,200/0/0. Each result and its candidate/report identity remain in the per-round directory.

## Negative and diagnostic evidence

The 019 raw-direction self audit is separate from the 019 geometry candidate: it stopped at the surface gate and recorded 297 surface self pairs. It is retained as a raw-direction diagnostic, not as a geometry result. The 017 and 019 containment failures, as well as the passing but unselected 018 result, remain explicit in `negative-trials.json` and their original reports, candidate arrays, exact checks, witnesses, and launch/execution receipts.

The complete round-016 topology attempt and all frozen 017–020 evidence are included. Large `fit-stage.log` stdout mirrors above 10 MB are listed by original path, SHA-256, and byte count in `omitted-payloads.json`; the structured fit reports, candidates, exact check reports, audit witnesses, source pins, and launch/execution receipts remain included. Nine earlier large text mirrors are preserved losslessly as `.zst` files with an exact restore receipt and tested restore helper.

## Source and test receipts

The 550 coherent-exit helper suite passed **19 tests**; the 551 pair-SAT routing suite passed **9 tests** in the fresh runs recorded under `tested-sources/`. Current helper/test sources, historical source snapshots, and exact source-pin mappings are included. The 552 regional-retention source and round-020 outputs are included; round-020's complete full-check report is the actual exact report used for its passing result.

## Limits

This package reports static construction checks only. It does not include visual inspection or a native run, and it does not establish physical skin/fat transfer, FEM/contact admission, biological or clinical performance, or completed anatomy. A zero contained-pair count is specific to the recorded construction gate and must not be read as physical qualification. The selected owner state is unchanged.
