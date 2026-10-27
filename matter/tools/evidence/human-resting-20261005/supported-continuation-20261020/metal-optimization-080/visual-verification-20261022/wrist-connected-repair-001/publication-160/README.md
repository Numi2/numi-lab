# Publication 160: offline construction progress

This snapshot records completed construction experiments and failures through the 1697 surface-crossing checkpoint. It adds the three 580 topology runs, the 581 failed global-retention attempt and passing retained-regions result, both 582 trials, the completed 583 v1 replay plus a v2 path-fix unit suite, 584 watchdog validation only, a 585 unit-tested pilot array, and a lossless recovery bundle for 49 old reports.

## Measured construction checkpoints

- 580 run 001: 1731 surface crossings, 0 contained, 0 self; 185/185 banks and 237 anchors.
- 580 run 002: 1717 / 0 / 0; 185/185 banks and 237 anchors.
- 580 run 003: 1697 / 0 / 0; 185/185 banks and 237 anchors. This is the lowest measured checkpoint in this snapshot. The exact checks pair the complete audit-130 result (20,889 skin points) with complete typed-bank surface reports. The run state still records `candidate_selected=false`.
- 581's first global-retention candidate failed with 1730 surface crossings, 3 contained pairs, and 0 self pairs. The later retained-regions candidate passed at 1705 / 0 / 0, but does not replace the 1697 checkpoint.
- 582 fitcandidate trial passed 1937 / 0 / 0. The raw002 trial passed 1956 / 0 / 0. Both are unselected.
- 583 v1 replay passed 1707 / 0 / 0. The v2 change is only a path fix verified by six unit tests; no v2 geometry run is claimed.

All these exact construction checks report 185 complete banks and 237 typed anchors. The 585 pilot inserts eight edge midpoints into the 1697-parent array. Output crossings were not measured in this snapshot; the pilot report explicitly makes no containment or clearance claim, and no full check was run on the output.

## Watchdog and tests

The first 584 watchdog validation failed with a macOS process-group termination permission error. The corrected v2 validation passed bounded child success, timeout termination of child and grandchild (return code 124), and strict selection of the checked 1705 state, with an empty live-process probe. This validates watchdog behavior only. The active 584 geometry output tree is excluded, so this publication does not claim a 584 geometry run.

The retained tests show three 580 runs with eight tests each, four 582 tests, six combined 583 v1/v2 tests, and eight 585 tests. `tests/test-run-receipt.json` records the exact test paths and outputs.

## Limits

This is offline static construction and software-test evidence. It admits no native simulation or physical step (`native_admission=false`, `accepted_physical_steps=0` throughout). It does not establish skin/fat transfer, tissue motion, physical qualification, or anatomical/clinical completion. The zero-surface-crossing goal remains open. 584 outputs still in progress and all later experiments are excluded.

The `cold-output-recovery-545-001` folder contains 49 individually compressed reports, logs, and target JSONL files. Each matched an exact member from published P142/P143 after fresh streamed GitHub downloads. The bundle includes the mode/mtime-aware restore helper and receipt; its 98 retired local paths can be reconstructed from the payloads.
