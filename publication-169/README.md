# P169 - connected-surface construction checkpoint

This review package adds completed construction evidence after P168. P168 is the published parent at commit 928bbd8599c11a9a58ed4121a64c05c815083301; its manifest SHA-256 is e841bb1e631409b4476578c2f53829e1bcdd46c49e35d1432ab0d5b1bde25140.

## In-scope results

- **Contained retention 629:** the retained restoration after the 1093/1 round reaches **1088 surface / 0 contained / 0 self pairs**, across 185 complete banks and 237 anchors. Candidate SHA-256: 21f3e30ad3ca7b13639696299bc35c21ee567ee97766c8c3162799369b57327e. It is the best passing construction checkpoint in P169, improving on P168's 1168 surface pairs and 619 run 003's 1089.
- **619 run 003 (v5 coordinator):** round 001 passed exact checks at **1089 / 0 / 0**. Round 002 reached **1093 / 1 / 0**; strict finalization rejected nonzero containment and the coordinator stopped. The passing round is retained, but the overall run is recorded as STOPPED. Four large fit reports are archived losslessly with exact original SHA-256, size, mode, and mtime plus successful fresh-destination restoration proof. Plaintext source reports remain intact.
- **580 run 004:** exact full checks and strict finalization pass at **1095 / 0 / 0**, across 185 banks and 237 anchors. Candidate coordinates are unchanged and affine ancestry was refreshed.
- **Component selection 627:** the initial empty-directory failure is noted, but no original invocation receipt is available. Run 002 passes exact full checks and strict finalization at **1149 / 0 / 0**. Candidate SHA-256: 1f6109634173e156a976bb104c656a375b9c15dacc2bcc3c48b042dafbacdb11. The root reproduction records identical candidate SHA and bit-identical NPZ arrays; it is reproducibility evidence, not the original invocation receipt.
- **Component selection 627 run 004 and field reuse 628:** these frozen diagnostics are retained. The run-004 predicted 1084 candidate failed the full Float32 area/orientation gate, including the latest-629 check on 13 faces. Field reuse 628 retained arithmetic-only comparisons and did not produce an admitted candidate.
- **619 v6:** the frozen v6 coordinator and its test source are included without an execution claim. The included run 003 used the v5 coordinator, whose source and test are pinned by published P168.
- **Sparse transfer 625:** the initial retention result is **1178 / 2 / 0**. A restored result passes with **1169 / 0 / 0**. It remains a negative comparative result against the 1088 checkpoint and is not promoted.
- **Refined branch 626:** the raw fit recorded 698 self pairs. A cut attempt ended with LP_NO_PROPOSAL; duplicate-slot failure and corrected relaunch receipts are retained. Its salvage continuation reached **1261 / 50 / 0**, and remains temporary/unselected.

P168's exact push receipt is retained for lineage. The source-pin index pins the new 627 and 619 v6 source/test files. Other helpers are tied to published P168 sources and launch provenance.

## Previously published output recovery

The cleanup evidence records ten old fit reports across 20 aliases, 420860302 logical bytes, and 420880384 unique allocated bytes retired after the root reviewed the plan and independently verified fresh filesystem restorations from GitHub. This package contains the plan, review, recovery, and execution receipts; P169 itself deletes no source data.

## Scope limits

These are offline construction checks only: native_admission=false and accepted_physical_steps=0. No physical or biological admission is claimed. At the packaging cutoff, 619 run 004 and later work, 627 run 005 and later, component0-crop-628, and later topology work are outside this frozen scope. P169 is a review draft; package creation did not update a Git ref.
