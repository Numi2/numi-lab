# P168 - connected-surface construction checkpoint

This review draft records completed work since P167. P167's verified push receipt is included for lineage. At the P168 packaging cutoff, no Git ref update had occurred and no source retirement was performed while staging this package.

## In-scope measured results

- **Strict 622 checkpoint:** candidate ea65e93df9ba3d7617450bc7ff2f6bbd56529be354d772a731e98ead686d10e4 completed exact construction checks at **1168 surface / 0 contained / 0 self pairs** across 185 banks and 237 anchors. This is the best passing construction checkpoint in this scope, improving on P167's 1231 surface pairs.
- **619 run 002 failure and salvage:** the coordinator ended with status STOPPED and error: ValueError: 603 did not emit proposal from unchanged round baseline. The retained salvage candidate ed78fb1f0f68072414a1190f92bbccf32602afa97856ab41d4a7aad173f5fd4c reached **1173 surface / 1 contained / 0 self pairs**. Its exact full check is retained as a partial result; it is not an accepted replacement.
- **623 negative result:** candidate d077cac6033606d5be9249677f757a11bc4b4be8fac89c9f5da160ebbbbadf4b completed checks at **1206 / 2 / 0** and did not satisfy construction progress. The failed arithmetic and proposal evidence remain included.
- **621 v1 refinement/normal failure:** the v1 branch proposed 113 new midpoint rows. The strict source/current normal gate failed with a retained disagreement witness. The separate 54-row direction feasibility diagnostic reports 50 feasible and 4 infeasible rows: 20906, 20934, 20958, 20963. The four rows remain explicit evidence; they are not hidden or treated as accepted.
- **621 v2 construction and 624 targets:** v2 retains the construction/audit/depth/624 outputs only; active matched-fit work is excluded. It produced candidate 399c59ed50c689840b1d659bc83fa3205e8dfeadfccc81db1cee6c3c5d654c2e after recording the infeasible midpoint rows. The 624 target report is PASS_FIRST_UNION_EXIT_TARGETS_ONLY for 50 inside rows, using exact first-union endpoints across all 185 banks. This is target construction evidence only, not a fitted or admitted geometry result.
- **624 helper correction:** both source revisions and test files are included. The v2 serializer fix preserves NumPy/path diagnostics in both success and failure JSON. The focused v1+v2 suite passed 22 tests; the test receipt pins commands and source hashes.
- **625 sparse transfer:** the proposal remains unadmitted, and its arithmetic result failed with the full Float32 area/orientation gate. New retention/full-check work is excluded.
- **619 v5:** tested source and test code are included. The v5 coordinator's next execution is reserved for a later publication.

The 619 run-002 full fit report is stored as a lossless zstd payload with exact original hash, size, mode, and mtime, plus a successful fresh-destination restore proof. Its plaintext source remains intact.

## Historical cleanup and lineage

The included cold-output retirement plan/receipt records 92 inodes, 184 aliases, and 702197760 reclaimed allocated bytes after its independent restoration checks. This is historical evidence; P168 performs no cleanup. The P167 push receipt pins commit f21e352a79c7dcf14b671c969138afffdea278f3 and manifest 0b192fbbabc1b96a2568fca6ee1420bcf5c43f15760305b61b17df9ef2890981.

## Scope limits

All reported work is offline construction evidence: native_admission=false and accepted_physical_steps=0. No physical or biological admission is claimed. Active run-003, 621-v2 matched-fit preparation, 625 retention/full-check, connected-refined-626, and later work are excluded. The manifest and scope decision record the packaging-time review state; the P168 push receipt records any subsequent publication status.
