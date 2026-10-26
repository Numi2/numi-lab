# P158: connected star and patch construction evidence

P158 supplements P157 with frozen, offline surface-construction branches from the 569–573 methods and a bounded regional search. The latest temporary candidate in this increment has 1,992 surface pairs, 0 contained pairs, and 0 self pairs across 185 complete banks and 237 anchors. It is unselected. P157’s selected main baseline remains 2,086 surface pairs.

## Measured branches

- The connected-star-569 search reduced its 2,086 input to 2,034 surface pairs (52 fewer) through 22 accepted patches. The exact full check reported 0 contained and 0 self pairs. Affine lineage regeneration preserved the candidate geometry.
- Round 041 continued from the 2,034 candidate, accepted five topology flips, and passed a fresh full check at 2,024 / 0 / 0. Its state remains candidate_selected=false.
- Connected-patch-570 run 002 applied one two-interior patch with an exact reduction of eight surface pairs, reaching 2,016 / 0 / 0. Its run-001 test attempt is retained with “Ran 0 tests / NO TESTS RAN”; run 002’s two focused unit tests passed.
- Connected-star-573 started from the 2,016 candidate and accepted ten patches, reducing surface pairs by 24 to 1,992 / 0 / 0. Its reports state coordinates and point IDs were unchanged; exact cached counts matched the original counter on each selected patch.
- Regional-patch-571 run 001 found no passing patch: zero operations were accepted after 11 exact triangle queries, and the input stayed at 1,992. This is a no-progress result, not a new full audit.

The eight-region witness recovery diagnostic is also retained as a negative result. It reports that the partition could not be recovered and no candidate was composed; no geometry audit was run for that attempt.

## Tests and limits

The package’s six frozen test modules pass together: 24 tests in 0.51 seconds. This is unit-test evidence only. It does not establish physical skin transfer, finite-element admission, completed anatomy, or clinical outcomes. All candidates in this increment are marked unselected; native admission is false and accepted physical steps are zero.
