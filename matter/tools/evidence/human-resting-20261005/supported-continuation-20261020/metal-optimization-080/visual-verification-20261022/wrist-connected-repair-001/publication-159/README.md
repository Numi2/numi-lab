# P159: bounded construction and retained-target evidence

P159 adds the completed 575–578 construction branches and the exact 577 topology pass. The latest candidate, `07b9d1186d026ed0342cd9a24629a7a9b323b7feb931da92d0cb3b07b4baa6e2`, passed full construction checks at **1,855 surface pairs, 0 contained pairs, and 0 self pairs** across 185 complete banks and 237 anchors. That is 137 fewer surface pairs than the P158 prior published construction checkpoint (1,992 / 0 / 0). The candidate remains unselected.

## Frozen branches

- **575 fit proposal:** candidate `45b40a7f…` is `CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED`. Its fit report says the exact global self audit was required but not run, and the all-185-bank point query was not run. It is not a passing construction result.
- **575 local retention:** full checks report 1,900 surface / 1 contained / 0 self pairs, so this branch fails the zero-containment target. The separate retained-regions candidate passes at 1,913 / 0 / 0, but remains unselected.
- **575 emitted-policy measurements:** the fit settings used a 1 mm trust radius and a 0.1 dimensionless edge-gradient factor. The existing admission limits stayed unchanged. After local retention, 2,452 inequality rows violated the 0.1 factor, with minimum slack −0.219814 mm; the existing 0.5 factor passed with zero violated rows. For the separate 1,896 candidate, 2,454 rows violated the 0.1 factor while the existing 0.5 factor still passed; its maximum Euclidean step was 0.882385 mm. These measurements do not establish a 0.1-factor pass.
- **578 witness target:** row 18892 was restored on the target branch while other target coordinates and non-coordinate payload fields were preserved. The target report remains pending full checks and makes no linear-feasibility claim. The v2 replay matched all payload fields bitwise. A separate checked retained-regions candidate passed full checks at 1,896 / 0 / 0 across 185 banks and 237 anchors; it remains unselected.
- **576 regional preflight:** retained as `PURE_PATCH_PREFLIGHT_DIAGNOSTIC_ONLY`; no geometry or solver ran.
- **577 harmonic topology:** 11 accepted patches reduced the checked 578 candidate from 1,896 to 1,855 surface pairs. The exact report has 0 contained and 0 self pairs, with unchanged coordinates. The candidate remains unselected.

## Tests, source pins, and limits

The included 577 unit-test module passes: 3 tests in 0.12 seconds. The compression receipt records original and archive hashes, sizes, modes, timestamps, and successful restoration checks for the two archived 575 text mirrors. Earlier package test attempts are preserved, including dependency collection failures and a seven-test run made before the unledgered 575 replay-v1 source was excluded. That historical run is not the final package test gate. P159 intentionally omits the old `replay_raw_direction575.py` v1 helper pending its exact source ledger; the original worktree source remains preserved.

All evidence is offline construction or diagnostic work. `candidate_selected=false`, `native_admission=false`, and `accepted_physical_steps=0`. This package does not establish physical skin transfer, finite-element admission, completed anatomy, or clinical outcomes.
