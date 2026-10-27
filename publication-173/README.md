# P173: 766-pair connected-surface construction checkpoint

P173 is a narrow evidence package based on P172 (commit 43ea91a21d376c4bda8811803e946820d048e10e, manifest SHA-256 938965267b9dda5eedc2230bd5440d9e10b1e28f3ff00d1526a202ac920fa1d9). It preserves one completed strict topology result and its matched target preparation and feasibility diagnostics. The 766 result is not marked selected in its state file; P173 makes no selection or physical-admission claim.

## Construction result

connected-topology-580/run-009-from772 emitted candidate SHA-256 47ec88d0442188e5e71cb8edf4e6adc05af7f13dbac965deff99ac2a253a8220. Its fresh full checks report 766 surface pairs, 0 contained points, 0 self pairs, 185 complete banks and 237 typed anchors. The check records construction_progress=true against the 772 input candidate (SHA-256 668ae678559450363eaaa99a1bb0d6d3323cb1fde0056104b664d85836facbdd), while state.json records candidate_selected=false. P172's published checkpoints remain in the ancestry.

An array-level comparison found point coordinates bitwise unchanged from the 772 input; 19 face rows changed (rows are listed in the comparison report), involving 17 vertices within the reported world-coordinate bounds. This is a stored-array comparison only. It is not a separate geometry audit, fit, back-verification, or physical admission.

## Matched preparation and limits

The 766-pair preparation has exact audit-to-target pair identity (766/766), 185 endpoint banks, and fit_not_run=true. The direction-target checker status is PASS_CERTIFIED_DIRECTION_TARGETS_ONLY. Its actual minimum joint-current cosine is 0.04926666417780694 and its actual minimum joint-source cosine is 0.04939142921741552; both are below 0.05. The checker status does not certify those values as meeting 0.05, and no fit was run. Five targets are marked as source-only exceptions. The separate positive-anchored-corners diagnostic found zero such corners.

Independent per-node feasibility reports record 132 fixed-bone outward-plane node lower-bound failures and 563 single-row witnesses across 113 nodes. The finite-pair SAT-plane diagnostic reports 101 node lower-bound failures and 99 single-row witnesses across 14 nodes; its per-axis availability check has 50 pairs with no box-feasible candidate axis. These diagnostics cover their stated row formulations and conservative component trust/source cubes. They omit coupled constraints and do not prove global or physical impossibility.

The included 633 objective audit is arithmetic-only and is tied to the earlier 782 candidate. It records a missing emitted-endpoint check in that 603 path; it is not a fit or geometry audit for the 766 candidate.

## Boundaries

All evidence is offline and construction-only: native_admission=false, accepted_physical_steps=0, and no anatomy or tissue-physics qualification is claimed. The 45 pre-existing head boundary edges and two loops remain; the skin is not closed. Directional-bounds v2/636 work and the mutable 632 solver probe are excluded.

The source-pin index maps exact hashes to this package, verified ancestor publications, or local unbundled inputs. It records external dependencies by immutable path and hash without duplicating large historical inputs.
