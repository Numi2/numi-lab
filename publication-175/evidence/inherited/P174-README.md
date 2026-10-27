# P174: 764-pair SAT-routed construction checkpoint

P174 extends P173 (commit 5385f2e1ab11139ed56babdd34b9565dd2e11eeb; manifest SHA-256 c7834936fcc8a9bb46d2982cb4d6ad00f7fe16db016288fff28a2abc21f4d30a). It packages the frozen 636-v2 bound method, the 637 finite-pair SAT routing run, its failed and non-improving branches, and the strict 764 result produced by the 631-v2 component merit pass. The initial fit report is stored as a zstd archive with a verified restore receipt; the original local report was retained.

## Measured construction result

The input was P173 candidate 47ec88d0442188e5e71cb8edf4e6adc05af7f13dbac965deff99ac2a253a8220 (766 surface pairs, 0 contained points, 0 self pairs). The 637 first-fit candidate 176fa94b9bfc677be0fae87dfa98cce93e9141c9415ca5ebc4bf86322bb27380 failed its exact surface self-intersection gate with 10 self pairs. Its 130 audit witness and report are retained.

A later 631-v2 merit selection produced candidate d0d8ba18026d37a1ea64a247e09e4f6e972db0b157d64ad075851ab76c1c4ca4. The matched full-check report is COMPLETE_EXACT_CONSTRUCTION_CHECKS: 764 surface pairs, 0 contained pairs, 0 self pairs, 185 complete banks, and 237 anchors. The surface count improved by 2 from the 766 input. The final state explicitly says candidate_selected=false; P174 does not promote it to an accepted physical result. The merit full-check stage generated the exact reports; the later finalizer reused those report hashes rather than rerunning the audits.

The separate 582 retention branch remains as negative evidence: 785 surface pairs, 0 contained, 0 self, and construction_progress=false. It did not replace the 764 branch.

## SAT routing and bounds

The 637 adapter enumerated signed finite-triangle SAT axes for all 766 input pairs. Under 636-v2 component boxes, 760 pairs had an individually box-feasible selected axis; 6 have explicit infeasibility witnesses. The routing report states that shared-node axis compatibility and whole-bank clearance were not certified. The exact full checks on the 764 candidate remain the governing construction evidence.

636-v2 retains the 4 mm trust limit and 12 mm source cap while widening the old 408 component cube asymmetrically inside those same balls. On the 766 preparation it reports 8,166 support nodes, 17 anchors on support, 1,446 expanded components across 482 nodes, maximum box step 4 mm, and maximum source distance 11.999993588392693 mm. These are selector bounds, not a clearance certificate.

## Tests and limits

The included pure unit suites for 636-v2, 637, and 631-v2 passed: 20 tests. No solver or geometry audit was run as part of P174 packaging.

All evidence is offline and construction-only: native_admission=false and accepted_physical_steps=0. The result still has 764 surface pairs. The 45 pre-existing head boundary edges and two loops remain; the skin is not closed. No tissue-physics, physical, biological, or anatomy-completion claim is made. The P173 baseline artifacts and inherited source-pin index are referenced by immutable publication path and hash rather than duplicated.
