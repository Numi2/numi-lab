# Bilateral knee muscle-delivery response

The earlier eight-muscle knee cohort accepted its mechanics steps but failed delivery: right short-head biceps (343) was already at activation 1.0. That failure is preserved. Add a separate symmetric six-muscle cohort (bilateral long-head biceps, semimembranosus, semitendinosus), omitting the short-head pair 343/383. This changes the experiment selection, not baseline muscle forces, excitations, anatomy, materials, constraints or acceptance gates.

Fresh-process baseline and +0.1 selected-excitation runs use the same assets, exact initial q/v, 125 us timestep and eight accepted steps (1 ms). Both endpoint and six-of-six delivery gates pass. Right/left knee angle differences are +2.43766099e-5 / +2.13154854e-5 rad. Loaded J is 0.965088427–1.232343584. The maximum endpoint FEM difference is 8.22689 um across 14202 nodes / 54133 tetrahedra. Exact results and hashes are in assessment/summary receipts.

This is a release-initialization mechanics test, not settled supported rest, range of motion, anatomy-repair, Brain/physiology or performance qualification. The original clipped-route failure is not superseded by this distinct subset test. Both complete native logs are independently gzip round-trip verified; no failed evidence or plain logs were deleted. Identical scene packs were replaced by verified independent APFS clones in cleanup064.

Strict existing runner self-tests pass, including clipped-route rejection, changed-ID rejection and initial-state mismatch rejection. All execution was on SSH Numi; one native GPU job at a time.
