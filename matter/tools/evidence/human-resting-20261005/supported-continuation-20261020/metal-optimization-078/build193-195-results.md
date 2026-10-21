# Builds193–195: measured conditional graph work and rejected Schur trial
All execution on SSH numi, Apple M6, Metal API validation enabled. No visual verification.

## Implemented and tested
Opt-in NM_FGMRES_SKIP_COMPLETED_GRAPH_WORK=1 prepares separate GPU-authored indirect ranges for graph Schwarz patch solves and node application. A healthy finite completion certificate suppresses both; active, unhealthy, deferred and nonfinite states retain existing work. The actual patch and node domains remain distinct. The prefix ICB path remains independent. Mandatory finalization, true-residual certification and physical accept/reject gates remain outside this omission.
Integration review repaired missing dispatch-buffer allocation, declaration scope and the actual indirect-dispatch ABI. Default remains off; strict settings accept only0/1.
Final native fixture:146 checks PASS, including949 patch groups versus56 groups for14202 nodes, multi-environment fallback and untouched completed scratch. Existing22 source contracts PASS. Final Schur CPU fixture29 checks PASS.
Generic initialization now surfaces upstream Schur partition failures and rejects incomplete patch descriptors before scratch sizing.

## Three matched full-scene pairs
Frozen194;3 roots per run; alternating order; prefix preconditioner ICB disabled in both arms to exercise the changed direct path.
All17 endpoint groups match EXACTLY in all3 pairs.
Post-first throughput ratios(control/candidate time):
0.9983893,0.9991124,0.9966234; median0.9983893, approximately0.16% slower. NO demonstrated speed gain.
This rules out graph threadgroup omission alone as a useful whole-scene optimization here. It does not rule out suppressing a larger command graph.
Final195 three-root integration also accepts and matches194 exactly after removing the irrelevant Schur experiment and extending fixtures.
These are bounded startup checks with validation and output costs, not settled throughput, native presentation qualification or >=0.1x performance.

## Schur experiment stopped for a specific reason
The candidate attempted to merge same-object authored preconditioner region seams without changing the physical operator. Its CPU oracle passed37 checks, but the full Human rejected before simulation. Earlier diagnostics misleadingly reported compact scratch capacity.
All54133 accepted Human tetrahedra have region bits[10:8]=0. There are no eligible authored region seams to merge. Early return left empty descriptors while the caller checked the other error field, masking that reason.
The unused seam-merge option and helper were removed from the shipping source. Frozen193/194 source closures, failed invocations and logs preserve it. Scratch limits were not increased and no restriction was bypassed.
The successfully executed193 common-clock Schur control took15.147263292s body time for one root; it is not an acceleration result or a qualified clock change.

## Resources and remaining work
Seven finished replay files were SHA-verified identical and hardlinked to an existing retained copy, preserving all paths. Available space increased13187968->15218816KiB, about14.5GiB.
0.1x is NOT achieved. Larger conditional solver graphs, supported tissue initialization/contact convergence, model-reduction fidelity and qualified timesteps remain unresolved. No long qualification run is justified by these startup results.
Archive includes exact source closures/provenance, invocations, failed trials, logs, comparisons and cleanup receipt. Every member was independently reopened and SHA-verified.
