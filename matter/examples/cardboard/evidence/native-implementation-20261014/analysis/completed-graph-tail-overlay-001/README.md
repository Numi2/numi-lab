# Completed graph-factor tail overlay

This modification-only patch follows `dynamic-grip-graph-overlay-001` on the frozen source-073 snapshot. It avoids reassembling selector-5 graph-patch matrices and refactoring them for an environment only when a later Newton-tail iteration has the existing nonlinear-completion latch and every patch has a valid previous factor.

The source-233 trace motivates the change: it contains 64 Newton iterations. Environment 1 is already a certified no-op at iteration 0 and stays one for all 64 records; environment 0 first becomes a certified no-op at iteration 39 and stays one for the remaining 25 records. The retained run took 122.40 seconds, completed one accepted step, and reports `physical_validation:false`. It predates this patch; it does not demonstrate a speedup.

Iteration zero, forced tangent-cache rebuilds, selected rich diagnostic captures, active environments, and invalid prior factors rebuild the graph matrix and factor. Mask validation, global AP/RAP/coarse setup, graph solve/apply, residual evaluation, Armijo, and final acceptance certificate remain on the existing path. No no-op shortcut bypasses the final certificate.

## Apply and verify

Start from the exact snapshot and apply the prerequisite dynamic-grip patch first, then `tail.patch`. The manifest binds the source files, both patches, and packaged outputs by SHA-256. The exact source-073 to prerequisite to tail-patch chain was applied in a fresh verification directory and the selected files matched byte for byte.

The completed-tail CPU predicate fixture passes 14 checks. It covers iteration zero, forced rebuild, diagnostic capture, active versus completed environments, valid and invalid factor state, and new-transaction recomputation. The existing graph-layout CPU regression also passes its 20 checks.

The Metal/runtime build and native same-protocol parity test have not been run by this overlay author. Root owns those checks. This is an optimization candidate, not a performance or physical qualification result.
