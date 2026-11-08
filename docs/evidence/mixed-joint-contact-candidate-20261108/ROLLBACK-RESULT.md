# Forced rejection after a joint solve

The injected shader changed the island status to failure after step 10 had executed the candidate joint solve and velocity update. The clean ten-step run accepted all ten; the eleven-step injected run accepted ten and rejected the final step with status 11. All three rejected-step contact rows retained joint-update markers. The first ten captured steps matched exactly, and the rejected step preserved q/v and the full captured rod-node records.

The actual final byte strings for q, v, scene bodies, rod nodes and rod edges were identical between the clean ten-step and injected eleven-step runs; the clean eleven-step negative control differed. The raw byte strings, SHA-256 hashes and sizes, per-step outputs, source and binary hashes, build commands, and capture/noncapture and repeat checks are retained in rollback/. The first probe build failed due to an incorrectly quoted JSON-output string; its compiler output is preserved before the corrected build.

This establishes restoration of the listed exposed owners for this injected solver-failure fixture. It does not observe hidden contact caches, external policy or material owners, or a real live retry after the failed step. It is not a general transaction proof and does not qualify physical forces. The joint shader remains an isolated research candidate.
