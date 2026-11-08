# Inconclusive: control loaded previous owner

Control exited 2 at 00:53:01.233425 UTC on 8 November 2026, after pre-submit rejection status 15 and a subsequent unpublished status 17; the mixed arm was not run. The declared binary/shader/asset hashes matched both before and after. However, inspection after this failure revealed that the unchanged probe embeds an absolute LC_RPATH to the original staged-publication library, rather than the copied guarded library declared in the plan. Its actual path and current SHA-256 are retained separately. The pre/post hashes did not include that rpath target, so its identity during execution is not independently certified. No upload-guard logs appeared. This run is inconclusive about the new guard, mixed outcomes and atomicity.

A separate corrected gate must explicitly fix and pin the executable rpath and actual linked library before dispatch. This failed run and its initial mistaken owner declaration remain unchanged.
