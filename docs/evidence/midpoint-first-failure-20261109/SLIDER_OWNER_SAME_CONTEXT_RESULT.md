# Same-context slider-owner rejection and retry

The MetalWorld-only midpoint test was tightened to reuse one actual
MetalWorld context for its accepted baseline, late rejected step and retry.
The late rejection returned the expected GPU failure category; the restored
nonzero initial q and v matched byte-for-byte, and retrying in that same
context reproduced the accepted q and v byte-for-byte. The eight analytic
position and velocity checks and the topology, effort, missing-stage and
competing-publisher guards also passed.

The output is `slider-owner-step-006.json`, the failure-category log is
`slider-owner-step-006.log`, and the rebuilt source and native artifacts
are hashed in `slider-owner-step-006.sha256`. The focused CTest passed.
This is still one owner with an analytically supplied stage increment; it
does not exercise Matter, contact work, or a whole-environment rollback.
