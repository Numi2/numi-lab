# Native vascular-budget 027 — early contact failure

Execution exited 1 after 77.322 s. The invocation requested 32 steps at 125 µs (4 ms total), but only 5 coupled rows were accepted (0.625 ms). The requested horizon did not complete.

At segment/root 5, Matter reported code 6 at object 19, failing slot/index 2941, with 0 microsteps and diagnostics [8.996410727490911e-9, -1.0006338341467594e-13, 1, 0]. This is the recorded contact failure; it is not a completed 64-step solver-budget trial. Full stderr and selected stdout are retained. Stderr contains no vascular_iterate trace rows, so no root-20 capture exists in this run.

All 13 invocation file assets were rehashed and match native-vascular-row23-026. Run source/executable/metallib/library hashes are bound in source-binary-binding.json; the physical-kernel baseline comparison is handled separately by verification-027. The output pack and large geometry assets were not copied.
