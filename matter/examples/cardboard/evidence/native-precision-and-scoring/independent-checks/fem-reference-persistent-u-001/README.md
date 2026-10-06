# FEM reference-displacement precision check

These files preserve the bounded validation performed while developing the opt-in persistent FEM displacement state (`NMFEMNodeStateGPU.referenceDisplacementAndMode`, Matter ABI 40).

- `metal-check.console-transcript.log` is the console output captured in the completed Metal test tool result for the listed `numi-matter-fem-reference-check` invocation. It is a transcript copy, not stdout redirected by the original invocation. The check completed successfully on Apple M4. The stale-package ABI regression was added afterward, so the GPU transcript reports 99 displacement checks; it does not include that CPU-only regression.
- `cpu-check.stdout.log` is raw stdout from the later CPU-only compiler/package run and includes the stale ABI 39 rejection regression (100 displacement checks).

These checks exercise the mechanics/API path and numerical arithmetic only. The synthetic cardiac materials and geometry are controls; the output explicitly does not claim source-calibrated cardiac or cardboard qualification. The parent task retains source-bound cardboard experiment receipts separately under this evidence root.
