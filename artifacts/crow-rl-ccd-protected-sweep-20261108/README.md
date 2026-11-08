# Protected first-failure CCD diagnosis

Source `3b703700` (capture implementation `fc1fc43d`). Fixed original learning
recipe and seeds reproduce the original failure, all 41 model/Adam arrays and
metadata, and all four canonical training files exactly. No held-out evaluation
or learner change. The protected snapshot is 14,736 bytes.

Exact production-kernel replay admits the first-failure cursor (generation 56),
not later reused scratch (generation 124). Failed head-box/sphere pair 100 in
lane 26 has constant projected centers and constant box orientation. The sphere
rotates; its geometry is invariant. All 257 signed distances are constant; the
independent FP64 separation is 10.2903015 µm versus a 10 µm tolerance, maximum
query error 1.050 nm. Normalized geometric sweep motion bound is exactly zero.
CA nevertheless exhausts 16 iterations because its motion denominator is
floored at tolerance. This identifies a stationary separated false unresolved
case; it does not yet repair CCD or establish learned manipulation.

Preserved failed measurements: isolated v1 probe timed out at 60 seconds during
Metal compilation with no output; separated v2 probe used different compiler
settings and changed interval.w by one ULP, failing the exact-byte gate. V3 loads
the actual production resolver library and a separately compiled query kernel
with production Metal4/no-fast-math settings. V3 and the CMake probe agree on
every result byte, with exact repeats. No trajectory rerun for those revisions.

Capture off/on native controls preserve results/status, protect first failure
through later candidate overwrite, leave disabled tail untouched, and reject
an older shader before rollout. The full Release build and generic contact
suite pass. These are scoped engineering checks, not biological qualification.

`manifest.json` references deduplicated actual Git blobs for all raw files.
`runtime-recovery.json` selects the minimal probe runtime and immutable inputs;
`verify_recovery.py` independently restores those actual Git bytes and executes
two exact native replays before validating and deleting its temporary files.
