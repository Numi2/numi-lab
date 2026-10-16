# Coupon diagnostic replay overlay 182-001

This isolated overlay adds an explicit `--coupon-resume-diagnostic-replay` mode
to accepted coupon checkpoint continuation. Ordinary resume retains the strict
runtime-settings digest. Diagnostic replay still requires the original and
current source-bound runner records, the unchanged checkpoint and protocol
manifest, exact native source/device fingerprints, accepted-prefix proof, and
the full effective-environment contract.

The explicit mode permits differences only in the five registered
read-only selectors and seven registered output paths. It also permits a
registered diagnostic path to transition between absent and set only after
both source and current runner proofs validate. All other `NM_*` and
`NUMI_MATTER_*` values remain exact, including timing settings. Selector
targets are checked against the accepted control; regional capture must use
the exact six-field form and a valid environment index. The source-runner
registry is checked against the exact 49-, 50-, or 51-key source variants.

The patch changes only `matter/tools/cardboard_probe.mm` and
`matter/tools/cardboard_coupon_resume.hpp`. The C++ contract test exercises the
actual header implementation. ObjC++ validation compiles the actual candidate
probe in both CMake target configurations against immutable source 181 headers
with `-fsyntax-only`; the one pre-existing unused-function warning is retained
in the log.

No native checkpoint-resume replay was run here. The overlay therefore makes
no claim that restored trajectories match an uninterrupted native run, and it
does not claim physical qualification. Source 181 remains immutable.

The manifest binds candidate/base hashes, the patch, test, validation script,
and logs. The manifest does not self-hash; its SHA-256 is recorded in
`validation/seal.txt` after all other files are frozen.
