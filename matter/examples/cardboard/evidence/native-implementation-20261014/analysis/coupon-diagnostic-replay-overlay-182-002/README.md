# Coupon diagnostic replay overlay 182-002

This fresh overlay is based on immutable source 181 and supersedes 182-001 only for the proof-report correction described here. It preserves the explicit `--coupon-resume-diagnostic-replay` gate and the strict ordinary-resume behavior.

Diagnostic replay still requires the original and current source-bound runner records, unchanged checkpoint and protocol manifest, exact native source/device fingerprints, accepted-prefix proof, and the full effective-environment contract. It permits differences only in the five registered read-only selectors and seven registered output paths. A registered diagnostic path may transition between absent and set only after both source and current runner proofs validate. All other `NM_*` and `NUMI_MATTER_*` values remain exact, including timing settings.

The report now publishes current runner proof digests whenever the current proof was validated, even when the validated runner has no output paths. It also records explicit source and current proof modes, distinguishing checkpoint-metadata-bound source proofs, source-run reconstruction for diagnostic replay, current diagnostic replay contracts, and unvalidated proof state. Tests cover validated no-path proofs as well as missing and malformed digests.

The patch changes only `matter/tools/cardboard_probe.mm` and `matter/tools/cardboard_coupon_resume.hpp`. The C++ contract test exercises the actual header implementation. ObjC++ validation compiles the actual candidate probe in both CMake target configurations against immutable source 181 headers with `-fsyntax-only`; the pre-existing unused-function warning is retained.

No native checkpoint-resume replay was run. This overlay makes no claim that restored trajectories match an uninterrupted native run, and it does not claim physical qualification. Source 181 and sealed overlay 182-001 remain unchanged.

The manifest binds candidate/base hashes, the patch, test, validation script, and logs. The manifest does not self-hash; its SHA-256 is recorded in `validation/seal.txt` after all other files are frozen.
