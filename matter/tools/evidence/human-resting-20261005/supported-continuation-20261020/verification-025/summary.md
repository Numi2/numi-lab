# Unified verifier 025 — setup failure

The rollback verifier completed its CPU fixtures and geometry/interface checks, but stopped before native app execution because the contact-geometry checker path was wrong. The exact error is retained in `logs/contact-geometry.run.log`: `tools/verify_human_supported_tissue.sh: line 48: /Users/numi/numi-human-aortic-contact-opt-20261019/build-integrated-validation-20261019/profile-build-039/bin/numi-matter-cardboard-contact-geometry-check: No such file or directory`. No unified rollback invocation, execution receipt, or native mechanics result was produced for 025.

Recorded passing checks: compact-boundary audit, float32 geometry audit, FEM/body interface fixture, mixed-donor accepted skin transfer, precomputed skin/fat layers, whole-body skin layers, and the registered-ligament CPU fixture. Their original compile/run logs are preserved. Large intermediate geometry CSVs and test binaries are omitted; binary hashes are in `source-identity.json`.

The fresh 026 attempt uses the corrected verifier path. Per the task handoff, the change between attempts is limited to that script path; this 025 package remains bound only to its captured input identity, test-binary hashes, and repository HEAD noted in `source-identity.json`.
