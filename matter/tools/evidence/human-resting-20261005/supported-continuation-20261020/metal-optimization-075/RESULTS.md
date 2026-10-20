# Opt-in distributed solver and contact increment
Source base: ea2cbd8489e054a22be20811b8f5d4f6615fa502. Device: SSH numi, Apple M6, 12 GPU cores, 16 GiB, macOS 27. No visual verification.

Implemented GPU CGS2/reorthogonalization and guarded ICB columns; deterministic active-corner radix incidence and parallel surface sorting; explicit work domains; inactive transport exclusion from orthogonalization dispatches; within-Newton diagonal/cavity preconditioner reuse; tiled Armijo norms and object-range geometry validation; distributed begin/correction/true-residual measurements. Legacy paths remain default. Physical residuals, material laws and acceptance/rollback are retained. The partitioned basis helper is preconditioner-only; it does not implement reduced mechanical coordinates.

Enable: NM_FGMRES_DISTRIBUTED_CGS2=1 NM_FGMRES_CGS2_ICB=1 NM_CONTACT_NODE_INCIDENCE_RADIX=1 NM_FEM_DISTRIBUTED_GLOBAL_MEASUREMENTS=1 NUMI_MATTER_NEWTON_PRECONDITIONER_REUSE=1. ICB retains existing admission limits. Exact native commands, environment, hashes and inputs are in each invocation.json; runner is retained. Controller/presentation sustained throughput is not qualified.

Three matched frozen-control/candidate pairs (four roots, dt 0.125 ms, total 0.5 ms simulated) give median speedup 1.035592: 3.56% throughput gain, approximately 0.00001151x mechanics RTF. All terminal comparisons pass max(1e-7, 1% relative) and both tissue audits pass. The preregistered 2x prediction failed. These are release transients, not supported-rest checkpoints. See global-matched-results.json.

11 targeted tests pass; six physics/rollback/vascular tests also pass with optimized paths enabled; 14 contract tests pass. Build and logs retained. A stale statically linked vascular checker failed against the changed shader ABI during development; fresh relinking passes. Early controls that accidentally loaded the current compiled-in shader, and cold pipeline timeouts, are retained but excluded from gains (AMENDMENT-1.md).

Cross-encoder whole-stage spans replace truncated encoder fragments. Exact all-call tangent-cache profiling on root 3: 96 calls, 8.061476998 GPU seconds, inside 11.444719589 seconds of linear stages. These intervals overlap and must not be added. Profile process wall 55.4085 s versus 54.7–55.3 s uninstrumented; no sampled-kernel extrapolation. No dropped/unavailable counters. See exact-cache-stage-profile.json. Exceptional exits can omit an unclosed span; they do not report it as zero.

Remaining: actual packed Krylov layout/full conditional Newton graph, exact anatomical Schur condensation, accepted-step recycling, real hybrid reduced coordinates, supported resting initialization, timestep qualification, and sustained coupled 10/60/330-second runs. No 0.1x claim. Follow-up experiments target the measured tangent-cache evaluator and residual-adaptive Newton forcing.

Large generated packs/binaries remain local and are not added to Git; compact logs/manifests preserve failed and successful trials.
