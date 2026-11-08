# Bound object tracking and per-environment observation stride

## Decisions and result

An existing direct-winner eligibility check used `config.bindings` after the configuration was moved into tracker state. On the tested Apple M6 implementation, this selected winner-only rendering even with a live object binding. The object reducer requires materialized image planes. Eligibility now checks the stored configuration.

The corrected 16x9 Franka fixture compiles a second camera with intrinsics scaled to that image. A separately compiled, independently rendered materialized image contains one selected valid-depth pixel in environment 0 and two in environment 1. Against that same fixture, the unchanged baseline reports a selected-pixel mismatch. The eligibility-only candidate matches those images and reports visible compact history in both environments, but the second environment's actor observation remains at its -5 sentinel: the reducer used the compact history stride for actor observations even when a depth suffix extended each observation.

The final candidate uses the full per-environment observation stride for actor output and retains the compact stride for the separate history buffer. Both environments now report matching selected image pixels, track confidence 1, compact-history confidence 1, and one/two selected normalized depth pixels. Both depth history planes match the materialized image to 2e-6, and a 13-float prefix canary remains unchanged. The graph-only no-binding direct-winner test and the existing larger tracked scene also pass. The source and executed binary identities are retained in `final-identities.sha256`.

The new semantic tag applies when object-track observations use an actor observation stride larger than their compact temporal history. A pre-fix synthetic zero policy with that extended layout is rejected under the new observation fingerprint; rebinding succeeds. A compact-layout policy stays compatible. These are contract tests, not trained policy performance. The earlier pre-depth-validity fixtures also pass their migration checks. All five listed native CTests passed.

## Evidence limits and failures

The first fixture incorrectly reused 160x120 camera intrinsics for a 16x9 renderer; an independent render also saw zero geometry. That failed setup and the one-line probe compilation mistake are preserved. The camera correction and the follow-on stride diagnosis were recorded before the corrected baseline and final measurements. The direct-winner positive control here does not assert a visible object. The observed one/two-pixel scene is a narrow geometry and device result, not sensor calibration, general occlusion qualification, real-world detection, or held-out task skill. Simulator instance IDs remain oracle selection. No performance measurement was attempted.

This change does not make visual state transactional across command failure or rejected physics. The full submitted-step and mixed-environment acceptance boundary remains open. In addition, the existing object reducer's depth-validity predicate deserves its own failure/retry and dropout qualification; this experiment does not prove its behavior when a frame is present but depth is invalid. The directional CCD and augmented-Lagrangian options remain disabled.
