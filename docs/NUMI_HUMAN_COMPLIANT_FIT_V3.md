# Numi Human compliant architecture fit v3

This historical checkpoint is superseded for current whole-body mechanics by
`NUMI_HUMAN_PASSIVE_FIT_V4.md`, which adds the missing activation-zero fit gate.

The 2026-08-31 source compiler checkpoint corrects a whole-body mechanics
error before adding more tissue constraints. The old bounded coarse search
forced some inferred optimal-fiber and tendon-slack lengths onto its limits.
That produced source-inconsistent passive preload, most visibly `17.4896 N`
in each fifth lumbrical at zero activation.

The source compiler now uses a wider positive, deterministic multiresolution
fit. A 21-by-21 pass classifies every one of the 416 source force surfaces.
Only fits above five percent NRMSE receive the expensive 41-by-41 global and
three-stage local refinement; already-good muscles use a smaller local path.
On the Mac mini M4 Pro this reduced the final source build from `13:46` for an
all-exhaustive search to `8:01` while preserving the difficult-muscle results.

## Source-fit result

The old and final adaptive artifacts compare as follows:

| Metric | Old NHMYO2 | Adaptive NHMYO2 |
| --- | ---: | ---: |
| Mean fit NRMSE | 0.12540 | 0.06569 |
| Median fit NRMSE | 0.03616 | 0.02247 |
| Maximum fit NRMSE | 2.97973 | 0.56737 |
| Fits above 10% | 130 | 88 |
| Fits above 20% | 64 | 42 |
| Fits above 50% | 20 | 2 |
| Fifth lumbrical fit NRMSE | 0.65814 | 0.10156 |
| Medial gastrocnemius fit NRMSE | 0.48960 | 0.04772 |
| Lateral gastrocnemius fit NRMSE | 0.47833 | 0.04958 |

These fitted lengths are explicit inferences from the retained MyoSim static
force surfaces. They are not claimed as measured anatomical fiber lengths,
tendon slack lengths, or pennation.

## Nonvisual whole-body result

The unchanged Apple-native runtime was run with identical NHCNT1 support,
NHEQ1 joint equalities, passive coordinate tissue, `0.0001 s` response step,
240 activation sweeps, twelve accepted pose steps, and all 128 residuals.
Against the prior artifact, the final adaptive artifact changes:

| Metric | Prior | Adaptive |
| --- | ---: | ---: |
| Normalized residual RMS | 3.79966 | 2.93227 |
| Maximum acceleration (rad/s2) | 147.46 | 62.30 |
| Coordinates above 100 rad/s2 | 2 | 0 |
| Coordinates above 10 rad/s2 | 13 | 7 |
| Coordinates above 1 rad/s2 | 56 | 50 |
| Fifth lumbrical passive force (N) | 17.4896 | 0.01567 |

Body weight still closes to `2.17e-9` relative error and replay remains
bitwise. The equilibrium gate remains failed: `internal_balanced=false`.
Largest unresolved coordinates are bilateral fifth-MCP abduction, bilateral
wrist flexion, bilateral third-MCP flexion, and left shoulder rotation. This
makes source-resolved fifth-ray and wrist force sharing the next mechanics
scope; it does not justify a stop torque, sign flip, or visual correction.

The raw trace, final source manifest, and source-name-enriched 128-DoF report
are in `docs/media/numi-human-compliant-fit-v3/`.
