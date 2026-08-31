# Numi Human passive-aware architecture fit v4

The 2026-08-31 passive-aware source checkpoint fixes a systemic fitting gap in
all 416 compliant muscle-tendon architectures. The previous objective sampled
only nonzero activations. Large active-force targets could therefore hide a
large fitted force at activation zero, even when aggregate NRMSE looked good.

The source fitter now includes activation zero and weights that passive channel
by `1024`. Each source manifest record exposes source-pose passive force, fitted
passive force, and their absolute error. This is still an inference of `L0` and
`LT` from the retained MyoSim force surface; it is not a claim of measured
subject-specific fiber length, tendon slack length, or pennation.

## Source-fit result

The directly comparable passive-oracle error across 416 muscles changes as
follows:

| Metric | Adaptive v3 | Passive-aware v4 |
| --- | ---: | ---: |
| Mean absolute error (N) | 3.5242 | 0.81787 |
| Median absolute error (N) | approximately 0 | approximately 0 |
| P95 absolute error (N) | 15.4167 | 4.44320 |
| Maximum absolute error (N) | 137.6033 | 29.38435 |
| Muscles above 0.1 N | 142 | 100 |
| Muscles above 1 N | 82 | 54 |
| Muscles above 10 N | 34 | 8 |

`UI_UB5` changes from a fitted `2.2306 N` passive force to `0.21346 N`, against
the source oracle's `0.07459 N`. Fifth-lumbrical, radial-interosseous, and
fifth-superficial-flexor fits reproduce their zero source-pose passive force to
numerical precision. Soleus passive error falls from `137.60 N` to `25.43 N`.

The v4 weighted fit NRMSE is mean `0.09582`, median `0.04276`, and maximum
`0.58500`. It is not directly comparable with v3 NRMSE because activation zero
and the passive weight change the objective. The complete source build took
`16:28` on the Mac mini M4 Pro and produced payload SHA-256
`a4381465ca97a7af20108180e2fc0d0f0d9c11778d913de4c0728bd20ed459a7`.

## Nonvisual whole-body result

The unchanged Apple-native runtime used the same NHCNT1 unilateral support,
NHEQ1 joint equalities, source passive coordinate tissue, `0.0001 s` response
step, 240 activation sweeps, twelve accepted pose steps, and all 128 residuals.

| Metric | Adaptive v3 | Passive-aware v4 |
| --- | ---: | ---: |
| Internal normalized residual RMS | 2.93227 | 2.58813 |
| Maximum acceleration (rad/s2) | 62.30 | 30.59 |
| Coordinates above 100 rad/s2 | 0 | 0 |
| Coordinates above 10 rad/s2 | 7 | 5 |
| Coordinates above 1 rad/s2 | 50 | 52 |
| Fifth-MCP abduction, right (rad/s2) | -62.30 | 1.76 |
| Fifth-MCP abduction, left (rad/s2) | -60.34 | 1.25 |

Body weight closes to `2.26006e-9` relative error, the maximum root-force
residual is `2.15353e-6 N`, and replay is bitwise. The result remains
`internal_balanced=false`. Bilateral wrist flexion is now dominant at
`30.59/26.72 rad/s2`; bilateral third-MCP flexion follows at
`-15.87/-13.63 rad/s2`. Left-knee acceleration is `11.40 rad/s2`, while the
unilateral limit/support solve makes its scalar net force zero. Those residual
families, not the now-reduced fifth-ray preload, own the next mechanics work.

The raw transcript, source manifest, and source-name-enriched 128-DoF report
are in `docs/media/numi-human-passive-fit-v4/`.

## Boundary

This qualifies a lower-rest-force architecture inference and a deterministic
static unilateral-support improvement. It does not qualify internal
equilibrium, dynamic contact, sustained standing, gait, subject-specific
tendon material, deformable fascia, or organ mechanics. No force clamp,
invented rest torque, or render-only correction is included.
