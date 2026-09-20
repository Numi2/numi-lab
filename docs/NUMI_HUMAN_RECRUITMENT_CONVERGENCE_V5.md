# Numi Human recruitment convergence v5

This solver frontier is superseded by `NUMI_HUMAN_GLOBAL_RECRUITMENT_V6.md`.
The v5 artifact remains the coordinate-sweep baseline.

The 2026-09-01 whole-body checkpoint separates an under-converged static
recruitment solve from missing anatomy. No geometry, muscle route, force law,
passive tissue parameter, contact, joint equality, or tendon attachment changed.
Only the deterministic activation-coordinate budget increased.

The visual probe now accepts `--whole-body-activation-sweeps <1..8192>` for
the nonvisual whole-body support certificate. The interactive/default compile
remains `240` sweeps; high-quality offline qualification can request more work
explicitly without making normal startup six minutes slower.

## Apple M4 Pro result

Both rows use the passive-aware 416-muscle NHMYO2 artifact, NHCNT1 unilateral
support, NHEQ1 joint equalities, 40 source passive-coordinate rows, a
`0.0001 s` response step, and all 128 residuals.

| Metric | 240 sweeps | 960 sweeps |
| --- | ---: | ---: |
| Internal normalized residual RMS | 2.58813 | 0.980016 |
| Maximum acceleration (rad/s2) | 30.5909 | 20.0873 |
| Coordinates above 100 rad/s2 | 0 | 0 |
| Coordinates above 10 rad/s2 | 5 | 2 |
| Coordinates above 1 rad/s2 | 52 | 34 |
| Maximum root acceleration residual | 2.97223 | 0.607936 |
| Relative body-weight error | 2.26006e-9 | 2.32887e-9 |
| Replay | bitwise | bitwise |

The `960`-sweep result reduces normalized RMS by `62.14%` relative to the same
v4 artifact at 240 sweeps. Bilateral wrist flexion remains dominant at
`20.09/16.53 rad/s2`. The next largest residuals are left third-MCP flexion,
thumb CMC flexion, right third-MCP flexion, left CMC abduction, right
fourth-MCP flexion, and right third-MCP abduction. Bilateral fifth-MCP
abduction remains below `2 rad/s2` and no longer dominates.

## Interpretation and boundary

The previous wrist and third-MCP magnitudes were partly solver convergence,
so adding new tendon anatomy to cancel them would have been a false fix. The
remaining result is still `internal_balanced=false` and the solver consumes
its complete 960-sweep budget. It is therefore a better diagnostic state,
not proof of static equilibrium, dynamic stability, sustained standing, gait,
or deformable tissue qualification.

Further production work should replace brute-force offline coordinate sweeps
with a deterministic bounded global recruitment method and cache accepted
activation/fiber state. Only residuals that persist after that solver gate
should motivate new sourced wrist, thumb, hood, fascia, cartilage, or ligament
mechanics.

Machine-readable evidence, the pinned source-name map, and the raw bitwise
replay trace are in `docs/media/numi-human-recruitment-convergence-v5/`.
