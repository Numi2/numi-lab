# Retained endpoint and work diagnostics

## Canonical-tangent first pair

`canonical-first-pair-comparison.json` compares `global-icb-baseline-1` with `global-icb-canonical-1`. The comparison fails only `max_J`: 1.11645574729 to 1.12882705044 (+1.1081%), beyond the predeclared 1% endpoint gate. `max_anchor_reaction_N` changes 2.42182409184 to 2.44481835624 N (+0.9495%), within that gate. Both runs report four accepted steps and passing FEM and fascia terminal audits. Since `max_J` is emitted by the FEM terminal audit and fascia has a separate audit, the failed comparison is the global FEM terminal `max_J` endpoint, not a fascia-audit failure. The retained audit is terminal-only; it does not identify which FEM object or which accepted root first diverged. The available FGMRES iteration total is not a per-root physical-state or line-search trace, so it cannot locate that first divergence. The Armijo trace-root environment flag alone did not enable the Armijo trace path.

## Matched adaptive-forcing work pair

`ew-matched-first-pair-comparison.json` compares `global-icb-work-baseline-1` and `global-icb-ew-work-1`. The runs use the same executable, binary SHA-256 (`c9dab7f2...`), input-invocation SHA-256 (`5d8f176d...`), source HEAD (`aadcd68e...`), and linked library and metallib hashes. Their per-run output and failure-receipt paths differ as expected. The sole simulation-environment difference is `NM_FGMRES_INEXACT_NEWTON_FORCING=1`; the candidate working-source hash differs in `runtime.mm` as expected for this feature. Each run advances four accepted 0.125 ms steps (0.5 ms total).

The retained `matter_fgmres_active_work` records give used Arnoldi columns by logged root index:

| Root index | Baseline | Adaptive forcing | Change |
| --- | ---: | ---: | ---: |
| 0 | 1,693 | 1,779 | +5.08% |
| 1 | 1,874 | 739 | -60.57% |
| 2 | 2,613 | 564 | -78.42% |
| 3 | 2,117 | 841 | -60.27% |
| **Total** | **8,297** | **3,923** | **-52.72%** |

These are measured used columns from the root counters, not a maximum-per-solve estimate or inferred dispatch savings. Both configurations encode 384 restart cycles per root; work-bearing cycles are 108/140/201/165 for baseline and 113/93/58/103 for the candidate. The first logged root uses 86 more columns under adaptive forcing; the counters are aggregated by root and do not provide a per-Newton residual, line-search, or first differing physical state.

The matched comparison fails the predeclared endpoint gate on `max_anchor_reaction_N`: 2.42182409184 to 2.51183820907 N (+3.7168%). All other captured endpoints and both tissue terminal audits pass. The individual terminal runs pass their FEM/fascia audits; this is a pairwise endpoint-equivalence failure, not a failed run. Body wall time is 43.496946208 s versus 36.183565542 s (1.2021x speedup, 16.81% lower candidate wall time), below the experiment's 1.5x performance threshold and accompanied by the endpoint-gate failure. Keep the candidate unqualified and off by default; the measured column reduction does not establish an accepted performance gain.

## Evidence limits

The retained Human artifacts capture terminal endpoint metrics and per-root FGMRES active-work totals, not per-root FEM deformation, Newton residual history, line-search acceptance history, or object-level `J`. They therefore establish which comparison metric fails and how much measured linear work changed, but they do not establish the causal first divergence or identify an offending FEM object. No acceptance gate or solver algorithm was changed during this review.
