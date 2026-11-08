# Spatial and accuracy-adjusted midpoint study

Native runs are on Apple M6 with the binary and shader hashes in
[BINARY_INPUTS.sha256](BINARY_INPUTS.sha256). The native physics inputs were
built at `1dd807964`; source changes after that build add research harnesses,
not changes to the measured physics. The preregistration and qualification
limits are in [PLAN.md](PLAN.md) and [FOLLOWUP_PLAN.md](FOLLOWUP_PLAN.md).
Results are fixed-law synthetic contact evidence, not physical calibration.
The midpoint and reference-area contact options remain explicit research
options; the separate experimental CCD and augmented-Lagrangian options
remain disabled.

## The spatial gate failed

The independent FP64 RK4 references passed the prescribed 0.5 versus 0.25 us
temporal checks at 2, 4, 8, 16 and 32 divisions, with material coefficients,
activation width, physical area coefficient, geometry and duration unchanged.
Their total nodal reference area was 0.0004 m² at each resolution, within
floating-point summation error. This temporal check does not establish spatial
accuracy.

| Refinement | 1 ms-binned force | Peak force | Final impulse | Maximum cumulative impulse |
|---|---:|---:|---:|---:|
| 2 to 4 | 92.90% | 56.01% | 20.01% | 29.23% |
| 4 to 8 | 70.43% | 28.44% | 3.37% | 12.56% |
| 8 to 16 | 27.24% | 38.45% | 6.66% | 8.32% |
| 16 to 32 | 12.43% | 48.07% | 0.63% | 3.68% |

Force-bin errors are the largest absolute bin difference divided by the finer mesh peak binned force; peak and final-impulse errors use the corresponding finer values. The maximum cumulative-impulse error is normalized by the finer final impulse. Raw 2 us force/impulse traces and the
[analysis](spatial-analysis.json), [16-division follow-up](spatial-followup-analysis.json), and [32-division follow-up](spatial-n32-analysis.json)
are retained. In particular, the 27-node solution is not a spatially
converged result, and the finer follow-up did not establish convergence either.
The cause of the discrepancies remains unproven. The 32-division reference was a separately preregistered CPU-only follow-up; it does not establish native 32-division Matter behavior.

## Native temporal qualification and timing

At 2 divisions, midpoint at 15.625 us passed the mesh-specific gates: 0.0855%
binned force, 0.2584% sampled peak force, 0.0108% total impulse, and 0.0516%
maximum cumulative impulse. The native sampled peak was 0.722802 N versus
0.724674 N in the 0.25 us reference. The 0.0855% result is a 1 ms-binned
force error and is not a peak-force result. The native area sum was
0.0003999999607912 m² and the maximum independently checked nodal-weight
error was 2.46e-8 of total area. The sampled peak occurred at 4.8047 ms versus 4.798 ms on the reference 2 us trace. The largest cumulative-impulse deviation was -2.615e-7 N s at 4.882 ms; the final deviation was +5.476e-8 N s. The retained [curve inspection](curve-n2-midpoint.json) includes intermediate values and the reference sampling resolution.

Backward Euler passed none of the six preregistered steps down to 3.90625 us
at this mesh. At the smallest step its sampled peak error was 2.73%, final
impulse error 4.33%, and binned force error 18.98%. Therefore there is no
qualified backward Euler partner within this ladder and no matched-accuracy
speedup claim. This says nothing about smaller untested timesteps.

Three complete 10 ms midpoint runs at the qualifying 2-division timestep
used no per-step snapshots, while `captureDiagnostics` remained enabled.
All 640 steps were accepted and each final physical digest matched the
accuracy trace. The three wall times were 28.2878, 28.1441 and 28.2788 s;
the median accepted simulated seconds per wall second was approximately
0.0003536. Initialization and the final verification snapshot are outside
the timer; GPU diagnostic collection and normal per-step log output remain
inside it. These standalone results are not an integrator comparison.
At 4 divisions, midpoint also passed at 15.625 us: 0.1151% binned force, 0.2411% sampled peak, 0.0194% final impulse, and 0.0871% maximum cumulative impulse. The four-division [midpoint](curve-n4-midpoint.json) and [backward Euler](curve-n4-backward-euler.json) peak and cumulative-impulse curves are retained separately. Three standalone 4-division midpoint costs were 30.6918, 30.6668 and 30.4342 s per 10 ms rollout; each final digest matched its accuracy trace. 

At 4 divisions, backward Euler found no qualifying step in the prescribed ladder: 7.8125 us failed the accuracy gates, while the 3.90625 us attempt rejected with status 10 after 815 accepted steps, before contact, and restored the prior physical state. 

At 8 divisions no midpoint step qualified. The 62.5 us rollout completed but failed the accuracy gates; 125, 31.25, 15.625, 7.8125 and 3.90625 us rejected with nonlinear-solver status 10 and restored the previous physical state. The smallest step rejected after 203 accepted steps, before contact. Its status values have the final midpoint momentum-certificate layout: reported norm 1.39946e-13 N s, limit 1.37215e-13 N s, and maximum kinematic defect 1.37643e-20 m. The 7.8125 us failure likewise occurred before contact. These observations identify the failing certificate; they do not by themselves distinguish an inaccurate solve from finite-precision sensitivity. 

At 8 divisions backward Euler also found no qualifying step: the 31.25 us rollout completed but failed the accuracy gates, and the other five attempts rejected. Across all 11 rejected native trials in the ladder, the recorded accepted physical state was restored and the restore/continuation replay checks passed. The complete per-attempt statuses and diagnostics are in [native-ladder-analysis.json](native-ladder-analysis.json).

## Transactions and dynamic owner

The contact test suite passed 12/12 after the study, including the native midpoint, fixed-plane area, retry-order and dynamic-guard tests; see [ctest-all.log](ctest-all.log). These tests do not replace the failed spatial and timestep gates.

For the fixed-plane transaction, a native regression changed the previous
failure's diagnostic index from 0 to 32 and changed its diagnostic value.
Both subsequent contacting steps were accepted and produced byte-identical
authoritative physical state and success status. This isolates the effect of
diagnostic ordering on the next step; it is not a dynamic-owner test. Earlier
retained inverted-element tests established matching failure category and
physical restoration even when the first offending index differed.

A native guard regression confirms that the supported fixed-plane fixture initializes while an articulated-plane fixture is rejected at the fixed-plane boundary. The articulated case uses the legacy residual policy solely to reach that boundary; the initial attempt with the physically scaled policy was rejected earlier by world validation and is retained. These are guard tests, not evidence of two-way dynamics.

A two-way midpoint body has not been implemented or qualified. The
[owner audit](DYNAMIC_OWNER_AUDIT.md) records why the existing full-step rigid
candidate cannot safely be used as the half-step FEM candidate, and the
requirements for impulse, torque, work, swept clearance and whole-environment
rollback. The guard remains in force. Prescribed supports are not counted as
a two-way result.
EOF'