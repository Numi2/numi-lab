# Native motor rows pass component gates; crow solver still rejects

The experimental native actuator prepares motor constraints from current source pose, velocity, targets and limits, then solves them with contact in the quality solver. Positive and negative scalar intervals represent the existing motor dead zone; native impulse divided by substep duration supplies the actual effort. Zero stiffness/damping retains source-velocity dry friction as a fixed impulse.

The sealed 576-condition study passed its absolute physical gates: maximum force-law error 2.0342e-9 N, momentum residual 6.704e-11 Ns and position/velocity consistency error 9.043e-13 m. These are finite one-DoF component results. They do not qualify the crow.

The first actual crow candidate was rejected before physics because generated motor rows exhausted the declared constraint capacity. The compiler now adds checked motor-row headroom while preserving existing contact capacity. Ten host checks pass. The 576 physical records remain exact after this correction; GPU timing is excluded from equality and retained separately.

A second review found that the initial shader dropped source dry friction when both stiffness and damping were zero. The retained old shader fails the frozen negative control after nine published conditions. The repaired shader accepts all 48 edge conditions: maximum force-law error 1.011e-11 N, momentum residual 2.281e-11 Ns and position/velocity consistency error 2.910e-13 m. All 576 earlier physical records remain exact.

## Actual crow result

The original bounded/temporal/hybrid arm accepts 368 recorded transitions through tick 91, reproducing the prior 364-record prefix exactly, then terminates for tilt at 1.20000863075 radians in lane 3. Its maximum recorded jaw penetration is 0.00539132553934 m, above the 0.0008 m gate.

The bounded/quality/non-hybrid control rejects its first control at substep 7. After the capacity repair, coupled/quality/non-hybrid reaches physics but rejects its first control at substep 2: status 11, 20 Newton iterations, 77 PCG iterations, zero backtracks, and zero accepted control transitions across four rolled-back lanes. It requires 42 constraints within capacity 160. Initial pose, velocity, scene and geometry bytes match the baseline. The zero-gain correction reproduces this candidate failure exactly.

The candidate prediction is contradicted. The temporal-to-quality comparison also changes CCD and cannot isolate either factor. The same-solver comparison isolates motor formulation more closely, but numerical rejection prevents a behavioral improvement conclusion. Aggregate diagnostics do not identify conditioning, an accuracy floor, or insufficient Newton iterations as the cause.

## Validation and retained failures

Full Release builds pass. Constraint IR and native contact owner checks pass. Three default-crow replays each preserve all ten physical files for 640 transitions. The protected deployment policy is unchanged. No policy or production profile is promoted.

The original crow runner completed all three native commands, then failed while reading candidate initial files that did not exist after host admission rejection. Its original source, outputs and separate corrected interpretation are retained. Failed local edit-helper attempts are also retained; their exact-match assertions stopped before remote source mutation.

Plans, source, exact runtime identities and failures are retained in this repository. Raw payload recovery and scoped native replay are governed by the adjacent Git recovery study. Equality for the 576-condition TSV excludes only the nondeterministic gpu_ms column; both original and replay timing bytes are retained.

## Next owning investigation

Capture the exact first failed quality problem and failed iterate inside the native solver, then independently analyze its residuals in FP64 before selecting a numerical correction. Preserve the first failure before later substeps overwrite transient arrays. Do not infer a cause from aggregate status or relax physical gates to obtain acceptance.

Multi-joint accuracy, hybrid event timing, reset/checkpoint/partition behavior, energy integral, biological calibration and end-to-end learning throughput remain unqualified. This slice does not complete the lifelong crow environment.
