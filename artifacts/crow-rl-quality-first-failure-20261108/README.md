# First crow quality rejection captured and replayed exactly

The opt-in native inspection now preserves the first completed quality-solver nonconvergence problem and failed iterate before later encoded passes reuse transient arrays. The 98,864-byte first-lane record reproduces the public control 0 / substep 2 rejection: quality status 7, 20 Newton iterations and 77 PCG iterations. An isolated native dispatch reproduces every record byte. Encoding eight controls instead of one also preserves the same first record.

The capture contains the exact dispatch, block topology, physical dynamics matrix, Jacobian, bias, free and warm states, failed velocity, impulses, row velocities and gradient. The packed record is bounded by a 16 MiB batch budget. It never supplies state to the physical solver, task, actor or learner. No capture is written for accepted component cases or the default solver. Older shaders without the inspection entry point reject before encoding.

## What the retained problem shows

There are 34 generalized velocities, 480 capacity rows and 160 blocks. The scaled native optimality residual is 2.217938888e-10, above the unchanged 1e-10 threshold. Feasibility and equality residuals are zero; complementarity is 5.914007033e-6. The failed Newton decrement is 1.824539936e-19.

An independent FP64 weighted-cone projection and Newton analysis passes five closed-form projection controls and a weighted boundary derivative control (maximum difference 7.483e-11). Reprojecting the failed state gives scaled optimality 2.202227937e-10. One FP64 Newton iteration reaches 7.687e-19. Rounding that reference velocity to FP32 gives 2.184668321e-10; the maximum failed-to-reference velocity difference is 2.289e-8.

This points to representability and residual arithmetic as the next discriminating investigation, rather than immediately extending the Newton budget. It does not establish a global FP32 accuracy floor or justify changing the physical acceptance gates. The physical mass matrix is symmetric positive definite; its unscaled condition estimate is 6.25e8 and includes different coordinate units and inactive scene coordinates.

Source review found that scalar motor blocks leave two padded row slots outside their block. All 64 such rows have zero Jacobian, bias and warm impulse here; their captured impulses are zero. This observation does not make unassigned scratch safe under other conditions and does not establish padding as the cause of this rejection.

## Noninterference and retained instrument failures

The first capture attempt was rejected before physics: the new inspection bit was missing from the world dispatch validator's known-flags list. That original shader/runtime, the failed attempt and the explicit repair plan remain retained.

The repaired capture-off path matches all eleven previous crow physical/failure files exactly. Capture-on matches all ten physical and contact files exactly; four public world-status records differ only by the explicitly enabled inspection bit in flags. The original whole-file byte prediction therefore failed on expected metadata. Its failed comparison is preserved; the amended comparator checks precisely that flag and requires all other status bytes exact. It masks no physical field or acceptance tolerance.

All 576 motor and 48 zero-gain regression conditions pass and preserve their physical records. Only nondeterministic GPU timing is excluded from equality for the 576-row TSV; both original and new timings are retained. The default crow accepts 640 transitions and preserves all ten physical files exactly. Full Release, Constraint IR and native contact-owner checks pass.

The original admission attempt executed three commands. The repaired study executed six, stopped at the metadata mismatch, then completed its remaining four after an explicit comparison amendment. No scientific failure or native attempt was discarded. One failed local inspection used the nonexistent .venv/bin/python path; the discovered python/.venv/bin/python was used for the sealed FP64 analysis.

## Remaining work

The crow still rejects its first control. No numerical tolerance, solver iteration budget, physical parameter, policy or production profile was changed or promoted. Next: identify the controlling coordinates and certify quantization/roundoff effects, then test an owning native numerical correction against independent physical residuals. Deterministic initialization of uncovered scratch also remains a correctness task. Multi-joint crow behavior, hybrid contact timing, reliable manipulation and the full lifelong environment remain unqualified.

Raw inputs, exact runtimes, failures, analysis and local drafts are retained through content-deduplicated Git recovery before temporary cleanup.

## Git recovery and storage closure

The [recovery record](../crow-rl-git-recovery-quality-first-failure-20261108/README.md) restores all 157 raw files (34,741,071 bytes) from actual committed Git objects. Nine native commands plus the FP64 analysis pass 41 byte-exact comparisons, including all captured failure files, protected records, reference arrays and analysis report. All 576 physical TSV records match excluding only GPU timing, whose original and replay bytes remain retained.

Pushed HEAD parity, actual Git readback, exact hash/mode/mtime and zero open handles precede deletion. The temporary restored workspace, original raw scope and local drafts were removed. New compressed original payload data totals 7,753,809 bytes; identical existing Git content is reused. The current build and protected deployment policy are retained. The crow remains rejected and the full goal remains incomplete.
