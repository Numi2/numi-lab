# Recovered Human/Brain standing launcher

The recovered launcher checks the pinned Human and Brain revisions, detects
SwiftPM release library locations, and enables accepted sensor, motor, COM/support
and body-motion audits. A failed Git status check now rejects execution.

The legacy `--persistent-stand-trace` flag invokes a separate replay path and is
incompatible with the Brain-owned transaction. It is deliberately absent; the
accepted-state audits carry the trajectory. The tendon package requires the
matching original bone payload, not the later patella-rebased geometry.

On Apple M4 Pro, 32 one-millisecond steps were accepted with unassisted native
Brain/physics joint commits. [Receipt](receipt.json) and [run](standing-32steps.log)
bind this measured prefix. This is not ten-second standing, learned balance,
push recovery, physiological validation or a performance qualification.

Set `NUMI_HUMAN_BRAIN_HUMAN_REPOSITORY` and `NUMI_HUMAN_BRAIN_REPOSITORY` to
the clean pinned participants when they are outside the default home checkouts.
`NUMI_HUMAN_BRAIN_BUILD_DIR` defaults to this repository's `build` directory.

## Touch source stride repair and full-duration failure

The Human touch shader read contact identities with a 32-byte stride although
MRNumiHumanStandContactGPU has four uint4 words (64 bytes). Correcting that
stride restores every geometry receptor. The accepted sensor audit now rejects
incomplete coverage instead of accepting invalid channels.

The corrected native M4 Pro run completed 10,000 accepted 1 ms transactions.
All ten touch channels had validity 0x70 on every step, with maximum relative
difference 1.579e-7 between their summed normal forces and native total support.
The retained absolute error reaches 0.548 N during the divergent multi-meganewton
impact; it is not a physical force-accuracy qualification. No root assistance
was applied. The raw physical progress trace matches the pre-fix run exactly.

Quiet standing FAILED: the first sample with zero contacts is step 3081
(3.081 s), and maximum root speed reaches 224.335 m/s. The source-prepared
tonic motor program is unchanged. This establishes repaired sensory routing
and its coverage regression gate, not trained balance, sustained standing,
contact convergence or physiological qualification. Both failed runs remain
retained under /Users/n/numi-standing-reconcile-runs-20261013; full raw logs
are copied to the reconciliation archives on the Air.

[Assessment](standing-touch-stride-assessment.json) and
[first accepted sensor rows](standing-touch-coverage-excerpts.log) retain the
measurement details. The source/binary manifest is preserved with the run.
