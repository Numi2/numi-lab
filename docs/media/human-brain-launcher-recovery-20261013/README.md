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
