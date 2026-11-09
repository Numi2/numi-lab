# Midpoint research checkpoint: 9 November 2026

This is a checkpoint, not a qualification or a performance result. The CCD
directional and interval-local selectors and augmented-Lagrangian contact
remain disabled by default.

* The hash-bound 64-division repeat confirms the retained reference; its
  temporal test and the 128-division temporal test pass. Their spatial
  comparison fails. `SPATIAL_N64_REBOUND_006_RESULT.md` and
  `SPATIAL_BOUND_FORCE_LANDMARKS.md` give the limits and separate force-peak,
  force-bin and impulse evidence. This is CPU-reference evidence and does not
  qualify native Matter at these sizes.
* The native 27- and 729-node midpoint rollouts are compared only to their
  respective fixed-mesh time-qualified references. Peak force, binned force
  and cumulative impulse are separately documented in
  `MIDPOINT_FORCE_LANDMARKS.md`; these do not establish spatial convergence.
* The predeclared compressed backward-Euler control is running in
  `recovery-sequence-005`. At this checkpoint its first 320,000-step rollout
  has not completed. Its streamed gzip is incomplete and is not an accuracy
  or replay result. The sequence can run the matched no-per-step-snapshot cost
  comparison only if both tested arms pass the accuracy and provenance gates;
  each arm still has diagnostic capture enabled in its runtime configuration.
  The later independent larger-mesh reference is conditional and has not run.
* All four genuine reacting-body native attempts stopped before the requested
  full trajectory. The existing rollback and wrench evidence does not qualify
  the missing high-and-low-word owner transaction. The restricted draft is on
  `codex/reacting-slider-compensated-20261109` with admission closed. Its CPU
  decimal arithmetic check and Metal compilation are not a native coupled
  transaction, do not prove body dynamics, and do not certify restoration.
* Storage recovery and its independent restore checks are documented in
  `storage-recovery-20261109/README.md`. The preserved Git fragments were
  treated as opaque bytes; a valid-pack or complete-fsck claim was not made.

Do not infer a speedup, spatial convergence, force-qualified backward-Euler
control, or successful two-way native rollout from this checkpoint.
