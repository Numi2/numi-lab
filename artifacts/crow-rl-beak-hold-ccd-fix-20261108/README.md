# Four-pose beak clamp/lift/release trial

The sealed six-arm study is inconclusive: clamp and partitioned clamp reject
at tick32; timestep refinement rejects at tick96. Open-jaw, no-contact and
zero-friction controls each complete 640 accepted transitions without native
rejections. Every arm, typed failure, accepted prefix and exact runtime is kept.
No successful full gesture or learned manipulation is claimed.

The later CCD initializer was erasing event counters in failed lanes. The
separate event-status repair reproduces the exact clamp trace/rollback while
preserving the first zero-time and advance-budget failures. Therefore neither
the old zero counters nor absent pair-distance captures identify the original
failure mechanism. See ../crow-rl-ccd-event-status-20261108/results.json.

Object poses vary deterministically, but are not biological replicates. The
no-contact arm disables ground contact too. This diagnostic begins with a rigid
surrogate between the beaks; it does not test retinal control, ground pickup,
food transfer, swallowing or biological calibration.
