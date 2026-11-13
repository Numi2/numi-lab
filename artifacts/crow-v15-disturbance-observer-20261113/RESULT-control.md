# Recovery observer parity control

Using the previously exposed seed 20261127, the new observer and unchanged straight-flight task binary reproduced q, v, policy actions and contacts exactly for every lane through its first episode, including the physically failed comparison lanes at intervals 760 and 762 and both 1,800-interval flight candidates. The same policy and task binaries were used, teacher mode was off, and the pre-aerodynamic body-wrench observer recorded all 2,688 expected lane/substep samples across intervals 895–915, all exactly zero. No observer errors were recorded.

This validates the observational host for a matched development disturbance experiment. It does not qualify disturbed-start recovery or flight. The complete control trace, replay, host, exact inputs and analysis are archived; the original reference state stream was independently reconstructed from Git and verified by part and full-stream hashes.
