# Differential wing-sweep turn: rejected

All 48 sampled pre-intervention q/v/actions matched the corresponding retained bank-and-pitch controls. Contact-substep coverage was complete; no solver errors or overflow were logged. The zero-residual middle pair reproduced its corresponding sampled control metrics exactly.

No positive/negative pair passed the predeclared sampled flight checks. With -0.30 mirrored residual, both lanes reached interval 1600 but heading errors remained 1.204 and 1.297 rad, sideslip was -1.714 and +1.249 rad, and forward body-relative velocities were -0.098 and +0.237 m/s. The matched control heading errors were 1.218 and 1.308 rad. With +0.30 residual, positive-direction lane 4 first registered a nonfoot ground contact at interval 1320 substep 9 and later terminated for physical failure at interval 1401; its corresponding control ended at interval 1466. These observations do not prove that the contact caused termination. The opposite lane completed 1600 but retained a 1.430-rad heading error and strongly sideways motion.

The tested residuals are model-specific engineering inputs, not crow measurements. This is one used seed, sparse samples and an obstacle-free intervention; direct body moments remained active before 9 seconds. It establishes neither a physically qualified controller nor navigation, landing or biological realism. Do not promote.
