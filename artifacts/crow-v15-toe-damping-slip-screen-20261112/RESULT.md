# Increased toe-joint damping: rejected

All 16 modeled toe joints reported the intended .030 damping versus .003 in the reference. The changed physical-model fingerprints were explicitly rebound for both actors with identical weights and transforms; the rebound two-step preflight succeeded. Eight runs on the previously used seed lasted six seconds without reported physical failures or failed environment steps. The geometry and other planned settings remained fixed. Because the model differs from reset, trajectories are matched by seed and configuration, not exact state parity.

In the 4–6 s walking window, forward progress was .109–.113 m versus .131–.139 m in the references. Relative to each matched baseline, the left contact-speed integral was .74–.79 and the right was .86–.98. Every candidate start briefly unloaded both feet for 2–20 substeps, where the baseline had none. No candidate toe-speed cap occurred during the walking window and no nonfoot impulse was reported there; early toe caps occurred in both model variants and are separately logged.

The candidate fails the predeclared support and per-foot reduction gates and is not adopted. The substep speed integral is not tracked material-point slip; the damping value is uncalibrated and the test does not qualify walking, takeoff, flight or landing.
