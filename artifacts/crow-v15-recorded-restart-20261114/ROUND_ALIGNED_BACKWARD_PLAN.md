# Matched two-millimeter backward offset

The +2 mm offset from x=17.770 yawed branch made the first-episode grip substantially worse. Test the equal -2 mm offset x=17.768 using the same branch yaw, toe targets, state, policy, timestep, observer and all other settings. Compare to the retained zero and +2 mm trials; this paired geometric perturbation probes local robustness, not a solver defect or general landing ability.

Run 400 intervals, count only before first termination, and apply the unchanged all-substep bilateral support and opposition, tilt, speed, nonfoot/ground and no-reset gates. Passing requires all final 250 intervals qualifying and no termination. Archive failure.
