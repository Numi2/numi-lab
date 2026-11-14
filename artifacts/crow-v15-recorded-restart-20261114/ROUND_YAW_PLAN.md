# Round perch alignment test

Hypothesis: the right front toe misses useful opposing contact because the bird and branch are misaligned. Earlier rectangular-fixture controls showed a +0.2 rad branch yaw improved bilateral overlap relative to -0.2 rad. On the physically round capsule, test +0.2 rad at x=17.755 against the recorded zero-yaw x=17.755 control; all other checkpoint state, forced landing phase, policy, scene geometry and observer stay the same. This is a relocated development fixture, not an autonomous original-course landing.

Run 400 intervals without suppressing termination and analyze only before first termination/reset. Report opposed contacts for each foot, consecutive complete grip intervals, root tilt and speed, all 16 substeps, nonfoot and ground contacts. Passing requires no reset or termination and at least the final 250 consecutive intervals meeting all existing force and posture gates. Preserve and report failure, not only the best interval.
