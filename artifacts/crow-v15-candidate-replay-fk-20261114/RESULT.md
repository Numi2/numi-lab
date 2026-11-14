# Candidate replay pose consistency

For the actual lane-7, seed-20261127 turn replay, a separately constructed experimental hip-roll v15 model was evaluated using native CPU forward kinematics at frames 1, 170, 376, 500, 898, 918, 1200, 1550 and 1800. Across 35 bird links and 315 comparisons, maximum COM-position error against the replay body states was 4.64e-7 m and maximum sign-invariant quaternion component norm error was 3.19e-8. Source, executable, native library hashes, input, full output and comparison are retained.

This establishes close pose consistency at those sampled frames for this reconstructed model. It does not independently validate every frame, contact timing, soft feathers, renderer camera, physical behavior or biological accuracy.
