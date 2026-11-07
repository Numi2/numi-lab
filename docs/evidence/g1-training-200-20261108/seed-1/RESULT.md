# Trial 1

The first of three registered trials completed in 71.45 s wall time with no
reported failed physics steps. The 200-update training loop reported 48.19 s.
On the selector seed, incumbent and candidate had respectively 93 and 85
height-or-tilt termination events across eight environments; neither had a
clean horizon. Candidate final forward progress ranged from 0.014 to 0.737 m,
so it failed the preregistered initial skill screen.

The continuous selector advanced the candidate on its positive 1.255 score.
This is incumbent-relative advancement, not evidence that the policy survived
its horizon or qualified for a physical task. The selector calls the normalized
task category "velocity"; both native evidence records retain the exact
unitree_g1_legs_locomotion task. Raw policies, optimizer state and traces remain
in the measured-host run directory. This single trial does not determine the
registered three-seed prediction, learning quality or time to skill.
