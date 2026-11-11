# Corrected V1 teacher flight diagnostic

One privileged teacher trial reached all five waypoints in the first episode at step 1307 without a recorded physical failure and with a minimum accepted-step obstacle clearance of 0.1391 m. It did not land: foot-support step count was zero and its terminal root height was 1.137 m. The 3000-step episode timed out. These waypoint flags are not evidence of supported landing.

The first 100 q samples matched the teacher-off actor run, but the next sample diverged and 250 of the first 350 differed. Thus this run does not establish that the teacher used the previously tested autonomous gait and launch. It is privileged control on one exposed development course, not a reliable teacher, autonomous navigation, an imitation-training set, or an unused-layout evaluation. The complete run and its inputs are retained, including the earlier build error corrected before simulation.
