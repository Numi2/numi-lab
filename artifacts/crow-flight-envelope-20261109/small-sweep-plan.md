# Exploratory bounded small sweep probe

After the +0.30 equal-sign sweep intervention produced yaw authority but severe forward overspeed, compare smaller +0.05/+0.05 and +0.10/+0.10 offsets after 6 s against the matched roll-1.5 no-offset arm. Preserve the same filtered actuator limits, roll feedback, +0.55/-0.55 pronation trim, 0.70 m/s target, v15 body, seed 20261111, 32 lane starts, five commands, 1600 steps and sampling.

Prediction: at least one offset decreases -0.03-lane median maximum heading error by >=0.05 rad, with no first-episode failures in those lanes, median maximum forward-speed error <=0.25 m/s, and median maximum height error <=0.20 m. Report all lane failures, metrics and unchanged strict screen. All simulator steps must succeed and complete first-episode samples are required. Stop after these two variants. Exploration only; no validation or freezing.
