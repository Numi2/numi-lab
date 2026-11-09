# Exploratory reversed-bank gain comparison

Following the heading-to-bank sign comparison, compare +1.40 and +2.80 heading gains against the +0.70 reversed-bank arm under the same 32 matched starts, development seed 20261111, 1600 steps, five command conditions, 0.70 m/s target, +0.55/-0.55 speed trim and existing bounded bank correction [-0.45,0.45]. Preserve all other controller and physics parameters.

Prediction: at least one arm decreases the median maximum absolute heading error in -0.03 rad/s lanes by >=0.10 rad compared with +0.70, without physical terminations in those lanes. Keep the prior strict sampled screen unchanged; record all physical failures, samples, speed and height errors and all command modes. Zero simulator failed steps and complete observations are validity conditions. Stop after these two exploratory arms; neither may be called qualified without separately preregistered perturbation and unused-seed validation.
