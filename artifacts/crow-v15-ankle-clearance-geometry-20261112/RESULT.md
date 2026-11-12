# Frozen-pose foot-clearance geometry

Native forward kinematics recomputed all nine sole/toe collision boxes in each foot for 16 recorded endpoint poses from lane 5 of the failed small-stroke development run. With all other joints and the torso fixed, changing the ankle alone by +0.1 rad lifts the rear toe but pushes a front toe into the ground in these poses; changing it by -0.1 rad lowers the rear toe. Changing the knee alone similarly tilts the foot. Thus rear-toe clearance alone is misleading.

For a paired +0.2 rad knee and -0.2 rad ankle perturbation, the minimum clearance of the whole foot was 2.5–4.6 mm in the sampled poses. This is a geometric intervention in recorded poses, not an actuator trajectory, dynamic stability result, or measurement of real corvid joint angles. It motivates a coupled controller experiment; it does not establish that the live controller can achieve these angles or that the other foot remains supported.
