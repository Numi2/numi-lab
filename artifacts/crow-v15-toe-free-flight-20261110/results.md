# Obstacle-collision-off toe-body screen

Across both seeds, 160/160 fixed-head, passive-toe lanes remained free of native physical and solver failures for the 1600-step screen; no lane passed the original speed-tracking gate. Maximum per-lane speed error ranged from 0.270 to 0.461 m/s against a 0.20 m/s limit. Heading and height are reported per lane; a positive separation between climb and descent is not by itself full qualification. Both controls matched the corresponding obstacle-on trajectories at all 960 q/v samples across steps 100–600, before the first observed course collision.

This establishes obstacle-free survival in this bounded screen, not controller qualification for the toe body. The fixed-head setup also differs from the earlier benchmark, so this result does not isolate toes as the cause of the tracking regression. Course visuals remain present while collision masks are disabled; this is not navigation, clearance, a landing test, or biological validation.
