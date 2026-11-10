# Teacher-off full-start development comparison

A single-trajectory exploratory actor and its original deterministic actor were each run from a full physical reset with the teacher disabled, on the same previously used development seed for each layout. Each result below is one first episode, not a generalization estimate.

| Layout | Actor | Waypoints | First-episode result | Minimum sampled obstacle clearance |
|---|---|---:|---|---:|
| J | Original | 0/5 | Left wing struck slalom A at step 2356 | 0 m |
| J | Exploratory | 2/5 | Left wing struck slalom A at step 2194 | 0 m |
| I | Original | 0/5 | Timeout at step 7000, no physical failure | 0.1325 m |
| I | Exploratory | 2/5 | Timeout at step 7000, no physical failure | 0.0000815 m |

The exploratory actor increased reached waypoint count on these two starts but did not complete either route or land. Its near-zero I clearance is unsafe and is a regression relative to the control's clearance. On J, immediately before impact its body was at (2.5089, 0.1100, 1.2829) m with linear velocity (0.2135, 0.2857, 0.2780) m/s; the first fatal pair and full action/contact details are retained in route-diagnostics.json. The frozen flight law and current v15 toe body remained in use. Flight toe angles remained near zero after the measured impulse-free window, but no accepted-state end-to-end success or unseen-layout performance is implied.
