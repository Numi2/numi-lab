# Flight leg pose: kinematic search only

A bounded grid over the current v15-derived model's hip, knee and ankle joint limits shows that positive hip and knee flexion can raise the foot towards the body. Extreme grid winners are at model joint limits; they are not automatically safe or realistic. This is a CPU kinematic calculation with a fixed body, not a physical motion, collision test, flight qualification or biological calibration.

The follow-up candidate uses moderate bounded action targets rather than the extreme winners. It must be checked in actual flight, for collisions and toe speeds, and visually at multiple angles. The printed limits are model assumptions. The existing shared host runtime is archived in crow-v15-hip-roll-journey-integration-20261111.
