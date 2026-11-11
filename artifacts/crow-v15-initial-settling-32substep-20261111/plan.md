# Initial settling time-step sensitivity

Repeat the same eight development starts for 120 intervals at 32 rather than 16 physics substeps, with identical initial policy weights, v15 body, actuation and collisions. Record all toe substep velocities and compare cap counts. This is a time-step sensitivity diagnostic only; the release controller assumes 16 substeps and is never reached here. A change in contact behavior requires independently requalifying walking, takeoff, flight and landing, not treating a higher substep count as a fix.
