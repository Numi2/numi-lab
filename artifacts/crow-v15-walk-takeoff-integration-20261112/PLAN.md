# Integrate the development walking and takeoff windows

The walking-only shader has a six-second gait interval and a launch guard at 10.3–11.4 s, but its launch interpolation and release logic remain at 6.3–7.4 s. Therefore its 4–6 s walk was never tested as part of the intended launch. This experiment ends the gait at 6 s and restores the launch guard to 6.3–7.4 s without changing the earlier walking interval or release law. The shader still uses the existing artificial flight controller and is not physically qualified.

Prediction: the retained walking run will match through 6 s; thereafter both feet should generate a measurable upward push before release, and leg and toe velocities should remain bounded. Record all eight lanes, both previously used seeds, every substep, and failure outcomes. A zero-load or unstable start is not a successful takeoff. The experiment does not establish downstroke ordering, free flight quality, navigation, landing or generalization.
