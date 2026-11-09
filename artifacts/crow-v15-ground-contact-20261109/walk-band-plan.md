# Exploratory existing v15 walking-band control

After the 350-step navigation baseline found essentially no prelaunch travel,
compare the same v15 body, frozen flight law, actor, 2 lanes, seed and 350-step
horizon with difficulty band 1 selected. This asks whether the already present
alternating ground carrier transfers to the v15 navigation task without code or
body changes. This is an exploratory engineering check on a previously used
layout/seed, not held-out qualification. The run receives the identical
read-only contact observer already checked against exact unobserved replay.

A passing experimental gait must have at least 0.05 m net horizontal travel
in both lanes, positive solver-reported support from each foot, at least one
accepted sample with left-only loading and one with right-only loading, no
physical failure, and no failed simulator steps. A missing contact or failed
run is a failure, not an excluded sample. Contact rows are endpoint samples;
this does not establish continuous-time slip, force integrals, or biological
realism. Stop after one candidate and retain all results before selecting an
intervention. A pass would still require unused seeds, perturbations, actual
contact feedback, takeoff handoff and gripping qualification.
