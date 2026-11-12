# Physically reached supported reset pose

A lane-zero reference state at 2 s had both feet and several toes bearing ground impulse and root velocity under 0.0001 m/s; the full source state and contact records are retained. The experiment copied its root height and non-root joint positions at episode initialization, before the single set of randomized start offsets, without copying velocities or cached contact impulses. This is not a mid-episode reset.

Across eight previously used seed-20261127 starts, the first-four-second both-feet-unloaded count changed from 2,523 to 3,025 substeps compared with the normal single-perturbation reset. Some starts improved, but lane 5 deteriorated from 182 to 1,399 unloaded substeps. Four starts still hit a toe or neck speed limit early. All eight retained continuous foot loading in the 4–6 s interval and had no measured non-root joint-speed-limit hits from 4–10 s; all were airborne during 8–10 s and none reached a waypoint by 10 s. No nonfoot ground or scene contact impulses were recorded. Different initial conditions preclude exact trajectory parity.

This pose does not reliably solve contact initialization and is not promoted. These tests do not establish biological gait, qualified takeoff, flight, route completion or landing.

**Correction:** inspection of retained first-step states shows that a later reset operation overwrote the proposed root height: the bird still began near 0.185 m. The above comparison therefore tests the non-root pose change, not a successful supported-height initialization. The corrected height-order experiment is separate.
