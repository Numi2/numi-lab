# Second corrective actor: teacher-off development results

The fixed actor trained from three guided recoveries was evaluated from full starts on previously used J and I. Neither is held out. Both runs failed to finish; do not use them as successful demonstrations.

| Layout | Waypoints | Outcome | Minimum sampled clearance |
|---|---:|---|---:|
| J | 1/5 | Timeout at 7000, no reported fatal contact | 0 m at right wing / right gate |
| I | 1/5 | Right wing / right gate collision at step 1883 | 0.000842 m at fatal sample |

The previous actor reached two waypoints on each used layout, although it struck the slalom on J and had near-zero clearance on I. The second actor regressed on both. On I, pre-contact position was (1.2933, -0.2231, 0.8070) m and velocity (0.1947, -0.0194, -0.2744) m/s; full contact, attitude and policy actions are in route-diagnostics.json. Actions are high-level policy outputs, not measured low-level actuator efforts. Flight toe angles after the measured impulse-free window stayed below 0.000324 rad in J and 0.000314 rad in I. No supported landing or autonomous release was reached. In-trajectory imitation loss does not establish safe navigation. K/L remain untouched.
