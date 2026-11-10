# Recovery-trained actor teacher-off development results

The fixed candidate was run from full physical starts with the teacher disabled on two previously used layouts.

| Layout | Reached | First physical failure | Minimum sampled clearance |
|---|---:|---|---:|
| J | 2/5 | Left wing / slalom A, step 2312 | 0 m |
| I | 3/5 | Tail / slalom B, step 3015 | 0.000420 m |

I reached one more waypoint than the previous actor, then reversed into the slalom; J regressed from timeout without fatal contact to a wing strike. Both are failed episodes, with no supported landing. The recorded preceding root pose, velocities, policy actions and collision contacts are in route-diagnostics.json; policy actions are not measured low-level efforts. The maximum toe angle after the sampled impulse-free window was 0.000432 rad on J and 0.000395 rad on I. Neither this development test nor its in-trajectory training scores establish success on untouched K/L.
