# Drive-target audit: -0.10 action request

The logger-only rerun exactly reproduced all 3,208 recorded endpoint fields and 5,888 recorded contact-substep fields of the preceding observer, with no recorded overflow or substep errors. All 368 required target rows were logged. The copy was taken at the first substep of each control interval, from the position-target trajectory consumed by the solver, before its own joint-limit clamp.

Across both knees, eight lanes and steps 335–380, the copied targets ranged from 0.205000028 to 1.02999997 rad and none lay outside the native model knee limits [-0.10, 1.55] rad. This is a measurement of one used-seed -0.10 normalized-request condition; it does not establish the target trajectory of the previous -0.70 action condition. It does not qualify a leg-powered takeoff. All per-lane, per-step targets and joint positions are in analysis.json.

The follow-on SLOW-PLAN.md separately prespecifies measuring the earlier request using its own retained shader and field-parity reference.
