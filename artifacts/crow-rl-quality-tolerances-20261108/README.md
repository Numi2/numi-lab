# Explicit quality-tolerance comparison on the coupled crow

The earlier diagnostic fixture hardcodes optimality 1e-10 and feasibility 1e-7.
Current MetalWorld API defaults are 1e-5 and 2e-5. This experiment preserves
the strict arm and explicitly compares both API defaults and three fixed
optimality controls; it does not change production or silently relax a gate.

Every arm uses the same compiled coupled motor templates, current runtime,
four initial poses, actions, 32 substeps, 160 requested ticks and disabled CCD.
The API-default arm leaves MetalWorldQualityConfig untouched. opt7/opt6/opt5
keep feasibility 1e-7 and select optimality 1e-7/1e-6/1e-5. Two repeats test
determinism only. No extra seed, cadence, controller or budget tuning follows
the result. Capture first numerical failures and retain physical termination.

Primary physical prediction: API defaults complete 640 transitions with no
physics failure or task termination and maximum sampled jaw penetration <=0.8mm.
The numerical gate alone cannot support success. Sampled control-boundary gaps
do not certify continuous collision avoidance, biological validity, manipulation
causality or broad robustness. This is one surrogate fixture, not feeding.

Strict reproduction must preserve the previously retained substep-7 capture.
Matched arms isolate optimality; opt5 versus API-default isolates feasibility.
No solver/state code, deployed policy or default profile is modified.
