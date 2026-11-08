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

## Result: all five tolerance arms reject

Strict reproduces the exact retained substep-7 failure. opt7 and opt6 reject
at substep8, with reported residuals .0196589 and .00259472. opt5 and API-default
reject at substep9 with residual .00108535. All arms exhaust20 Newton updates,
accept zero control transitions and roll back four lanes. All repeated physical
files are byte-identical. The API-default physical-completion prediction fails.
The empty traces contain no jaw-geometry observation; the raw analyzer's0.0
empty max is a sentinel, not measured zero penetration. physical-interpretation.json
makes this missingness explicit. No tolerance or physical gate is promoted.

Independent FP64 reprojection agrees closely for opt6/API-default, while opt7
shows17.23% residual discrepancy. The prediction of <=2% agreement across all
three is contradicted. This discrepancy requires further owning investigation;
it cannot be assigned to one arithmetic or projection defect from these data.
FP64 Newton converges on the retained problems in2/3/5 updates. Their unscaled
Hessian conditions include tiny uncoupled coordinates; diagonal-equilibrated
conditions are about8,254–13,471, so raw1e11 conditions alone do not explain failure.

The previously sealed native two-FP32 solver, recovered from actual Git,
solves all three in2/3/5 updates. Native and independent residuals are <=5.08e-15;
velocity differences from FP64 are <=1.92e-13, and repeats match byte for byte.
Rounding its outputs to FP32 gives residuals7.68586e-8 (opt6),4.57193e-8 (opt7)
and1.41332e-8 (API-default), each below that input's original tolerance.
This removes the need to expand canonical world precision merely to satisfy
these three declared gates. It does not solve the stricter1e-10 publication
problem, prove general representability, or establish a physical trajectory.
The split experiment also changes linear solver arithmetic/path, so it does not
isolate precision from direct versus PCG as the cause.

A syntax error in the first diagnostic runner prevented any native command.
Its source/plan/logs are retained; the corrected runner used the same six-command
budget, inputs and predictions. No failed physical trial was retried or extended.

Next work is a native checked refinement/publication path that recomputes the
certificate at the actual rounded output, compares against the unchanged declared
tolerances, and preserves transactional rejection when they cannot be met.
Then the complete physical fixture, reset/replay and end-to-end performance
must be qualified. The full lifelong crow goal remains unfinished.
