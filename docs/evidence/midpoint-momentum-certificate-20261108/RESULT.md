# Fixed-mesh midpoint impulse accounting

The independent CPU FP64 midpoint reference now computes a Euclidean impulse
residual for every node on every accepted step, using the accepted endpoint
velocities and freshly evaluated midpoint forces. It separately reconstructs the
whole-body impulse from the change in total momentum, gravitational impulse and
normal contact impulse. Internal element forces are excluded from that global
formula. The node forces are re-evaluated from the same independent reference
model, so this is an equation and bookkeeping check, not an independent
validation of its material or contact law.

The predeclared n=2, 10 ms fixture at 15.625 microseconds completed 640 steps on
the Apple M6. The maximum nodal impulse residual was 1.132262920324223e-12 N s;
the maximum independently reconstructed global residual was
2.250642512635011e-12 N s. Both met the fixture-specific 5e-10 N s limits.
The new arithmetic test also verifies that opposing local residuals can sum to
zero while the maximum local residual remains nonzero, and rejects a nonfinite
input. These limits are not a general dimensional tolerance policy.

All five selected CPU CTests passed, including the unchanged force and impulse,
clearance, energy and work gates. Against the previously retained fixed-mesh
FP64 RK4 result, the maximum 1 ms force-bin difference normalized by peak
reference force was 0.085494%, and the total impulse difference was 0.010997%.
They remain within the original 0.25% thresholds. The complete result is in
trajectory.json, the comparison in comparison.json, the test output in
ctest.txt and source/binary identities in inputs.sha256. Code under test is
196cca87b; its plan was committed in advance at 4df37e8db.

This is not a native Matter midpoint implementation, an independent check of
the physical constitutive law, spatial convergence, material calibration, or a
completed force-qualified control fixture. The original native backward-Euler
temporal-force gate remains failed. Contact impulse and work remain the
independent reference model's values, not a measurement of augmented-Lagrangian
contact. AL and the experimental CCD options remain disabled.
