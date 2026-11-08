# Spatial accuracy, cost and dynamic reaction: pretrial plan

Registered before these trials, based on `20de95448`. Question: do fixed-law
midpoint FEM-plane rollouts converge spatially, and does either integrator
advance more physical time per wall second at the same accuracy? Separately,
can a genuine dynamic rigid owner react to contact in a complete transaction?
The earlier 0.085% result is a 1 ms-binned force result, not peak accuracy.

Keep the 20 mm cube, density 1100 kg/m3, mu 2000 Pa, lambda 5000 Pa, gravity
9.81 m/s2, initial gap 200 um, zero velocity, plane, 100 um activation width,
1e16 N/m5 reference-area coefficient, 0.1 um clearance and 10 ms duration
unchanged. Mesh edge divisions 2, 4, 8 (27, 125, 729 nodes). Verify area sum
against 0.0004 m2 and independently reconstructed nodal weights. Use the
existing independent FP64 explicit RK4 reference at 0.5 and 0.25 us per mesh,
recording force and cumulative impulse at a fixed 2 us clock as well as the
true sampled peak at every RK4 step. Accept a temporal reference only if the
0.5/0.25 us force-bin and total-impulse differences are <=0.25%, peak force
and maximum cumulative-impulse difference <=1% (cumulative normalized by
final impulse). Compare the finer temporally qualified mesh references using
the same gates; only claim spatial qualification if the 4-to-8 result passes.
If a reference fails or cannot be run, retain it and mark its comparison
inconclusive; do not interpret an unqualified pair as convergence.

For each mesh, run midpoint and backward Euler with identical reference-area
contact law, geometry, physical solver tolerances, and outer transaction.
Compare 1 ms-binned force and total impulse within 0.25%, and peak force and
maximum cumulative-impulse difference within 1%, all relative to that mesh's
qualified 0.25 us reference. Require all requested physical steps accepted,
positive certified clearance, and independent momentum/force-law gates.
Measure peak as the maximum of step contact forces; report its sampling
resolution. Compare cumulative impulse by linear interpolation of the
step-integrated impulse at the reference's 2 us times. Timesteps are a bounded
dyadic ladder 125, 62.5, 31.25, 15.625, 7.8125 and 3.90625 us; choose the
largest tested step passing all gates. No claim of the continuous largest
admissible step is implied. Failure, budget exhaustion and no eligible step
are results. Safety cap: 2560 physical steps per 10 ms run; 8 divisions maximum.

Benchmark each qualifying selected integrator/mesh with three complete 10 ms
rollouts, alternating order, no per-step diagnostic snapshots or trajectory
readback, same native binary/shader, capture policy and physical fixture.
Include all failed/rejected work in wall time; reset/initialization and final
snapshot are outside timed interval and identical across arms. Record accepted
simulated seconds/wall second and completed steps; compare medians and all
individual runs. A practical-benefit claim requires >=10% improvement in the
median and every paired midpoint run faster; otherwise report inconclusive or
no demonstrated benefit. Do not infer performance from solver iterations or a
snapshot-instrumented run. Keep GPU execution serial, bind versions/hashes.

Dynamic extension: do not substitute a prescribed support or write a second
host physics owner. First inspect the existing free-body and articulated
candidate contracts and identify the actual borrowed MetalWorld transaction.
Only extend the scoped midpoint path if it evaluates both bodies at consistent
midpoint poses with a common force/impulse authority; certify equal/opposite
linear impulse, angular impulse about a common origin, interface work and
whole-environment rejection, restoration and deterministic retry. Leave the
mode fail-closed until that complete native test passes. Retain a precise
unsupported boundary if a valid coupled implementation cannot be established.
Failure category, acceptance and restored physical state must be invariant;
first offending-element index is diagnostic and must not govern future steps.
