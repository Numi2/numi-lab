# Preregistered retrospective representability diagnostic

Question: can world-space FP32 published positions satisfy the existing local
momentum, velocity and position gates simultaneously on the retained
straight, x-axis eight-node fixed-support fixture at origin x = 0.25 m?
Competing explanation: the solver could choose a different representable
configuration within the position gate and satisfy the other gates without a
state-representation change.

Use the accepted guarded record from `../rod-local-momentum-20261108/analysis-v3.json`
without changing its data. Compute an independent FP64 linear implicit-Euler
reference for this straight, no-contact stretch problem. Enumerate every FP32
world x coordinate within 1 micrometre of each reference free-node position;
pinned positions are the captured FP32 support coordinates. Exhaustively
minimize the worst of (a) momentum residual, (b) kinematic velocity residual
and (c) velocity error against the reference, in m/s, over all position
combinations and arbitrary real free-node velocities. Dynamic programming on
adjacent coordinate pairs makes the finite chain search exact for this scalar
nearest-neighbour model. The arbitrary-real velocity relaxation is more
permissive than the runtime's FP32 velocity representation. The existing limit
for all three quantities is 0.001 m/s. Prediction: the optimum exceeds that
limit, so no solution in this specified discrete representation and domain can
meet all gates. If it does not, reject that prediction.

As a constructive comparison, examine nearby FP32 coordinates expressed
relative to the fixed origin and reconstructed in FP64 for analysis. This is a
candidate representation only, not the current runtime or a physical result.
Check whether a feasible configuration exists using the same gates. Also check
that the analytic stretch residual at the retained published state agrees
within 1e-4 m/s with the separately retained FP64 total-energy finite difference;
the fixture is straight with no contact. Verify the independent reference's
linear equations below 1e-8 m/s and compare the DP with brute force on a small
three-node synthetic chain. Fail closed on malformed input, nonlinear edge
reversal, differing material stiffness, non-finite values, failed calibration,
or an incomplete enumeration. No measured GPU trial or production change is
implied. This is retrospective mechanism analysis, not a fresh held-out test.

Stop after this one fixed-support case and the prescribed controls; do not
widen position, velocity or momentum gates to obtain a result. The result may
motivate a persistent local or compensated position representation; it cannot
establish its coupled behavior, collision accuracy, replay or generality.
