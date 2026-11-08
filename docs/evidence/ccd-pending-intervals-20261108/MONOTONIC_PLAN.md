# Feasibility of a whole-sweep monotonicity certificate

After the independent geometry reconstruction, examine a new analytical
certificate as an offline FP64 feasibility calculation only. For each box
corner, use the quadratic unnormalized quaternion rotation numerator R(alpha)
and denominator D(alpha) for the captured nlerp path. A conservative simple
clearance lower bound uses the initial center when its displacement is
nonnegative. If that quadratic cannot be proved positive by its Bernstein
coefficients, investigate whether a positive initial guarded gap plus a
nonnegative derivative over the whole interval can certify clearance.

For positive center displacement and shortest-arc unit-direction quaternion
interpolation, physical-time derivative is kappa/D with kappa >= dot(q0,q1).
Thus the derivative numerator R'D-RD' + centerDelta*kappa*D can be bounded
below by using dot(q0,q1). Check the lower-bound quadratic in Bernstein form.
Record all coefficients and all eight corners. This is ordinary FP64 algebra,
not outward-rounded Metal interval arithmetic or a production safety proof.
A passing numerical screen does not authorize acceptance, cap changes or
promotion; a negative coefficient does not by itself prove real intersection.
