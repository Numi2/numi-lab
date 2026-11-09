# Recompute-coupled endpoint rounding diagnostic

The exploratory frozen-geometry estimate lies below the physical residual
limit, but does not prove a valid corrected midpoint transaction exists.
Before running the following independent CPU diagnostic, predict that a small
fixed-point iteration can reduce the force-recomputed full-step residual below
the same limit on the captured precontact step while retaining the midpoint
kinematic equation. The test is scoped to this one immutable, frictionless
fixture and cannot qualify a native implementation or contact behavior.

At each iteration evaluate the independent FP64 force at the midpoint of the
accepted and current endpoint positions. Round the force-balanced full-step
endpoint velocity to its nearest FP32 value, then reconstruct each endpoint
position from x_n + h (v_n + v_{n+1}) / 2 using the existing high/low reference
displacement encoding. Recompute forces and the full residual after the change;
never apply the old backward-Euler endpoint residual. Stop after at most 12
iterations, or at a repeated discrete state. Record every iteration including
non-improvement, residual norm, maximum kinematic defect, minimum plane gap
and exact endpoint hash. Check the original residual cutoff without modifying
it. Require a finite positive gap and the original geometric clearance. The
calculation is exploratory; it must not write accepted Matter state or select
new runtime tolerances. A passing CPU candidate is not an accepted native
Matter step and does not resolve contact or full-scene qualification.
