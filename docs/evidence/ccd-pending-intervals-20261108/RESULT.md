# Pending CCD intervals: predicted sweeps and a feasibility bound

Thirteen separately compiled diagnostic selectors captured the pending nodes,
projected box endpoints, stationary plane and geometric margin in the two
250 rad/s thin-impact fixtures. The uninstrumented candidate and every
selector agreed on both fixture results and on all 16 captured event records,
except the intended unresolved-pair diagnostic fields. A fresh targeted
baseline also matched the corresponding earlier 12-case candidate outputs,
excluding elapsed time. Repeats are included in the 16 event records; they
are not independent physical scenarios.

Independent FP64 reconstruction evaluates 100,001 positions per interval,
with a quaternion/translation Lipschitz sampling bound and an added 0.2 µm
numerical guard. For all eight corners over the complete predicted remainder,
the diagnostic lower bounds are 57.53 µm at scale 1 and 28.67 µm at scale
0.5, above the captured tolerance plus geometric guard of 12.34 µm and
11.17 µm respectively. The same holds on each pending interval. These are
rejected trial sweeps of a single floating box over a fixed plane. The
numerical allowance is not a formal bound for every Metal rounding operation,
and the observation neither accepts the step nor qualifies contact forces.

An additional offline FP64 feasibility screen combines two possible tests for
each corner: positive whole-interval Bernstein clearance, or a positive
initial guarded gap and nonnegative whole-interval derivative lower bound.
All eight corners passed one of these numerical tests for both sweeps. The
latter lower bound uses the exact nlerp-to-physical-time derivative and
kappa >= dot(q0,q1) for hemisphere-aligned quaternions. The smallest derivative
bound on the corners requiring monotonicity was 3.179e-4 at scale 1 and
1.590e-4 at scale 0.5 in the normalized numerator units. This is not yet an
outward-rounded implementation or a production safety proof. Next, implement
it as a separate candidate with explicit preconditions, fail-closed fallback,
interval arithmetic and adversarial checks, then test production cap 16 and
whole-step response before considering any promotion.

The original data, plans, scripts, input identities, executable and shader
hashes, raw traces and results are retained. Build commands depend on the
existing local AIR objects whose hashes are recorded; binaries are not all
included in this evidence directory. The source Collision.metal is from main
commit 48d061e3b. No production code or experimental selector was changed,
and no timing result is a performance measurement.
