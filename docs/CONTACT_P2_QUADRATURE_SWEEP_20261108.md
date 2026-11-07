# P2 trajectory-state surface quadrature diagnosis

The P2 order-8 trajectory failed its force quadrature check against order 12
even though the initial manufactured active state passed. This study uses the
retained fine trajectory states without integrating again. It cannot repair
or qualify that trajectory.

## Prediction registered before execution

At all 101 retained 0.1 ms snapshots evaluate the same P2 surface gap and
finite-band plane pressure using scikit-fem triangle rules 8, 12, 14, 16,
18, and independently constructed positive tensor-product Gauss-Legendre
Duffy rules of orders 48 and 64 on each triangle. Retain both reference
rules, their point counts and source identities.

Require order-48 versus order-64 maximum total-force difference <=0.05% of
the retained fine peak 1 ms bin, and reproduce the original order-8 and
order-12 recorded force within 1e-10 relative to that peak. Predict that
both rules 16 and 18 differ from the order-64 force by <=0.25% of that
peak. If the reference check fails or either prediction fails, retain the
failure and do not select an unqualified runtime quadrature. Check positive
gaps at evaluated points only. The unchanged 0.25% dynamic force and impulse
gates remain open; this sweep is not a new trajectory.

Use one CPU process, a 180-second timeout, one BLAS thread and no GPU. The
fixed states come from the failed order-8 trajectory; quadrature
convergence on them cannot establish the trajectory obtained when a higher
rule also changes forces and motion.

## Result

All sampled gaps were positive. The smallest was 82.04 µm, compared with
85.05 µm from the earlier order-8/order-12 sampling; neither establishes a
global minimum between quadrature points or times. Original order-8 and
order-12 forces were reproduced within 6.13e-16 of the retained peak bin.

The order-48 and order-64 Duffy references differed by at most 0.000541%
of that peak, satisfying the 0.05% check. Relative to Duffy-64, maxima were
1.4450% for order 8, 0.4168% for order 12, 0.2939% for order 14, 0.1233%
for order 16 and 0.09191% for order 18. The prediction for orders 16 and 18
passed for these retained states.

This is a numerical quadrature study on the previously failed trajectory,
not a new integration. Changing surface quadrature changes its forces and
therefore its trajectory; a fresh complete trajectory must repeat temporal,
quadrature, force, impulse, work, clearance and energy checks. Spatial
qualification also remains open.

[Evidence](evidence/contact-p2-quadrature-sweep-20261108/) retains each
sample and rule, independent Duffy construction, hashes, timings and gates.
