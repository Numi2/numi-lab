# Independent P2 n=2 dynamic diagnostic

The P2 mass, force, power and two-order quadrature checks passed at the
declared initial and active states. This plan evaluates a complete 10 ms
trajectory with the same physical block and regularized continuum contact
energy. Its consistent mass and order-8 surface quadrature are different
from P1 nodal contact discretization. It is a CPU research reference, not
the production solver.

## Prediction and acceptance checks registered before execution

Use DOP853 and independently refine max step from 8 to 4 µs, rtol 1e-8 to
1e-10, displacement atol 1e-12 to 1e-13 m and velocity atol 1000 times
greater. Integrate continuum contact impulse and assembled nodal contact
work; save snapshots every 0.1 ms. The unchanged 0.25% limits apply to the
maximum difference in 1 ms force bins normalized by fine peak bin and total
impulse normalized by fine impulse. At every solver evaluation reject
nonpositive order-8 surface quadrature gap or volume Jacobian.

At all snapshots also evaluate order-12 surface quadrature and require
positive gap and a normal-force difference <=0.25% of the fine peak bin.
Check global vertical momentum balance <=0.25% of the fine impulse.
Check total-energy drift <=0.25% of the free-fall kinetic scale
0.5*m*(g*0.01)^2. Contact work plus change of contact potential must be
<=0.25% of the maximum sampled contact potential or absolute contact work,
with floor 1e-12 J. Compare coarse and fine contact work using the same
scale and 0.25% tolerance. These are numerical diagnostics, not a physical
validation or production acceptance test.

The quadrature and sampled-time checks do not prove clearance between
quadrature points or between solver evaluations. Report that limit even if
they pass. The n=2 trajectory is not spatially qualified; a separate
preregistered n=4 and finer comparison is required before any spatial
reference or impact-force conclusion. Run coarse then fine with 240/480-second
process-group timeouts, single-thread BLAS, and no GPU. Retain failures.

The first coarse invocation completed integration but failed at JSON
serialization because the DOF count was a NumPy int32; it emitted no trace.
Its exception and process information are retained, and the fine run did
not start. A separately recorded attempt corrects the output type only.
