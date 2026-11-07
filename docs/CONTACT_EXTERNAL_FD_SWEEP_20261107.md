# External energy directional derivative step-size diagnosis

The initial state-319 energy directional check failed at h=1e-8 m, with
1.5641e-5 relative error. Before running this diagnosis, predict that the
error is finite-difference truncation: for decreasing h in 4e-8, 2e-8,
1e-8, 5e-9, 2.5e-9, 1.25e-9, and 6.25e-10 m, the central difference error
should fall approximately quadratically before roundoff dominates. Require
two consecutive values <=2e-6 and each accompanying tangent finite-difference
comparison <=2e-6. Otherwise retain a failed or inconclusive result. The
original failed 1e-8 check and threshold remain unchanged. Use exactly the
same retained state, seeded direction, mathematical energy and assembled
residual; this is a diagnostic, not a new force or impact qualification.
