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

## Result

Energy derivative relative errors at 4e-8, 2e-8, 1e-8, 5e-9, 2.5e-9,
1.25e-9 and 6.25e-10 m were 2.501e-4, 6.251e-5, 1.564e-5,
3.904e-6, 8.723e-7, 1.922e-7 and 3.908e-7. The first four decrease at
approximately the expected second-order rate, and the last increase is
consistent with roundoff. Three consecutive smaller steps pass the
predeclared 2e-6 condition; their tangent finite differences also pass.
This supports finite-difference scale as the cause of the original failure,
but does not erase it or establish dynamic/spatial accuracy.

The registered raw sweep and identities are retained in
[evidence](evidence/contact-external-fd-sweep-20261107/).
