# Analytical P1/P2 linearized standing-wave comparison

This is a contact-free diagnostic of bulk dynamics, not the free-sided
impact trajectory. Use the same Neo-Hookean small-strain parameters and
density on the 20 mm cube. Constrain lateral displacement, leave its normal
end faces free, and set u_z=A cos(8 pi z/L), zero velocity, with A arbitrary.
The exact continuum angular frequency is (8 pi/L)*sqrt((lambda+2 mu)/rho);
the exact initial acceleration is -omega² u. This is the exact *linearized*
continuum problem, not the finite-amplitude nonlinear solution.

## Prediction registered before execution

Assemble independent scikit-fem P1 and P2 linearized stiffness and consistent
mass on n=4 and n=8. Include the lumped P1 mass used by the earlier FP64
diagnostic as a separate comparison. For each interpolate the same exact
mode onto the element DOFs. Compute the Rayleigh angular-frequency error and
the initial-acceleration error in the mass norm, using a converged SPD mass
solve. Require positive mass and iterative-solve relative residual <=1e-9.

Predict that at n=8 P2's mass-normalized acceleration error is smaller than
both consistent-mass and lumped-mass P1 errors. Retain all results even if
the prediction fails. The 0.25% dynamic impact force and impulse gates stay
unchanged; this spectral initial-state test cannot qualify them or establish
full-trajectory phase error. A Rayleigh quotient is not a single eigenfrequency
when the interpolated shape excites other modes.

Run four assemblies (P1/P2 n=4/8) in one sequential process, with a 180-second
timeout, one BLAS thread, no GPU, and preserve output and solver failures.
