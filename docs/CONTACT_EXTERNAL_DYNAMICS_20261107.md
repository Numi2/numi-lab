# External P1 DOP853 dynamic comparison

Use the separately assembled scikit-fem P1 energy, force and nodal
regularized plane law, and integrate the same 27-node, free-sided geometry,
mass, gravity, initial state and fixed-law parameters with SciPy DOP853.
The retained n=2 RK4 h=0.5 µs fixed-law trajectory is an independent
implementation comparison. This is not a spatial reference or real-material
validation; both discretizations are P1 on the same mesh.

## Prediction registered before runs

Compare coarse and fine DOP853 runs for 10 ms; use max steps 8 and 4 µs,
relative tolerances 1e-8 and 1e-10, and respective position absolute
tolerances 1e-12 and 1e-13 m (velocity 1000 times those values). All other
state components use the position absolute tolerance. Compute contact
impulse and contact work in the ODE, with 1 ms force bins from differences
in the impulse accumulator. Require their force bins and total impulse to
agree within the unchanged 0.25% thresholds, using the fine trajectory's
peak bin and impulse for normalization. Both must stay clear at all sampled
times, and maximum global momentum imbalance must be <=0.25% of total
contact impulse.

If those prerequisites pass, compare the fine external force bins and
impulse against the retained RK4 0.5 µs n=2 result at the same 0.25%
thresholds. A passing comparison verifies this coarse-mesh numerical
implementation only; no claim of mesh convergence follows. Preserve failures
and all raw traces. Limit each run to 240 seconds and stop on a failed
execution. Disable BLAS threading to bound contention. No GPU.
