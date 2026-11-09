# Stage-precision midpoint timestep ladder

Use the original fixed-law n=8 fixture, 10 ms duration, mesh-specific 0.25 us
reference, physical tolerances and original force/impulse, clearance, local
residual and transaction gates. Test the same descending dyadic ladder 125,
62.5, 31.25, 15.625, 7.8125 and 3.90625 us with both default-disabled
precision options enabled. Stop at the first passing complete 10 ms rollout;
the previously retained 3.90625 us result may supply the final ladder point.
For each attempt retain the full trace, errors, failure and restored state,
including the independent local residual without a unit floor. Report the
largest *tested* passing step only, not the largest possible step. Identical
physical law and geometry are required; this test is not a comparison with
backward Euler and cannot establish spatial accuracy or a speed benefit. The
native executable and shader are the hash-frozen `stage-native-001` inputs.
CPU reference work may run concurrently, so these instrumented runs are not
benchmarks. Stop on an unexpected process error or timeout and retain it.
