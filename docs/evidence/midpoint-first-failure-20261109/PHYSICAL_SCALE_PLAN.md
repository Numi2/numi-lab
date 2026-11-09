# Independent local residual scale: follow-up plan

The earlier diagnostic divided each nodal residual by its gravity impulse plus
an absolute 1e-10 N s floor. That floor is an arbitrary unit-dependent scale,
and cannot by itself establish a consistent local accuracy gate under changes
to mass or timestep. The runtime certificate and prior recorded results remain
unchanged. At the retained n=8 failure, recompute both diagnostics on the same
accepted state and reflected endpoint, including the net residual vector.

Before running a new comparison, instrument the independent FP64 evaluator to
also record each accepted step maximum nodal residual divided by m_i |g| h,
with no added floor. All nodes in this particular free-falling fixture have
positive mass and gravity. Keep the original 0.005 diagnostic limit and report
both values; a no-floor value above 0.005 invalidates this additional local
check. This is a fixture-specific scale, not a general-purpose contact criterion:
a separately defined scale would be needed for zero-gravity or constrained
nodes. Retain the existing independently checked global balance and the
original runtime acceptance requirements.

Re-run the previously qualified n=2 and n=4 midpoint h=15.625 us complete
10 ms trials, with the same geometry, material, contact law, solver and shader;
compare final physical hashes with the original rollouts. Compare the older and
new local diagnostics at each accepted step and preserve any rejection. Do not
change runtime tolerances or relabel earlier measurements. A passing local
check does not repair the failed spatial gate, establish a matched-cost result,
or qualify the unsupported dynamic-body transaction.
