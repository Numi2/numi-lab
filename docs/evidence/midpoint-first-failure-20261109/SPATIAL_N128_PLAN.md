# Bounded 128-division reference follow-up

The 32-to-64 spatial comparison failed all four preregistered gates; this
follow-up tests whether the same physical problem changes less under a
further doubling. It cannot establish physical calibration. Keep the
geometry, initial state, material coefficients, activation width, nodal
reference-area pressure law, duration and accuracy gates unchanged. Compare
0.5 and 0.25 microsecond references at 128 divisions first; a temporal failure
makes the spatial comparison inconclusive and calls for further temporal
refinement rather than a relaxed gate. Only compare 64 and 128 at a timestep
that independently passes the temporal gates for each mesh. Retain both force
bins and sampled peak separately, along with total and cumulative impulse,
minimum gap and area. A passing comparison covers this fixture, not other
materials or motions.

The independent ordered parallel evaluator is the previously parity-checked
implementation. The new capacity-check option does not change its equations.
Verify a small existing trace byte-for-byte and rerun the energy-gradient
self-test against the exact new binary before interpreting any reference.
First allocate a 128-division model and evaluate its initial force once;
record resource usage, wall time, total mass, reference area, and status.
Do not start full 128-division rollouts if this capacity probe is rejected or
uses more than 6 GiB resident memory, the data volume drops below 3 GiB free,
or the temporal execution estimate would exceed the bounded available run.
Any stopped or failed attempt remains evidence. Native Matter runs and speed
claims are separate work. There is no change to the runtime contact law.
