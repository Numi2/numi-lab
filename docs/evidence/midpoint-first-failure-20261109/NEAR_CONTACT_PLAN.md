# Near-contact fixed-law qualification plan

The prior initial gap of 200 micrometres is outside the 100 micrometre
activation band. Retain that result and separately test an initial gap of
95 micrometres, inside the band, with all other geometry, material, gravity,
reference-area coefficient and clearance unchanged. This is an intentionally
prestressed barrier configuration; positive contact work cannot be interpreted
without its stored contact potential and integration rule. Use n=8 and the
same 10 ms duration. Generate independent RK4 references at 0.5 and 0.25 us,
including the initial contact force in the reference peak. Require the
original temporal gates before comparing the finer reference against the
native 3.90625 us midpoint candidate with the same force-bin, peak, total and
cumulative-impulse gates and physical state/replay checks. The native first
force sample occurs at its step, whereas the reference includes t=0; retain
this sampling distinction. Do not claim spatial convergence or dynamic-body
behavior from this case.

Before running the near-contact reference, the extended reference must pass its
energy-gradient self-test and reproduce the prior unmodified n=8 0.25 us
output byte for byte when the extra gap argument is omitted. Record source and
binary hashes. Timeout, noncompletion, reference instability, or a failed gate
is retained as evidence. Native timing is not measured during this diagnostic.
