# Native midpoint near-contact follow-up

On Apple M6 the default-disabled stage-precision candidate completed all
2,560 steps of the preregistered 10 ms, eight-division near-contact case at
3.90625 microseconds. The initial 95 micrometre gap is inside the unchanged
100 micrometre activation band; the reference initially carries 0.232086 N.
The native calculation retained the reference-area contact law and its
original physical parameters and error limits.

The mesh-specific, temporally qualified FP64 reference comparison passes:
1 ms-binned force error 0.004807%, sampled peak error 0.06394%, total impulse
error 0.00004349%, and maximum cumulative impulse error 0.001495%. The
reference sampled peak includes the initial state, whereas the native peak is
sampled at accepted steps, at 3.90625 microsecond intervals; this difference in
sampling is included in the reported comparison. The native minimum gap was
92.0736 micrometres and the area sum was 0.0003999999808 m². The independently
checked maximum local scaled momentum residual without a unit denominator floor
was 0.0002870, below 0.005; the largest kinematic defect was 2.46e-18 m.

Continuation replay, full restore replay, fingerprint mismatch rejection and
forced rollback passed exactly for the captured physical state and status.
Normal contact work was +2.63692262e-7 J, compared with +2.63692249e-7 J from
an independently accumulated impulse/midpoint-velocity product. This begins
with stored barrier potential and is not a complete energy balance or proof
of energy injection. The original physics input executable and Metal library
are identified in `stage-native-001-inputs.sha256`; the execution record,
full trace, logs and analysis are retained. The measured wall time includes
diagnostic capture and replay and is not a performance benchmark. This is a
single fixed-plane mesh, not a spatially converged or two-way dynamics result.
