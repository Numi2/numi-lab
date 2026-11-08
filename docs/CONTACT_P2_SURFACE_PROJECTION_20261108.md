# Matched-state surface-resolution diagnostic

The P2 n=4 and n=8 force trajectories differ by 9.685% of the n=8 peak bin,
despite passing sampled surface quadrature checks. Both first show force at
4.6 ms on the 0.1 ms trace and share the free-fall activation time 4.515 ms;
neither trace resolves releases shorter than the sampling interval. This
does not identify the cause.

## Prediction registered before execution

At retained fine n=4 and n=8 trajectory snapshots 4.6, 5.0, 6.0, 8.0,
9.5 and 10.0 ms, project each P2 surface displacement into the other mesh's
P2 surface DOFs, leaving the original state untouched. Compare integrated
continuum plane force of original and projected gap using identical
Duffy-64 reference quadrature. Query interpolants 1e-12 m inside the bottom
face to avoid tetrahedron-finder boundary ambiguity. This is a prescribed
state transfer, not an integration or a change in contact law.

As an implementation control, the n=4→8 projection of the n=4 field should
reproduce its force within 0.05% of the retained n=8 peak 1 ms bin. For
n=8→4, predict at least one projected force difference exceeds 0.25% of
that peak. If not, retain the failed prediction. Require positive sampled
gaps and reproduce the original Duffy-64 force from each trace within
1e-10 of that peak. A larger projection error indicates a surface
representation effect at these states; it cannot allocate the original
trajectory difference between surface and bulk response or establish
accuracy of either mesh. A small error likewise cannot rule out contact
feedback. Dynamic force-bin and impulse limits stay at 0.25%.

Run one bounded CPU process for at most 180 seconds, one BLAS thread, no GPU.
Retain all original hashes, per-state forces, projection checks and failures.

## Result

All sampled gaps were positive and each original Duffy-64 force reproduced
the corresponding trace value exactly. The n=4→8 projection control changed
force by at most 9.67e-8 of the n=8 peak bin, well under 0.05%.

The opposite n=8→4 projection changed instantaneous force by up to
**44.67% of the n=8 peak 1 ms force bin** among the six sampled states
(at 5.0 ms), exceeding the 0.25% diagnostic threshold. At that state the
original Duffy-64 force was 0.11937 N and the projected-force value
0.08031 N. This supports a substantial surface-representation effect for
the observed fine-grid gap field. The operation changes that gap field, so
it does not apportion or establish the cause of the 9.685% original dynamic
bin discrepancy; bulk response and contact feedback remain coupled.

Both runs first show contact at the 4.6 ms sample. Their shared theoretical
free-fall activation is 4.515 ms. At 0.1 ms sampling each shows one episode
through 10 ms, with impulses 0.000413926 and 0.000408281 N·s, a 1.383%
difference normalized by n=8. Shorter unobserved separations are possible.
Order-18 vs Duffy-64 force error at the original n=4 and n=8 snapshots was
at most 0.01091% and 0.01226% of the corresponding peak bins; this bounds
instantaneous quadrature discrepancy at those states, not the accumulated
effect of changing quadrature during a new trajectory.

The [evidence](evidence/contact-p2-surface-projection-20261108/) contains
all sampled forces, input hashes, onset calculation and exact normalization.
The original 0.25% dynamic force and impulse qualification gates remain
unmet.
