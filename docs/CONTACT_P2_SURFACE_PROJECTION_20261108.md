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
