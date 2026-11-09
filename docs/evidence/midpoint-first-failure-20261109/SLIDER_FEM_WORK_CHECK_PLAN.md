# FEM-side interface-work diagnostic check

The retained slider diagnostics reconstruct the rigid-side work from both the
measured physical wrench and the published rigid momentum change, but only
report the FEM-side work formed from contact impulses. Before any new native
slider qualification, independently recompute each FEM contact node normal
impulse from its reference area, unchanged pressure law, and reconstructed
midpoint gap in FP64. Multiply both the recorded nodal impulse and the
independent impulse by the nodal midpoint velocity. At every contact-area
node the absolute difference must be at most the existing impulse-balance
limit `max(1e-12 N s, 1e-5 * |independent nodal impulse|)` times that
node midpoint speed, plus the already used 1e-24 J arithmetic floor.
The maximum node-wise ratio must not exceed one; checking nodes separately
prevents cancellation between opposing errors.

First check one real, accepted, default-disabled diagnostic step for each
preregistered initial gap: 200 µm outside activation and 95 µm inside it, at
15.625 µs. The former must have zero FEM work and the latter nonzero force.
Record exact executable, library, shaders, source, command and outputs. This
is instrumentation validation, not a completed 10 ms rollout, a work/energy
accuracy qualification, a speed measurement, or permission to enable the
slider. Retain a failure rather than changing the tolerance. The synthetic
analysis test must also reject a corrupted FEM work gate ratio.


The first instrument check recorded only the maximum per-node ratio; its
rollout analyzer did not enforce that value. For the follow-up, retain every
contact-area node identity, measured impulse, independent law impulse and
midpoint velocity. The analyzer must independently reconstruct both work sums
and the maximum ratio from those records and reject missing, altered or
inconsistent data before applying the same unchanged bound. Repeat the same
two one-step native checks with a new named build and retain the earlier
outputs. This fixes the instrument and does not change the physical hypothesis.
