# Fixed-input free-sided bulk response diagnostic

To isolate part of the bulk mesh response, independently integrate the
original free-sided Neo-Hookean block under the same prescribed, spatially
uniform bottom traction at n=16 and n=32. Material, density, geometry,
gravity and initial state are unchanged. The contact law is inactive; the
bottom load is a diagnostic external force, not a simulation of contact.
The prescribed total force is the retained n=32 nodal-contact force sampled
at 0.1 ms, linearly interpolated in time and distributed by reference
surface area. It is **not an independent physical force reference** and its
different spatial distribution and interpolated timing mean this experiment
does not reproduce either original contact trajectory. A missing plane permits
crossing the plane in this artificial loaded problem and is not a clearance
test or a valid contact result.

## Prediction registered before running

At identical prescribed loading, if volume-wave discretization materially
changes the surface response, the maximum n=16 versus n=32 area-averaged
surface displacement difference will exceed 0.25% of the 100 µm contact
activation distance. The threshold is a diagnostic scale, not a force
qualification gate: the nonlinear contact pressure may be sensitive even
to smaller differences, and larger differences do not assign a fraction of
the original impact discrepancy.

Run n=16 and n=32, h=2 and 1 µs, duration 10 ms, trace at 0.1 ms. For each
mesh compare maximum area-averaged surface displacement difference between
timesteps against 0.25% of 100 µm before interpreting spatial differences.
Record mean and RMS surface displacement, surface velocity, total momentum,
center of mass, and input impulse. Momentum must agree with prescribed
impulse plus gravity, with discrepancy below 0.25% of total input impulse.
Use positive-duration RK4 with the same source model and independent body
state, 180-second timeout per run, and stop at the first execution failure.
The unchanged 1 ms force-bin and integrated impulse qualification gates
remain 0.25% and cannot be passed by this test.

## Result

All four runs completed under budget, in 6.8 and 13.3 seconds for n=16,
55.7 and 112.2 seconds for n=32. The maximum 2 µs versus 1 µs change in
mean surface displacement was 2.17e-15 m at n=16 and 1.34e-14 m at n=32,
passing the registered temporal check. Momentum balance relative to total
prescribed impulse differed by at most 7.51e-15 and 3.10e-13, respectively.

The maximum n=16 versus n=32 mean surface displacement difference was
4.968 µm at 4.8 ms, or 4.968% of the 100 µm diagnostic length scale.
This exceeds the preregistered 0.25% diagnostic threshold. The maximum
difference between RMS surface displacements was 0.988 µm. Thus a substantial
mesh-dependent bulk response appears with identical applied total traction,
without contact feedback. This is consistent with the separate standing-wave
dispersion result and gives a specific bulk-resolution effect for a
force-derived transient.

The load is prescribed from a numerical contact trace, not an independent
physical force reference. Its uniform spatial distribution and temporal
interpolation differ from the original interaction, and no contact geometry
is enforced here. The displacement comparison cannot identify how much of
the original 12.43% force-bin discrepancy comes from bulk discretization
versus contact or their feedback. Neither original force nor impulse
qualification gate is met.

[Evidence](evidence/contact-prescribed-load-20261107/) contains the immutable
input, hashes, exact run identities, raw traces, and calculation. No GPU
or performance comparison was involved.
