# Backward Euler control precision repair

The retained precontact 1 microsecond failure is consistent with Newton
velocity corrections being rounded away. Test a default-disabled, tightly
scoped BE control that accumulates accepted and working velocity as high and
low FP32 limbs, reuses the existing compensated force/residual and trial
geometry machinery, and publishes the BE endpoint directly. The physical
contact law, material, geometry, timestep, nonlinear and linear tolerances,
and budgets are unchanged. This is a representation repair, not a change to
the backward-Euler or midpoint equations.

Keep the option restricted to the existing single-environment frictionless
fixed-plane FEM fixture, reject unsupported owners and edits, include the
mode and persistent low limbs in fingerprints and transactional state, and
preserve whole-step rejection and rollback. The BE candidate uses h throughout;
it must not use midpoint reflection or the midpoint residual. No existing
mode changes without the explicit research flag. Both separate augmented-
Lagrangian contact and interval-local CCD experiments remain disabled.

First reproduce free-fall and exact restore/continuation with the new mode;
then rerun the 1 microsecond step beyond its previous failure and through the
10 ms duration. Compare with the same n=2 temporal reference, original force
and impulse gates, independent per-node momentum and BE kinematic residuals,
clearance, and physical state restoration. Retain all failures and input
hashes. A passing short prefix is not a completed rollout; a completed rollout
that misses physical accuracy is not an accuracy-qualified comparison arm.
Instrumented times are not benchmarks. Any later speed comparison must use a
common law, the selected accuracy-qualified steps and complete no-snapshot
rollouts.
