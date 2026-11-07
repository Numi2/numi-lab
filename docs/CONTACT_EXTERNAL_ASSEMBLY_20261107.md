# External assembly prerequisites for an independent reference

This is an assembly parity investigation, not an independent impact
trajectory or spatially qualified reference. The pinned Python 3.13 stack is
NumPy 2.3.4, SciPy 1.16.3, and scikit-fem 11.0.0; it constructs a P1
tetrahedral volume basis and independently integrates shape functions over
the bottom facets. It uses the retained 27-node fixture topology, material
parameters and native snapshots, but does not call the C++ FEM model or
runtime when assembling energy, residual and tangent. The contact law is
implemented separately at nodal reference-area weights. The C++ reference
probe is a comparison, not a dependency of the external assembly.

## Checks specified before the retained comparison

At retained native states 0, 150, 200, and 319, compare external energy and
residual to the independent C++ FEM probe for the same state and a seeded
direction. Require energy relative difference <=1e-8, residual L2 relative
difference <=1e-8, and analytical tangent-vector vs C++ centered finite
difference <=1e-6. Independently compare external energy's directional
finite difference with its assembled residual, and residual's centered finite
difference with its analytical tangent, both <=2e-6 relative. Also test
affine analytic stored energy, uniform active-plane potential and summed
normal force against explicit closed forms. Report raw values and failure;
these tolerances establish implementation parity only, not the unchanged
0.25% force and impulse qualification gates.

The exact energy density is mu/2*(tr(FᵀF)-3)-mu*log(J)+lambda/2*log(J)^2.
The separately assembled nodal plane energy is
-k*(g²-d²)^2*log(g²/d²), zero when g>=d, integrated by P1 boundary nodal
weights. Pointwise energy law is identical, including its finite band.
The n=2 topology and native node positions are from a retained earlier
native run; its model reference coordinates are reconstructed at full
precision, as in the prior FP64 comparison. This does not verify n=32 or
n=64 spatial behavior.

The initially pinned SciPy 1.15.3 package installed but its bundled PROPACK
failed to load on this host due to a Mach-O zero-fill section error. That
failure is retained; the separate pinned SciPy 1.16.3 stack imports.

## Initial result and retained failure

The initial four-state run passes the analytical affine-energy and uniform
contact controls and the external-to-C++ energy, residual and tangent checks.
However, state 319 fails the registered external directional energy
finite-difference check: 1.5641e-5 relative, above 2e-6. States 0, 150 and
200 pass all checks. The complete first-run data and failure are retained;
assembly qualification is not complete. Investigating whether this is
finite-difference cancellation or an actual inconsistency requires a separate
step-size study, not silently changing the declared threshold.
