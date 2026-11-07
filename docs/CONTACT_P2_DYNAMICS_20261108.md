# Independent P2 dynamics prerequisites

This is a research reference, not a Matter implementation or a promotion.
It uses the same 20 mm geometry, density, material, gravity, initial gap and
continuum finite-band plane energy. It changes the spatial discretization:
quadratic tetrahedral displacement, consistent (not lumped) mass and
order-8 contact-surface quadrature. Therefore it does not implement the
same P1 nodal discrete contact law, although the continuum potential is
unchanged.

## Preregistered prerequisite checks

For n=2, assemble exact consistent P2 mass by volume quadrature and require
symmetry, strict positive definiteness, integrated mass 8.8 g within
1e-10 relative, and constant gravity acceleration within 1e-9 relative.
At initial and nonuniform active-contact P2 states, require positive
Jacobian and quadrature gaps. Compare assembled total energy directional
derivative with a centered energy difference, and the contact power
f_contact·v with the negative contact-energy directional derivative, both
within 2e-6 relative. Compare total contact force and full nodal contact
force from order-8 and order-12 surface quadrature on the active state
within 0.25%. These checks qualify only the chosen test states; they do
not prove dynamics or all-time nonpenetration. Stop on any failure.

A later dynamic plan must separately specify time refinement, force bins,
impulse, contact work, quadrature-point clearance, global momentum and
spatial refinement gates before execution. Use the unchanged 0.25% dynamic
force-bin and integrated-impulse thresholds. Do not infer hard-contact,
replay, production or physical-material qualification.
