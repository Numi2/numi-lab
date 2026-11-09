# Independent dynamic-slider reference, before native integration

This is a reference for a future full Matter/MetalWorld transaction, not a
replacement for that transaction. Use the same n=2 FEM cube, geometry,
material, gravity, reference-area pressure law, activation width and initial
200 micrometre gap as the fixed-plane fixture. Give the plane a genuine 10 g
mass constrained to vertical translation, initial vertical speed +0.02 m/s,
and gravity. The body reacts to contact; its motion is not prescribed. The
slider holds its orientation and lateral position, and its support must supply
the corresponding constraint moment at zero work. Offset the contact patch
from the slider axis so the torque about a fixed origin is nonzero.

Evaluate the coupled semidiscrete equations independently in FP64 with RK4
at 0.5 and 0.25 microseconds for 10 ms. Verify total reference area, positive
clearance, equal-and-opposite interface impulse, equal-and-opposite moment
about the same fixed origin, whole-system vertical momentum balance including
gravity, and energy including elastic, gravity, kinetic and barrier terms.
Retain relative interface work, force bins, peak, cumulative impulse and
body motion. Use the original temporal gates for force and impulse; add
body-displacement and velocity checks with absolute floors stated beside their
results. A temporally qualified CPU reference does not certify a native
midpoint transaction, spatial convergence, or speed. The actual native
feature stays rejected until both owners share a half-step candidate, full-step
publication, force law, clearance, residual certificate and rollback.
