# Shared vertex-triangle witness trial 029

The existing compensated VF energy evaluator and rounded-world-point normal
used different geometric witnesses. Captured failure 028 has identity kind0
(vertex-triangle), root5/tick0/contact2941, Newton22. A shared compensated
witness now supplies energy separation, normal and support barycentrics.
The mapped parent conversion also preserves point-minus-triangle orientation
on either parent and records the triangle projection as the contact point.

Before native integration require the existing focused geometry checker,
including independently evaluated captured states, both mapped parent
orderings and a tiny-gap fallback; then the unified preparation/geometry/
force/replay/rollback regression. No physical/material/timestep/tolerance
relaxation. Keep the optional64-column existing solver override in this trial;
default32 remains unchanged.

Then request32x125us in the full scene. More than5 accepted rows is the minimum
evidence of passing the former027/028 failure point. Final acceptance still
requires every original physical certificate; a different early trajectory
is expected after a force-normal correction. Preserve any new failure.
This short numerical trial is not sustained supported-body qualification.
