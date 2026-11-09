# Native FEM and reacting-slider midpoint qualification plan

This plan is registered before any native coupled midpoint rollout. Its purpose
is to evaluate a two-owner transaction, not the already completed owner-only
slider test or the exploratory CPU comparison. The feature stays explicitly
opted in and unavailable for other geometries.

Use the 27-node, 10 g freely reacting one-axis vertical slider fixture from the
retained independent FP64 references. Run both the 200 µm initial gap and the
95 µm initially active gap. Keep material, gravity, physical pressure
coefficient, reference areas, 100 µm activation width and duration identical
to each corresponding reference. The slider state must evolve through
MetalWorld; prescribing its motion, postprocessing a fixed-plane reaction, or
integrating a second independent solver is not an acceptable substitute.

For the primary 200 µm case, run a complete 10 ms native rollout at
15.625 µs and 7.8125 µs, comparing each with its temporally qualified
0.25 µs FP64 reference. Separately run both steps for the 95 µm near-contact
case and compare against its own qualified reference. Check maximum 1 ms-bin
force error at most 0.25% of reference peak, sampled peak force error at most
1%, total impulse error at most 0.25%, and maximum cumulative-impulse error at
most 1% of the reference total. On the primary fixture also use the retained
0.25% relative gates for final slider position and velocity. Require every
accepted endpoint and the midpoint to exceed the existing 0.1 µm clearance.
All steps must be accepted with no suppressed failure, and each reported
physical residual must satisfy its stated absolute-plus-relative limit.

At each accepted step separately collect FEM-side and slider-side physical
contact impulse, moments about the same fixed origin, constraint reaction
moment, midpoint velocities and each owner's contact work. Check that
opposite impulses and total moments including the constrained body's reaction
balance to the larger of 1e-12 N s or 1e-5 of the reference contact impulse,
and 1e-14 N m s or 1e-5 of the reference contact moment respectively. The
torque tolerance is on integrated moment over a step, not instantaneous
torque. Reconstruct each owner's work from that owner's own impulse and
midpoint velocity; compare the resulting work histories and total against the
FP64 reference, reporting absolute error when the reference work approaches
zero. Record stored barrier potential and discrete work-plus-potential change
for the initially active case; positive contact work alone does not establish
energy creation. If the force or torque representation cannot supply an
independent owner measurement, fail this gate rather than deriving both sides
from the same sample.

After at least one nonzero contact step, force a late whole-environment
failure, verify the failure category and bytewise restoration of MetalWorld
q/v and all physically authoritative Matter state including nodal state,
low parts, material and contact history, scheduler and generalized candidate,
then retry from the restored state and compare the complete resulting physical
state byte-for-byte with an uninterrupted continuation. Also exercise
rejection prior to contact and the initially active fixture. The diagnostic
offender index need not be deterministic, but cannot affect acceptance,
failure category, restored physical state or later execution.

Keep intermediate solver traces and failed attempts, as well as source,
executable, shader, fixture and reference hashes. Report the first failing
gate and retain numerical values; do not change a tolerance after observing a
result. These experiments do not establish spatial convergence or a cost
benefit. The separate spatial and accuracy-adjusted cost milestones remain
independent.
