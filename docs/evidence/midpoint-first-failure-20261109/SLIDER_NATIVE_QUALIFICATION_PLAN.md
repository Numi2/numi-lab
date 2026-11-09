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

For a subsequent native rollout, also compare each owners cumulative

For a subsequent native rollout, also compare each owner's cumulative
interface-work history and their sum with the separately time-qualified FP64
work histories. At the 2 µs reference clock, linearly interpolate native
step-integrated work. For FEM, slider and relative work separately, require the
largest absolute difference to be at most 1% of the greater of the reference
history's maximum absolute cumulative work and 1e-12 J. This adopts the
existing 1% cumulative-history accuracy scale and a fixed absolute floor; it
is added before a new rollout and does not replace the local work, impulse,
force, torque or replay checks. Report final absolute errors as well.


Per-node qualification also checks the measured normal impulse directly
against the independently reconstructed law impulse with the existing
impulse bound, even when the node velocity is zero. The sum of retained nodal
impulses must reproduce the reported FEM impulse within a double-precision
roundoff bound. This closes a zero-work blind spot without changing the
physical impulse tolerance.


For the initially active fixture, qualify the final slider position and
velocity of its independent 0.5 and 0.25 µs references using the same 0.25%
state-accuracy gates as the primary fixture. Use fixed normalization floors of
1e-8 m and 1e-6 m/s, respectively, for near-zero reference values. If that
temporal check passes, require the native near-contact trajectory to meet the
same final position and velocity gates. This adds state qualification before a
new full native rollout and does not change existing force or work gates.


For both initial-gap fixtures, also time-qualify the reference total contact
torque impulse about the common fixed origin in each transverse direction.
Use a 1% tolerance normalized by the greater of its absolute fine-reference
value and 1e-12 N m s; the resulting 1e-14 N m s floor is the existing local
absolute torque bound. Once the temporal reference passes, apply the same
criterion to the accumulated native FEM torque impulse. Keep the independent
per-step equal-and-opposite contact and support torque checks.

Before any new native rollout, the instrument must record all three
components of FEM contact impulse, body-wrench impulse and contact/support
moment. For this frictionless world-z plane, the x/y contact impulse components
on each side and their balance must remain within the existing per-step
impulse bound. The z contact and support moments must each be at most the
existing 1e-14 N m s absolute torque floor; balancing nonzero axial moments do
not satisfy this geometric restriction. Compute FEM work from the full impulse
and midpoint velocity vectors. The existing per-node normal-law work and
impulse checks still apply. Earlier attempts did not capture all of these
components and remain incomplete historical evidence, not all-axis passes.

For a new native trace, record the actual compiled world-z gravity alongside
the body mass and initial velocity. Independently recompute the body contact
impulse at every accepted step as `mass * (v1 - v0) - h * mass * gravity` and
require agreement with the reported momentum impulse to double-output
roundoff. This must use the full physical high-plus-low velocity when a
compensated owner exists. It is an input-integrity cross-check in addition to,
not a replacement for, the local residual, owner-wrench balance, and force
reference gates. The old high-only partial traces were checked using their
compiled FP32 gravity; they do not establish a complete coupled rollout.
