# Scoped compensated reacting-slider transaction: prerequisites

The retained four coupled trajectories and the replayed first failure show
that a single binary32 rigid q/v is not sufficient for the preregistered
position and per-step momentum requirements. A correction must preserve the
physical information rather than adjust the tolerances. The scope remains one
fixed-root, world-z prismatic body coupled to one frictionless FEM plane.

The complete change must provide all of the following before it can be tested
as a candidate:

* MetalWorld owns high and low q/v words. Both words enter and leave the same
  transaction, are bound to the execution fingerprint and are checked for
  canonical finite values. Caller-supplied initial and returned endpoint lows
  must not be silently dropped. Every rejected step restores both words.
* The generalized midpoint solve uses the full initial velocity; its physical
  mass action, reaction wrench, and endpoint velocity account for the low
  word. The discrete position increment is evaluated with compensated
  arithmetic and published as both words. It is not sufficient to save a low
  word that is absent from the next solve.
* Matter receives compensated start, midpoint and endpoint plane geometry and
  velocity from the same rigid owner. Gap, normal force, tangent, clearance,
  and the late residual certificate must use the full values. A separate
  correction to the output alone would not correct the actual contact law.
* Contact force and moment must still travel through the actual body wrench
  and measured support reaction. The publication must not substitute a second
  host-integrated body or compute the two impulses from the same sample.
* The failure path must restore MetalWorld q/v high and low, and every Matter
  physical field, followed by a bitwise deterministic retry. Run this before
  contact and after a nonzero contact in both initial-gap configurations.
* First validate against the retained saved failing state and independent FP64
  full-step equations, then rerun both full 10 ms trajectories at both
  preregistered timesteps and apply the existing force, impulse, position,
  clearance, work and restoration gates. The 27-node result is not spatial
  convergence and diagnostic snapshots are not a performance benchmark.

The current midpoint admission already requires a world-z plane with zero local
offset, so the present fixture does not exercise addition of a nonzero proxy
offset. Keep that restriction unless the compensated plane projection also
preserves the offset and its low word through every contact evaluation.

The existing generic candidate adapter explicitly rejects compensated body
geometry it cannot represent. Do not remove that guard until the owner supplies
and consumes the low words end to end. The diagnostic admission remains off
by default during this work. The previous CCD and augmented-Lagrangian
experiments also remain disabled.

## Data-path audit for the low word

The existing precision protections for the FEM node are not sufficient for a
moving plane. In `matter/src/metal/contact.metalinc`, `projectProxy` stores the
plane offset in the single float `NMRigidStateGPU.centerAndRadius.w`. The
reference-plane gap subtracts that float from a compensated FEM point. The
normal force and tangent then use the compensated gap, so their input must
include the rigid offset low word as well. Merely correcting the published
body position after the solve would leave these force and tangent evaluations
at the old plane.

The required rigid data is needed at all three projections in
`matter/src/runtime.mm`: accepted pre-dynamics, each Newton candidate, and
post-commit reconciliation. Candidate propagation must cover every trial that
can produce an accepted line-search step, not just the final candidate. The
existing midpoint endpoint clearance routine constructs interval bounds from
an FP32 plane offset; it must instead enclose both offset words without
rounding the low word away. Contact admission and line-search feasibility
also inspect a scalar separation; their treatment of the low word and
roundoff must be justified conservatively for the restricted plane. The late
position and momentum certificate must use the same full rigid state.

The proposed midpoint arithmetic produces a kinematic target. It is not the
ABA body state: ABA must still receive the physical wrench and produce the
actual reaction, and acceptance must independently establish full-step body
momentum, FEM momentum, contact moment, clearance and interface work. The
current diagnostic admission stays closed until those paths are wired and
qualified together, including both-word rollback and retry.
