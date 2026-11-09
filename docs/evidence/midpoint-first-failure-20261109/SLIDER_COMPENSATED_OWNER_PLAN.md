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
EOF'