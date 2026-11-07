# Proposed midpoint fixture: supported domain and acceptance contract

Status: **CPU contract tests only. No Matter midpoint option exists.** Both
existing contact/CCD experiments remain default-off. The plan is limited to
a single immutable frictionless FEM object against one immutable static
infinite plane, fixed mesh/topology, stateless elastic material, no attachments,
other contact pairs, articulation, MPM, adaptation, mutation, external
commands, multirate stepping, prescribed motion or deformable self-contact.
Start with one environment, one substep and reference-displacement nodes.
Reject unsupported configurations before allocating or encoding execution.
No hidden fallbacks to an incompatible path are allowed.

## Equations and ownership

For physical step h, accepted position x₀, velocity v₀ and τ=h/2, use a
working midpoint velocity w and position m=x₀+τw. Solve

    M(w-v₀) = τ f(m),  J = M + τ² ∇²U(m)

with the same normal law and material force/tangent in the existing solve.
The physical endpoint is x₁=2m-x₀, v₁=2w-v₀. The runtime's public clock,
request timestep and episode counters retain h. Only the restricted
mechanical/contact solve uses τ. Do not globally replace the dispatch clock:
other readers use physical h independently of the FEM solve.

The half-step residual is an impulse over τ; the physical momentum residual
is twice that value. Its physical absolute/relative acceptance thresholds
must be evaluated in physical units. Check the midpoint force and tangent
against an independent FP64 residual and directional derivative, including
near-contact and unloaded cases. Fingerprint the integrator, law and all
parameters; reject restoration across different fingerprints.

## Before any state is published

1. Keep accepted FEM state, contact history, clocks, statuses and diagnostics
   immutable while solving. Keep the force sample and half-step history in
   scratch. Reject any failed inner or nonlinear solve through existing status
   and complete rollback.
2. Reflect the certified working state into a separate full endpoint candidate.
   Check finite coordinates and velocity. Check the clearance floor for every
   relevant node at both endpoints; the linear position interpolation remains
   clear against a static infinite plane if both ends pass. Validate the
   reflected FEM material state and Jacobians, and reject any failure.
3. Convert each half-step contact impulse to the physical impulse h f(m).
   Preserve contact identity and generation. Make work the physical impulse
   dotted with (x₁-x₀)/h. Ensure any published history and exposed sample
   reflect the physical interval, without altering the half-step linearization
   consumed by the solver. Verify exactly which buffer lanes represent
   impulses, solver curvature, force, age and identity before mutating them.
4. Certify the physical residual against the reflected state and midpoint
   force, verify the final endpoint and status, then commit state, material
   fields, contact history and clocks together. If any late check fails, all
   of them must restore exactly, and failure must remain visible. Test
   save/restore and deterministic continuation at nonzero contact force.

The CPU transaction test exercises a positive-gap midpoint whose reflected
endpoint violates clearance, an endpoint with an inverted element, and a
valid force-producing step. It checks half-to-full residual and impulse/work
conversion and preservation of its copied authoritative state and history
on both rejections. It is an executable specification, not evidence that the
Metal runtime owns equivalent buffers or performs these checks.

## Native gates before extension

Compare the restricted native step with independently verified CPU midpoint
at common timesteps and all nodal states, residuals, impulses and work.
Check failure reporting, clearance, physical impulse balance, rollback and
exact replay after active contact. Only then perform the unchanged 1 ms
force-bin and 0.25% reference gates over the original duration, with temporal
and mesh refinement separately. Test near-contact starts, impact and release.
The 15.625 µs CPU result is promising but is not a native, mesh-qualified or
performance result. No friction, moving geometry, AL method or sutures are
included in this first fixture.
