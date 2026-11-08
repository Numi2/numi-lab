# Whole-sweep monotonic box-plane certificate candidate

Implement only in this isolated research directory. Retain the existing
box-plane subdivision path as fallback. On eligible centred floating boxes and
static planes with positive projected center displacement and strictly
positive endpoint quaternion dot, test each vertex for either (a) positive
whole-interval clearance Bernstein coefficients using the initial center as a
lower bound, or (b) positive initial guarded clearance and strictly positive
whole-interval Bernstein coefficients for an interval-arithmetic derivative
lower bound. Only if all vertices pass may the certificate return clear; all
other cases use the unchanged search. Keep both existing experimental
selectors disabled; this is a collision-correctness candidate, not their
performance tuning.

For q(alpha)=q0+alpha(q1-q0), define D=q dot q and the unnormalized rotated
vertex projection R. The physical-time map has derivative kappa/D with
kappa >= dot(q0,q1) for hemisphere-aligned endpoints, including the
small-angle fallback. If centerDelta >= 0, a conservative numerator for the
clearance derivative is R'D-RD' + centerDelta*dot(q0,q1)*D; D is positive.
Construct outward-rounded interval Bernstein coefficients; require finite,
strictly positive bounds. Reuse existing geometric and integration guard.
Record this derivation and scrutinize implementation assumptions before any
promotion. An algebraic numerical screen alone is not proof of Metal
rounding or of the actual integrated trajectory.

Before full-step evaluation, test the scalar candidate against independent
FP64 sampled geometry across deterministic unused rotations, both translation
directions, near tangencies, intersections and scale/offset variations. Any
candidate-clear case with a crossing or ambiguous independent bound fails. Run
production cap 16 as decisive with the retained 12 fixtures and independent
whole-trajectory observer, including force-free near misses. Compare response
residuals and restoration, not just geometry or iteration counts; retain all
failures. Diagnostic cap 128 may be used separately. The existing
four-contact response candidate fails mixed contact and has a recorded normal
residual failure, so a collision certificate cannot promote that solver.
Do not publish this as a production fix until independent adversarial checks,
actual-path assumptions, response and rollback are adequately qualified.

## Main contact-response control

After the research-solver trial, link the identical collision certificate
candidate with the original main contact-response AIR objects, and rerun the
same 12 fixtures with both advancement selectors off at cap 16. Compare
impulse, full-step status, trace-on/off, observed restoration, normal residual
and independently bounded accepted trajectories. This distinguishes whether
any completion depends on the unpromoted four-contact response. The candidate
also requires a separate audit of all nonfinite input and intermediate
handling before production use; finite randomized inputs do not prove it.
