# Barrier-free normal-contact comparison: prepared experiment

Status: the first frictionless FEM–plane stage is implemented and measured.
The [experiment report](FEM_PLANE_AL_EXPERIMENT_20261107.md) retains failed
calibrations, force/work traces and restoration checks. The physical/reference
gates failed; keep the option disabled. No speed, friction or physical
calibration result is claimed. The original prepared design follows below.

## Question and owning code

Does a projected augmented-Lagrangian normal law reduce Matter's total accepted
simulation cost at fixed physical error, compared with its present logarithmic
potential?

The owning paths at efde734 are:

- matter/src/metal/contact.metalinc::ipcBarrierResponse: current IPC normal law.
- refreshDeformableContactFriction: uses the normal response to bound friction
  and transports tangential history by stable source identity.
- nm_contact_checkpoint_deformable_contact_histories,
  nm_contact_commit_deformable_contact_histories, and
  nm_contact_rollback_deformable_contact_histories: transactional history.
- matter/src/runtime.mm: encodes those operations on the borrowed timeline.
- NMDeformableContactHistoryGPU in matter/include/numi/matter/shared.h and the
  snapshot state in matter.hpp: state identity and capacity contracts.

The same contact file also contains Human-support complementarity equations
and checkpointed impulse history. Those are useful implementation references,
but are a different contact domain and do not establish the proposed general
deformable method.

## Smallest controlled implementation

Keep the material model, geometry, integrator, force transfer, precision,
friction law, execution ownership, CCD acceptance and outer transaction fixed.
Change only the normal enforcement law, with an explicit experimental selector.

For a signed normal gap g >= 0, a nonnegative force multiplier lambda and
fixed penalty rho, an initial projected normal candidate is
max(0, lambda-rho*g). Its compatible augmented potential and derivative
must use the same gap and units. Update multipliers from a working trial
configuration while CCD limits only the collision-free configuration actually
accepted. This is a proposed design for Matter; it is not a reproduction claim
for MuJoCo's complete cloth implementation.

Maintain separate accepted/candidate/checkpoint multiplier arenas, keyed by
stable contact identity and topology generation. Include new arenas in reset,
rollback, snapshots, continuation hashes and capacity migration. Never let
rejected trial multipliers reach accepted friction history. Reassemble friction
from the candidate's normal response within the same owning nonlinear solve.
Do not add a CPU solver, independent queue, internal commit/wait, or a second
physical response authority.

The existing barrier feasibility/Armijo gate cannot simply be removed: author
the candidate's objective/residual and geometric acceptance explicitly.
Otherwise the comparison changes both the law and its acceptance semantics
without measuring either.

## Staged comparison and predefined gates

1. Start from the existing small FEM compression/drop fixture in
   apps/matter_physics_probe.mm, using a static plane and zero friction.
   Freeze and hash its exact generated package before trials.
2. Add sliding on that plane, with identical static/dynamic friction.
3. Add a moving rigid proxy with equal and opposite impulse accounting.
4. Only then add rod/tissue contact, holding and intentional puncture behavior.

For the first normal-contact stage, use matched initial state, mesh and loads,
and timesteps 0.25, 0.5, 1 and 2 ms. Refine the reference until halving its
timestep and mesh scale changes the tested observables by less than 0.25%.
If the reference does not converge, retain the result as inconclusive.

Predeclare these initial acceptance limits against the refined reference:
force and normal-impulse relative error <= 1%; displacement error <= 0.5% of
the pad thickness; minimum gap >= -1 micrometre; zero missed intersections;
exact accepted-state continuation after restore; zero unreported failed steps.
Use absolute force/impulse tolerances for near-zero references, set from
instrument resolution before observing either arm. Add measured friction/slip,
torque, damping and control gates before admitting the later stages.

The primary prediction to seal after instrument qualification is >= 10% lower
total elapsed time for the same accepted physical duration and all accuracy
gates. It is refuted by an accuracy failure or a valid cost ratio > 0.9;
invalid reference or timing evidence is inconclusive. Preregister a fixed
paired execution order, independent geometry/load units, repeat count and
stopping rule before native results are collected. The provisional 10% target
is not a measured effect.

Report accepted simulated seconds / wall second with all rejected work in the
denominator. Retain force/displacement/impulse traces, minimum gap, Newton and
Krylov work, contact counts, dispatch count, per-stage GPU cost, full elapsed
time, failures, frictional holding/slip and restore hashes. Iteration count,
unbounded penalty escalation or a visually stable trajectory cannot satisfy
the completion criterion.

The current CCD experiment is deliberately insufficient evidence for this
comparison: it measures geometric kernels, not the coupled material rollout.
