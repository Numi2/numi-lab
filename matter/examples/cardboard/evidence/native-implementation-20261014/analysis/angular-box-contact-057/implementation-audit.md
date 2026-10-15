# Angular BOX Armijo CCD implementation audit

Date: 2026-10-15  
Checkout: `/Users/home/numi-cardboard-20261006`  
Observed HEAD: `62371ed242d94ce47fbad16e77f7b44c9ff050e2` (worktree was dirty; this audit is bound to the file hashes below, not to that commit alone).

This is a read-only source audit. No repository source was edited and no GPU work was run.

## Finding

The current native path already has a real six-coordinate free-body Newton candidate and evaluates a rotated BOX at the exact trial endpoint. The immediate angular trial blocker is the Armijo feasibility kernel: it compares base/trial quaternion components exactly and returns `NM_STATUS_INVALID_DISPATCH` on any orientation change. It then uses an exact fixed-orientation point-segment/OBB distance helper. Replacing the rejection with endpoint-only distance would admit rotation-through-board trials whose endpoints are both clear.

The native contact geometry is not missing rotation support. `boxContact` transforms the FEM point through the candidate box quaternion, computes the signed point-to-OBB distance and witness, and returns the world normal. The free-body projector adds the six candidate increments to the live body velocities, advances position with the bridge time step, integrates orientation using a world-frame left quaternion increment, rotates inverse inertia into the candidate frame, then projects every proxy from that body state. This is the geometry the CCD path must reproduce.

## Exact path to preserve

Runtime orders the trial as follows:

1. It copies the current projected rigid states into private Armijo storage.
2. `nm_fem_armijo_prepare_free_rigid_candidate` forms `baseCandidate + alpha * NewtonCorrection`, using the same `femArmijoMetrics.z` alpha as the FEM candidate.
3. `nm_project_primal_free_rigid_candidate` recomputes private rigid geometry from unchanged `currentBodies` and that candidate.
4. `nm_contact_validate_fem_rigid_armijo_path` runs over the full canonical pair table before sparse candidate compaction or contact evaluation.
5. Contact is evaluated at `femArmijoCandidate` and `femArmijoRigidStates`; private residual construction gathers FEM rows and the free body's 3 linear + 3 angular rows. A failed trial status prevents Armijo acceptance.

Relevant anchors:

- `contact.metalinc:4005-4084`: body candidate, six velocity increments, endpoint pose, rotated inverse inertia and proxy projection.
- `fem.metalinc:4128-4175`: FEM trial position from the same line-search alpha.
- `runtime.mm:15637-15683`: private rigid copy/candidate/projection.
- `runtime.mm:15874-15890`: CCD dispatch, before sparse pair flags and contact evaluation.
- `runtime.mm:15808-15845`: endpoint contact evaluates the exact private trial FEM/rigid buffers.
- `fgmres.metalinc:2523-2642`: sparse active-pair-to-slot map and six-row free-body contact residual.
- `fgmres.metalinc:2695-2747`: six-row rigid inertia residual.
- `fem.metalinc:4275-4324`: a nonhealthy private-trial status prevents Armijo acceptance.

The exact free-body orientation map is not ordinary endpoint quaternion interpolation. At a trial candidate, the projector computes

`omega(alpha) = omega_source + candidate_angular_velocity(alpha)`  
`q(alpha) = Exp(dt * omega(alpha)) * q_source`

with the world-frame increment on the left (`contact.metalinc:3981-3998, 4048-4062`). The generalized correction is scaled linearly in alpha, so the CCD fraction `s` between the current Newton endpoint and this specific tested trial must use

`omega(s) = omega_base + s * (omega_trial - omega_base)`

and recompute `q(s) = integrateWorldQuaternion(q_source, omega(s), dt)`. Do not use `slerp(q_base,q_trial,s)`, `Exp(s * dt * deltaOmega) * q_base`, or only the tested endpoint: those do not generally equal the projector's candidate family, especially when base and correction angular velocities are non-collinear.

At each `s`, construct the same BOX proxy transform as `projectProxy`: body orientation times proxy local orientation, and body position plus the rotated proxy-local center. The FEM point follows the affine current-Newton-candidate to trial-candidate segment. The source body pose must come from the unchanged `currentBodies` buffer for this transaction; use `bridge.time.y` exactly. This also avoids reconstructing the source quaternion by inverse-rotating a previously projected state, which would add avoidable FP32 cancellation.

## Minimal safe extension

Extend `nm_contact_validate_fem_rigid_armijo_path` to receive the unchanged source body buffer and bridge timestep for body-backed dynamic BOX proxies. Keep the current translation-only fixed-orientation path for prescribed BOX/capsule proxies. For the admitted non-articulated, one-body, six-DOF dynamic BOX, evaluate an exact candidate pose at any path fraction from the source body and interpolated base/trial linear and angular velocities. Keep proxy half-extents fixed; reject nonfinite/degenerate source or candidate state.

Use the existing `boxContact` signed distance as the distance oracle. For a point outside the box it measures Euclidean distance to the clamped OBB; inside it returns the negative distance to the nearest face. In box-local coordinates this is the signed distance to a fixed axis-aligned box and is 1-Lipschitz.

A conservative-advancement bound for the alpha-path is:

`L = |(x1 - x0) - (C1 - C0)| + dt * |omega1 - omega0| * max(|x0 - C0|, |x1 - C1|)`

where `x0,x1` are FEM endpoint positions and `C0,C1` are body-center positions. The bound follows from the relative local point velocity: translation contributes the first term; the derivative of `Exp(dt*omega(s))` contributes at most `dt*|deltaOmega|` times the maximum COM lever. The SO(3) exponential Jacobian has operator norm at most one; the proxy-local orientation is a fixed rotation and does not increase the bound. The norm of the affine relative position is bounded by its endpoint maximum.

For each pair:

- Retain the existing strict floor `max(contactAcceptance.x * numericalLimits.x, 16 * FP32_epsilon * coordinateScale)`; do not relax it to the broad contact slop.
- First use a cheap orientation-independent broadphase: the minimum norm of the affine segment `(x(s)-C(s))` versus the enclosing sphere radius `|proxy.localCenter| + |boxHalfExtent|`. If that segment stays outside radius + floor + roundoff, the pair is proven clear for every orientation.
- Otherwise evaluate the exact projected BOX SDF at `s=0`, then advance by a strict fraction (for example `0.8 * (gap-floor)/L`). A 1-Lipschitz SDF and this step bound prove the gap remains above the floor between samples. Evaluate the actual trial endpoint explicitly.
- If a gap reaches the floor, any input or bound is nonfinite, advancement stalls, or a fixed evaluation cap is exhausted before reaching `s=1`, mark the private trial failed. Do not shorten the tested alpha internally, alter the candidate, or change the physical solver tolerance; let the existing line search test its next alpha from the same Newton base.

For exact endpoint identity, use the already materialized base and tested-trial rigid states at `s=0` and `s=1`; recompute intermediate poses from `currentBodies`. Include scale-aware FP32 distance roundoff as the existing translation helper does. A radial broadphase is only a safe early-clear test; it must never admit a pair whose bound is nonfinite or ambiguous.

This adds a feasibility gate only. It should write only the per-trial status, preserve accepted FEM/material/contact/cohesive histories, and use existing status-to-Armijo rejection. It must execute for all canonical FEM/rigid pairs even when the pair was inactive or absent from the sparse sample list.

## What the source already supports and what it does not

The contact sample uses the candidate witness and normal, forms rigid angular impulse as `cross(witness - bodyCenter, impulseOnRigid)`, and `rigidDirectionalResponse` uses the candidate inverse inertia. The six-row free-body residual gathers the candidate sample's three angular impulse components. Thus translation and rotation are present in the generalized residual.

This does **not** establish a correct rotating-box Newton operator. `contact.metalinc:5527-5532` explicitly says the full coupled geometric derivatives for movable/rotating rigid bodies are not implemented/tested. `fgmres.metalinc:2720-2830` maps spatial contact Hessian action through point motion and moment arms, but it does not add the derivative of the changing moment arm/normal-witness geometry. Angular CCD can prevent a trial from crossing the feasible boundary; it cannot by itself make the angular contact tangent complete or establish convergence/force accuracy. A native finite-difference check of the six-row contact residual versus the production operator is still required.

Also distinguish this Armijo alpha-path certificate from physical-time CCD. The current kernel checks the path between the current Newton candidate and the tested Armijo candidate. It does not by itself certify the final rigid body's physical trajectory over the full accepted simulation timestep. If “no tunneling over dt” is required, add a second sweep from the accepted source pose to the accepted candidate endpoint using `q(t)=Exp(t*omega_trial)q_source` and the integrator's exact FEM trajectory. Do not claim the Armijo-path proof covers that trajectory unless the two paths are shown to coincide for the tested update.

Current scope remains one non-articulated six-DOF free rigid body with BOX geometry in the sparse full-contact path; the admission predicate excludes articulated or multi-free-body layouts and other rigid shapes. The current validator also fails closed for all rotations, including a representation-only quaternion sign change, because it compares quaternion components directly (`contact.metalinc:4765-4770`).

## Failure-closed native regression set

Add cases through the production validator and Armijo trial path, not a parallel collision routine:

1. Zero angular change and pure translation: same pass/fail outcomes as the current fixed-orientation distance check, including floor-boundary cases.
2. Rotating rectangular BOX whose start and endpoint are clear but whose mid-path sweeps through a FEM point: must reject the exact tested alpha.
3. A rotating near-miss with a certified positive minimum above floor: must pass; include nonidentity proxy local orientation and nonzero local center.
4. Combined translation, rotation, and FEM node motion; include a non-collinear `omega_base` and `deltaOmega` so slerp/incremental-from-base implementations demonstrably disagree with the exact projector path.
5. An inactive-at-base pair that crosses during the trial, plus an endpoint-clear high-angle sweep. The path validator must run before sparse compaction and still reject; if its bounded advancement cannot certify a multi-turn path, failure is correct.
6. Nonzero linear and angular velocity with anisotropic positive-definite finite inertia. Compare projected trial orientation/inverse inertia and candidate contact torque to an independent double-precision oracle.
7. Backtracking: alpha 1 crosses, a smaller alpha is clear. Verify the first trial is rejected, the second is independently rebuilt from the unchanged base/source, and rejected private states/histories are not committed.
8. Invalid quaternion, nonfinite velocity, nonpositive timestep, invalid extents/inertia, and exhausted advancement budget all fail without candidate publication.
9. Same physical quaternion represented by opposite signs yields the same geometric verdict; raw quaternion bit comparison must not remain the rotation gate.
10. Compare dense and sparse private-contact layouts for identical CCD status and endpoint contact residual when sample capacity admits the same contacts. Capacity overflow remains a rejection.

Use an independent CPU double-precision oracle to sample/solve the minimum point-to-moving-OBB gap for the manufactured cases. It should verify the conservative bound, not train a threshold or replace native evidence. The native acceptance certificate remains unchanged.

## Source binding

Observed source file SHA-256 values:

- `matter/src/metal/contact.metalinc`: `cdb425493f265cbdae00a931dd7654c3efba5d8f4f8ae39687f2b034173185ae`
- `matter/src/metal/fgmres.metalinc`: `173bb36a5b160938bafaf7d3664c803eadd6b2f3820772db97dcb9faff79bff8`
- `matter/src/runtime.mm`: `509d6f79af288cafc9ee3fd306c8c74921853a9d054eec5cff2a5b4e9267ff25`
- `matter/include/numi/matter/sparse_private_contact_admission.hpp`: `859c763ca42e8b6867f0ec77bcb5661ae143cd79ad2df87dde083b51ba138371`
