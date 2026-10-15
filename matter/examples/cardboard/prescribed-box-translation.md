# Prescribed fixed-orientation BOX translation

The opt-in prescribed rigid translation contract now covers `NM_RIGID_BOX` as
well as capsules. A BOX must be kinematic, have zero inverse mass and inertia,
keep its orientation and half-extents unchanged, and advance its center and
body center by `dt * linearVelocity`. The first interval infers its start from
the endpoint and velocity; subsequent intervals must continue from the last
accepted rigid state. Rotation, free rigid unknowns, unsupported barrier
scaling, and non-FEM/adaptive/mutable-topology worlds remain outside this
contract.

For prescribed BOX contact, the candidate initializer and post-commit
certificate check the FEM-node trajectory against the translating box using
the endpoint's local frame and a segment-to-AABB distance. The usual contact
floor and transaction certificate still decide acceptance. A failed native
step must restore both the FEM transaction and projected rigid proxy states.
`NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT` selects these endpoint checks and
rollback on the same numbered control step as the full-contact Armijo path;
`all` applies to every step. Malformed selectors do not enable the endpoint
gate.

The focused rollback fixture command is:

```sh
cardboard-full-contact-armijo-check BULK.nmatter --prescribed-box-overflow
```

It uses the synthetic full-contact fixture and deliberately undersized contact
capacity to trigger a short rejected transaction after projecting a translating
support BOX. It is a software regression check, not a cardboard or tool
validation. The older `--overflow` case also checks rigid-state rollback for a
static support BOX.

The positive geometry-transaction path is exercised with:

```sh
cardboard-full-contact-armijo-check BULK.nmatter --prescribed-box-translation
```

It first requests one small, separated BOX translation and requires an accepted
cursor and the expected `dt * velocity` center displacement with unchanged
orientation and extents. It then retries from that accepted endpoint with
independent endpoint-velocity and orientation mismatches; both must reject and
leave the accepted FEM, material, rigid, cohesive, and contact state unchanged.
This is a kinematic endpoint/CCD instrument check. It does not qualify FEM
deformation, cardboard creasing, or physical tool contact.

The companion `--prescribed-box-sweep-reject` mode moves the support BOX 2 mm
through the pad during one `0.1 ms` interval. It expects a native contact
rejection and byte-identical accepted FEM/material/rigid/cohesive/contact state.
This deliberately invalid sweep is a conservative-CCD rejection test, not a
physical loading experiment.

The shipping support floor is likewise authored as a prescribed BOX in
`cardboard_shipping_rigid_driver.hpp`. In the current probe flow, the floor is
placed under the accepted assembly while grips remain active by issuing
successive kinematic end poses, with each placement increment limited to
10 mm, then a stationary contact-gate step is attempted before clamp release.
The probe does not call `ShippingRigidWorldRunner::resetRigidState()` on this
path. So the placement is segmented endpoint motion, not a reset/teleport; it
remains an externally controlled tooling operation and has not itself been
physically qualified. The new BOX sweep/rollback path must pass its native
regression before any claim about that operation is made.
