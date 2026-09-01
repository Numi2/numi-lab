# NumiLab Human whole-body validation gate

`tools/validate_numilab_human.py` is the fail-closed acceptance ledger for the
full Human, not a visual showcase. It evaluates the machine-readable manifest
in [`numi-human-whole-body-validation.v1.json`](numi-human-whole-body-validation.v1.json)
and refuses to call the Human qualified while any required regional or
continuum-layer claim is missing, contradicted, or supported only by weaker
evidence.

Run the strict gate from the repository root:

```sh
tools/validate_numilab_human.py --output /tmp/numilab-human-report.json
```

Exit code `0` means every required scope reached its declared evidence level.
Exit code `1` means the report is valid but the Human is incomplete. Exit code
`2` means the manifest or validator input is invalid. During development,
`--allow-incomplete` emits the same report but returns success so tooling can
inspect known gaps without weakening the strict default.

## Evidence levels

- `structural`: source identity, coverage, topology, registration, or
  multi-pose continuity. This is not loaded mechanics.
- `preflight`: a bounded prescribed displacement or isolated mechanics test.
  This is not a live production owner.
- `one_step`: an accepted live transaction with force/reaction checks,
  rollback, and replay.
- `sustained`: a loaded trajectory with bounded strain, contact, force,
  energy, and failure behavior over time.
- `validated`: comparison against independent material, anatomical, or
  experimental evidence, including held-out response.

The minimum level is declared independently for every requirement. A stronger
receipt may satisfy a weaker requirement; a mesh, render, preflight, or
one-step receipt cannot satisfy a stronger one.

## Current evidence baseline

The manifest covers 15 musculoskeletal regions and six continuum layers. The
current report contains 72 verified requirements, 44 contradicted
requirements, two insufficient requirements, and 64 missing requirements.
No complete scope is yet promoted.

The all-DoF support audit is documented in
[`NUMI_HUMAN_WHOLE_BODY_ALL_DOF_V1.md`](NUMI_HUMAN_WHOLE_BODY_ALL_DOF_V1.md).
It preserves all 128 coordinate residuals and source-muscle contributions;
the current ledger remains 0/21 complete because this diagnostic does not
promote unresolved mechanics.

The fresh integrated Release build also passed the native NumanX label 12/12
on Apple M4 Pro. Those tests qualify transaction ownership, capacity,
rollback/replay, and attachment fixtures; their one-tet fixture does not count
as regional anatomical or material evidence.

The contradictions are intentional development signals:

- NHTENDON covers all 832 endpoints for 416 muscles, but 194 endpoints remain
  point fallbacks and distributed surface coverage is `0.7668269`, not `1.0`.
- The BodyParts3D registration receipt explicitly says that skin, muscles,
  tendons, organs, vessels, nerves, and other soft-tissue layers are not yet
  represented by that payload.
- Costal cartilage has a 14-region prescribed-displacement FEM preflight with
  replay and rollback, but `production_owner_fraction=0`; it is insufficient
  for live deformable-mechanics promotion.

The bilateral knees have one-step source-muscle actuation, exact QAT/PTL
attachment transfer, passive ACL/PCL/MCL/LCL/QAT/PTL FEM reactions, force
closure, positive deformation Jacobians, bitwise replay, and rollback. They
also have bilateral one-step exact-surface cartilage/meniscus contact in the
live Human force transaction: seven source pairs, all 69,701 samples, explicit
ownership of 57,930 cross-body mechanics samples and 11,771 retained
same-body samples, nonzero pressure/area/force, force and world-origin moment
closure, bitwise replay, and rollback. The 65-step prescribed-closure CPU ramp
still supplies constitutive preflight and elastic-energy identity. ABI 10 now
searches each current paired triangle plus its three manifold edge-neighbours,
recomputes barycentric scatter and the oriented current normal, and fails
closed on non-manifold source surfaces or malformed neighbour records. Both
exact payloads give all 69,701 samples a one-ring candidate; a cross-edge M4
Pro fixture passes. The left live Human additionally passed a bounded two-step
`10 microrad` trajectory with 25,310 to 30,989 closed samples, positive
deformation Jacobians, balanced nonzero contact, bitwise replay, and rollback.
A 1 mrad request failed closed at the projected-anchor compatibility gate
before solve and is diagnostic only. The knees remain incomplete because the
query is not multi-ring/global, same-rigid-body meniscus interfaces lack
relative deformable ownership, the law is an explicit penalty rather than
unilateral nonpenetration, and sustained physiological loaded flexion is not
qualified. The right knee is a mirrored left specimen, not an independently
segmented right subject.

The shared Human/Matter adapter additionally has a one-step Metal transaction
fixture where internal FEM contact and tendon traction share one force arena,
reaction scatter, rejection decision, and rollback/replay boundary. This is
infrastructure evidence, not regional anatomy evidence, so it does not raise
either knee above `preflight` contact status.

The adapter's reduced exact-surface fixture and both anatomical payload cooks
now pass. The fixture separately proves body-Jacobian scatter, FEM isolation,
typed same-body retention, malformed-flag rejection, force/moment audit,
replay, and rollback.

The bilateral Achilles chain now has a dedicated one-step receipt rather than
depending on the contradicted whole-body `100%` enthesis claim. Six nonlinear
gastrocnemius/soleus rows terminate in six distributed named-calcaneus
envelopes, produce nonzero bilateral ankle torque increments, close force and
moment, borrow the owning command buffer, roll back on rejection, and replay
bitwise. This promotes active tendon-to-bone transfer for both ankle/hindfoot
scopes only. Passive ankle tissues, articular/contact mechanics, sustained
loading, and OpenSim Rajagopal equivalence remain missing.

The bilateral feet now also have a dedicated reduced plantar-fascia windlass
preflight. Ten exact BodyParts3D surface routes connect named calcaneal,
metatarsal-head, and proximal-phalanx patches while retaining one authored MTP
coordinate per foot. All ten tension-only rays engage, close force and moment,
change the one-step articulated response, and replay bitwise. This is a tendon
force-transfer law, not deformable FEM or a live NumanX continuum transaction;
therefore ankle/foot passive tissue remains insufficient rather than promoted.

The ten authored foot support witnesses now also close a deterministic static
floating-base wrench. On Apple M4 Pro they carry 952.864475 N against
952.864477 N of body weight with a `2.20e-6` maximum root force/moment
residual, all reactions nonnegative, and bitwise replay. This is recorded only
as a preflight: the internal muscle residual remains `22.9850`, root
acceleration residual remains `5.7539`, and `internal_balanced=false`.

## Development order

The gate makes the next useful work explicit:

1. Replace projected rest anchors with a moving-enthesis/initial-continuum map,
   then add a broader current-surface or implicit nonpenetration owner and
   qualify sustained physiological knee flexion/loading without regressing
   extensor-chain ownership.
2. Close the internal muscle/passive-force equilibrium exposed by the new
   support-wrench gate, then promote the reduced plantar windlass law through
   the owning Human/NumanX transaction and qualify loaded foot contact while
   retaining the existing compound-toe DoF policy.
3. Qualify hip muscle entheses, passive capsule/ligaments, and articular
   contact bilaterally.
4. Complete shoulder, elbow/forearm, wrist, hand, and finger force-transfer
   chains from the upper-extremity source model, with region-specific passive
   structures and contact where applicable.
5. Promote axial/costal/fascia mechanics into the live Human transaction.
6. Add skin, organ, vessel, and nerve geometry and coupled deformable owners
   with calibrated materials and interaction gates.
7. Run whole-body sustained loaded trajectories and independent anatomical,
   material, and physiological comparisons before a realistic-human claim.

Visual inspection remains useful for catching topology, side, and presentation
defects, but it is secondary evidence and is not a whole-body promotion gate.
