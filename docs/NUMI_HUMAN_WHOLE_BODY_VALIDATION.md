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

At revision `385e77e`, the manifest covers 15 musculoskeletal regions and six
continuum layers. The baseline report contains 82 verified requirements, 18
contradicted requirements, two insufficient requirements, and 76 missing
requirements. No complete scope is yet promoted.

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
remain incomplete because compliant articular contact and sustained loaded
motion are not yet qualified. The right knee is a mirrored left specimen, not
an independently segmented right subject.

## Development order

The gate makes the next useful work explicit:

1. Add knee cartilage/meniscus contact and a sustained flexion/loading
   qualification without regressing the extensor-chain ownership.
2. Apply the same exact force-owner pattern to the Achilles/ankle/foot chain,
   then qualify foot contact and the existing compound-toe DoF policy.
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
