# Native hand and wrist actuation, 2026-10-23

This increment uses the existing Metal articulation, MyoSim muscle mechanics, compliant tendon state and accepted-state/replay owners. It adds accepted/baseline path and fiber kinematics to the selected-muscle telemetry so a zero-force channel can be checked against its actual constitutive law. It adds no force floor, joint lock, root assistance or alternate simulator.

## Executed checks

Build 004 source SHA-256 is `78cb1bacca1048543dca150661ab34fd18e1f77fdacc8c335a84f3af2b3fa791`; binary is `f45b8b3124d0109e828c937f619a020778ea54a794dc5916190f0cbb7c5d8f82`. The exact command, shadow header, dependency/library hashes and compiler are in `scoped-native-build-manifest-004.json`. This is a scoped source build linked to the retained existing library, not a clean-main whole-scene build claim.

All these matched baseline/activation transactions ran on SSH Numi, one native job at a time, with bitwise accepted-state replay:

| Cohort | Accepted steps / simulated duration | Native wall seconds | Target q response L1 (rad) |
|---|---:|---:|---:|
| Finger flexion | 800 / 0.1 s | 67.118 | 6.67536 |
| Finger extension | 800 / 0.1 s | 76.794 | 0.768388 |
| Wrist flexion | 80 / 0.01 s | 35.066 | 0.218865 |
| Wrist extension | 80 / 0.01 s | 33.349 | 0.183561 |

Wall times include startup/loading/serialization; they are not integrated throughput. The one-time reference configuration seeds q36/q74 to 0.1 rad; all 44 hand/wrist coordinates remain free subject to authored limits. Raw results and exact invocations are retained per run.

The old strict flexion result remains FAIL because it required positive endpoint force in every selected channel. Actual channels 243/306 are slack: normalized tendon lengths 0.998323449/0.998250578, with exactly zero tension and finite residuals. The derived qualifier accepts this only when the measured path/fiber state, source model and Float32 tendon law agree. It retains raw statuses, checks activation, force/tension products, residual <=1e-4, joint bounds, replay and aggregate motion/torque. The residual qualification bound is an explicit conservative test bound from the owner's stationary-path branch, not a claim that the dynamic owner already mandates it. Maximum measured residual across the four trials is 6.71e-8. Seven tests cover valid slack and contradictory zero/incorrect force, force on slack, NaN, activation mismatch and timestep mismatch; all behave as expected. See `owner-law-derived-checks-001/test-report.json`.

Build 003 and its earlier one-step/10 ms results remain retained. Build 004's full 44-row accepted and baseline q/v results match 003's 800-step flexion exactly. An accidental intermediate link-output collision was restored before testing; `build-004-binary-output-collision-restoration.json` records the incident and original hash verification.

## Geometry and physical limits

The source hand repair 068 reduces global severe folds without worsening any formerly <=90-degree edge past 90 degrees. It passes exact Float32 self/surrounding-skin/bone triangle checks for both sides at eight retained poses and two actual 10 ms reference flexion/extension endpoints. Its topology preserves the baseline 115 head boundary edges; it is not a globally closed-skin claim. Most remaining sharp hand edges are inherited interdigital web margins; two new palm folds and distal-thumb deviations remain under repair. Surface source registration is distinct from physical skin/fat admission.

The 100 ms flexion endpoint has 160 left-hand/surrounding-skin crossing pairs. Retained witnesses identify index/middle fingertips against thigh skin. The tested articulated core has support contact, but does not run the complete deformable self-contact owner. Those crossings are a real integration failure; they are not fixed by reducing hand size or locking wrists. The distinct earlier default-initialized 10 ms failure (106 pairs) has been restored under its own report name after an audit filename collision; original poses and classification were preserved throughout.

Regenerated skin/fat cells, conservative mechanical transfer and whole-scene contact remain incomplete. Invalid layer candidates and the stale inferred capitate attachment remain rejected. The reduced-surface preparation now removes 632 orphan rows (8,722 allocated to 8,090 referenced); both stale capitate rows are orphans. Ordered compaction preserves all face/tet position tuples, signed-prism masks, Float32 position determinants and volumes exactly. It does not repair or hide invalid cells; see the compaction receipt. The 068 source must not be paired with old source-hash-mismatched FEM headers. No grasping, sustained whole-scene dynamics, completed anatomy, biological or clinical validation is claimed.

## Earlier retained repairs and a rejected trial

The old radial/Taubin render-only wrist contour stays disabled by default. The left APL force-map envelope uses the existing compiler and unchanged anatomical endpoints; its <=2.2 amplification bound and exact identities are retained under `results/left-apl-envelope-under-2p2`.

A proposed 005 initialization-handoff change was tested and rejected before physics: the existing guard intentionally forbids arbitrary `--pose-q` stepping outside admitted construction owners. The proposed source changes were reverted, with admission unchanged. `initial-pose-005-rejected-trial.json` records the corrected diagnosis; that binary/source is not promoted.

## Reproduce

Use `run_hand_actuation_cohort.py` with explicit `--binary`, `--binary-sha256`, `--source-manifest`, cohort, `--initial-reference-posture`, exact skin identity, step count and dt. Derive the endpoint qualification using `qualify_hand_muscle_owner_law.py` on the retained result and invocation. Asset paths in receipts refer to SSH Numi. The existing artifact schemas and runtime ownership are retained.
