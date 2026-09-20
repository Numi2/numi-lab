# Numi Human passive FEM attachments

`NumiHumanTendonFEMLoadAdapter` now admits a passive attachment-only mode for
cartilage, ligaments, fascia, and other continua attached to two or more live
articulated bodies. The mode is selected by an empty endpoint-replacement map,
zero `productionForceOwnerFraction`, and entirely inactive node-load records.
At least one active bone anchor is still required.

The adapter builds every fixed-node target from the owning body pose before the
Matter solve, then projects each accepted fixed-node reaction through that
body's point Jacobian into the borrowed Human generalized-force buffer. Target
assembly, nonlinear FEM, reaction scatter, Human acceptance, and Matter
commit/rollback remain in one borrowed Metal command buffer. There is no
synthetic tendon force and no removed or duplicated MyoSim `J^T` share in this
mode. Existing active tendon replacement semantics are unchanged.

## Apple qualification

The exact fourteen-region BodyParts3D costal-cartilage payload was used as a
full-size two-body structural witness. Its 1,582 sternal-band nodes followed a
synthetic sternum body and its 1,289 rib-band nodes followed a second synthetic
rib body. A deliberately bounded 1 micrometre relative translation was applied
over one 0.1 ms step. This is an attachment and transaction test, not a
physiological breathing amplitude or a static material calibration.

| Gate | Apple M4 Pro result |
| --- | ---: |
| source regions | 14 / 14 |
| FEM nodes / tetrahedra | 13,516 / 46,278 |
| minimum regional maximum displacement | 0.998378 micrometres |
| maximum displacement | 1.03381 micrometres |
| minimum / maximum `J` | 0.999851 / 1.00013 |
| sternal / rib fixed-node reaction L1 | 0.291129 / 8.93317 N |
| sternal / rib body generalized reaction L1 | 0.224998 / 6.02424 N |
| active tendon-replacement regression | passed |
| accepted replay | bitwise identical |
| rejected transaction | rollback verified |
| wall time | 43.02 s |

The transient reaction magnitudes are not expected to balance after one dynamic
step and are not a static force calibration. The production owner fraction is
still zero. Current imported ribs and sternum collapse to one Human torso body,
so binding both sides there would create no relative mechanics. Production
thorax ownership therefore remains gated on separate live rib mechanics or a
deformable rib-cage owner.

The preserved transcript is
[`docs/media/numi-human-passive-fem-attachments-v1/costal-cartilage-passive-m4-pro.transcript.txt`](media/numi-human-passive-fem-attachments-v1/costal-cartilage-passive-m4-pro.transcript.txt),
SHA-256 `b2b4b4d5da0c24b5309a64f8b7982a9f2fbf1aba8689026e935d7a63c2475240`.
The unchanged active replacement path is preserved separately in
[`tendon-active-regression-m4-pro.transcript.txt`](media/numi-human-passive-fem-attachments-v1/tendon-active-regression-m4-pro.transcript.txt),
SHA-256 `1005b9fda08bfd3acd79f6c67058de5a7055dace9a326bffb65874337788226e`.

## Next production consumer

The next intended consumer is specimen-specific knee cartilage, menisci, and
ligaments registered to distinct live femur, tibia, and patella bodies. Source
registration, articular separation, attachment coverage, contact ownership,
and loaded multi-angle visual gates remain mandatory before that model can be
called production anatomy.
