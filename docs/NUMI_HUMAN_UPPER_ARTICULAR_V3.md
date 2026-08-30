# Numi Human upper-limb articular validation v3

The old upper-limb showcase was rejected after direct review showed separated
shoulder, elbow and wrist geometry.  Its registration artifact predated the
humeral-head articular-center gate.  The replacement payload rebuilds the
complete 185 BodyParts3D bone surfaces against the pinned mechanics meshes and
keeps the humeral heads constrained to the source joint axes.

Default right scapula-to-humerus distance fell from 7.600 mm to 0.421 mm;
left fell from 7.908 mm to 0.336 mm.  Right/left humerus-to-radius distance
fell from 5.400 mm to 0.549/0.342 mm.  This is a registration correction, not
a camera or postprocessing fix.

## M4 Pro review

The earlier whole-bone framing remains useful context but can obscure the
interface behind torso or pelvis geometry. The replacement views target the
exact posed mechanics anchor and render only adjacent source bones; see the
[joint-focus implementation and receipts](NUMI_HUMAN_UPPER_JOINT_FOCUS_V1.md).

| Inspection | Joint-focused native frame |
| --- | --- |
| shoulder elevation, q36 = 1.2 rad | ![shoulder](media/numi-human-upper-joint-focus-v1/shoulder/myosim-fullbody-articulated-bodyparts-bones-posed-focus-joint-child-body-41-rear.png) |
| elbow flexion, q39 = 1.4 rad | ![elbow](media/numi-human-upper-joint-focus-v1/elbow/myosim-fullbody-articulated-bodyparts-bones-posed-focus-joint-child-body-42-oblique.png) |
| wrist deviation/flexion, q41 = 0.25 and q42 = 0.6 rad | ![wrist](media/numi-human-upper-joint-focus-v1/wrist/myosim-fullbody-articulated-bodyparts-bones-posed-focus-joint-child-body-45-oblique.png) |
| neutral hand / functional fist | ![hand](media/numi-human-upper-joint-focus-v1/hand-neutral/myosim-fullbody-articulated-bodyparts-bones-focus-joint-child-body-62-side.png) |

All five inspections use exact NHEQ1 dependent-coordinate projection and the
native Apple M4 Pro renderer. Four fixed cameras were retained for each pose.
Acceptance combines the full angle set with 312 minimum-gap and bidirectional
interface-patch evaluations plus 156 bilateral parity evaluations; it does not
depend on a single attractive frame.

## Evidence boundary

This validates bone placement and kinematic continuity.  The persistent
ulna-to-triquetrum space is anatomically expected until TFCC, cartilage and
ligament geometry are present; forcing direct bone contact would be wrong.
Loaded joint stability, compliant cartilage, TFCC mechanics, ligament
restraint and deformable tendon/fascia remain separate qualification gates.
