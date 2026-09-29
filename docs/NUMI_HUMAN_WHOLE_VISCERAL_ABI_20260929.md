# Brain, ocular and visceral source references - 29 September 2026

The Human-owned configuration v2 adds 192 pinned BodyParts3D source members
to the unchanged 387-surface prefix. The new NHANAT1 ABI-5 composite contains
579 surfaces, 738,379 vertices and 3,869,040 indices. All 571 unique members
of its 46 declared source families are represented. This is source-family
membership and single-link reference geometry, not whole-Human qualification.

The native reader admits ABI 1-5 without changing header/record/vertex/index
layout. Each earlier ABI retains its layer and surface bounds; ABI-5 layers
are 1-15. Maximum capacity remains 1024 surfaces, one million vertices and
six million indices. The ABI-5 mask/default is 32767; defaults remain 255 for
ABI 1-3 and 1023 for ABI 4. Earlier ABI mask overflow fails before capture.

| New layer | Source meaning | Semantic | Material |
| --- | --- | --- | --- |
| 11 | Neural-region reference | 51027 | 23 |
| 12 | Brain ventricular-region reference | 51028 | 24 |
| 13 | Anatomical-junction reference | 51029 | 25 |
| 14 | Ocular-region reference | 51030 | 26 |
| 15 | Ocular muscle reference | 51031 | 27 |

Five source ventricular-system regions retain their material/cardinal source
type rather than being relabeled as fluid cavities. Fourteen ocular muscles
remain separate from ocular regions, ducts and immaterial lacrimal spaces.
The ileocecal junction is a reference, not connected-lumen admission. Colors
identify source classes without encoding physiological state.

New per-camera fields are `neural_region_pixels`, `ventricular_region_pixels`,
`junction_reference_pixels`, `ocular_region_pixels` and
`ocular_muscle_reference_pixels`. Selected populated layers must appear across
the camera family unless opaque pleura/lobes are selected. Every anatomy mesh
is retained, including hidden instances. Seven cranial bone meshes are omitted
from the explicit head inspection selection; the bone payload is unchanged.

The independent Human auditor checks source faces through a separate OBJ
parser, normals through NumPy and actual MuJoCo body COM/orientation. All
579 surfaces pass three source poses; 11 masks/focus profiles retain 44 PNGs.
Maximum added position error is 0.1832 micrometres under the unchanged
20-micrometre gate. Head, neck and pelvis bindings are source-named; independent
cervical and eye joints do not exist in the source. Eight skull transforms
match the shared source frame algebraically, without clinical containment proof.

Fifteen new meshes preserve topology defects. Source/self/interdomain volume,
connected lumens, tissue mechanics, subject anatomy and clinical registration
remain open. Physics library and shaders are unchanged. The executed probe
SHA-256 is `555fecc6e23b8dfbd1b72a8638551005adaa0f089362ffab7de7b1b8cb580854`.
Native commands, packets, poses, independent audits and tests remain under
`/Users/home/numilab-human/Build/whole-visceral-coverage-20260929`.

The 43 final regression checks pass without skips, including new-reader
ABI 1-4 geometry parity and unknown ABI/layer/mask refusal before capture.
