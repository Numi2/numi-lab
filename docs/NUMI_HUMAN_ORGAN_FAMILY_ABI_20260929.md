# Complete declared source-family visual payload, 29 September 2026

The Human-owned compiler adds 77 exact BodyParts3D members missing from its
existing 18-region organ graph: 18 material organ components, 40 vessel
segments, 15 ducts and four cardiac cavity references. The composite retains
the original 310 surfaces byte for byte and packages 387 total surfaces.
All 378 unique members of those declared source families are represented.
Shared and aggregate/descendant source representations are retained; this
does not establish disjoint tissue, connected lumens or whole-body anatomy.

NHANAT1 ABI 4 preserves header/record/vertex/index layout and adds layers 9-10.
ABI 1, 2 and 3 retain their existing layer/count bounds. Maximum ABI-4 capacity
is 1,024 surfaces, 1,000,000 vertices and 6,000,000 indices.

| Layer | Source meaning | Semantic | Material |
| --- | --- | --- | --- |
| 9 | Immaterial cardiac cavity reference | 51025 | 21 |
| 10 | Duct / biliary-tree segment | 51026 | 22 |

The mask range extends to 1-1023. Default masks remain 255 for ABI 1-3 and
become 1023 for ABI 4. Zero, overflow, repeated mask flags and masks without an
anatomy payload fail before capture. Useful masks are 63 for existing organs
and branches, 256 for cavities, 512 for ducts and 1023 for the combined view.
All surfaces remain in the packet. Selected instance flags are 11, hidden
flags are zero, and the native pose snapshot records total count and mask.

New per-camera fields `cavity_reference_pixels` and `duct_surface_pixels`
report layer pixels. Opaque selected pleura/lobes can cover internal geometry;
the pixel gate follows the outermost selected layer. In profiles without an
outer layer, each selected populated layer must appear across the camera
family. Packaged surface counts do not mean individual pixel visibility.

The Human auditor independently checks source types, complete family members,
every source triangle, position, normal, source/core binding and actual native
COM frame. Twelve audits pass: raw rest, projected neutral and coupled torso
flexion/rotation, each with four masks. Maximum added position error is
0.120 micrometres under the unchanged 20-micrometre gate. Native COM/orientation
witness tolerance remains 1 micrometre. The existing physics library and
shaders are unchanged; shared-host timings are not a performance result.

Three added meshes preserve duplicate-face/vertex-link defects. The source
right atrium/right ventricle cavity overlap remains. No source repair,
volume/mass assignment, connected-lumen admission, deformation or mechanics
is introduced. Source-derived images retain the existing BodyParts3D and
Z-Anatomy attribution/license boundary.

Executed native probe SHA-256 is
`68d85d07385fd5df487bccc5f0d8ace5eaba61e1fe1e601e1a8cf9cfa767b07e`.
Binary, executed source, compiler/native/test commands, packets, poses and
audits are retained under
`/Users/home/numilab-human/Build/organ-family-coverage-20260929`.
