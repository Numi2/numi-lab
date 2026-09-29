# Source lung-lobe and pleural visual surfaces, 29 September 2026

The Human-owned compiler now appends the five source-authored Z-Anatomy lung
lobes and one source-authored pleural mesh to the unchanged 304-surface
BodyParts3D torso payload. This is reference-atlas geometry with measured
bone-centroid registration, not clinical registration, respiratory motion,
fluid domains or tissue mechanics. The native visual probe reads the payload;
it does not select anatomical sources, invent missing geometry or repair them.

NHANAT1 ABI 3 preserves the header, 32-byte records, 24-byte vertices and
32-bit indices. ABI 1 retains its 64-surface/layer-1-3 limits. ABI 2 retains
its 1,024-surface/layer-1-6 limits. ABI 3 admits 1,024 surfaces and layers 1-8.
Vertex and index capacities remain 1,000,000 and 6,000,000.

| Layer | Meaning | Semantic | Material |
| --- | --- | --- | --- |
| 7 | Lung-lobe envelope | 51023 | 19 |
| 8 | Pleural source surface | 51024 | 20 |

`--torso-anatomy-layer-mask N` selects visible layers using bit `layer-1`.
The default is 255; zero, values greater than 255, duplicate flags and a mask
without an anatomy payload are rejected. Geometry for every surface remains
in the native packet. Hidden surfaces have instance flags 0; selected ones
retain casts-shadow, receives-shadow and sensor-visible flags (1+2+8 = 11).
The pose snapshot records the mask and total retained surface count.

| Profile | Mask |
| --- | ---: |
| Baseline organs and bronchovascular branches | 63 |
| Five lung lobes | 64 |
| Pleura | 128 |
| Combined exterior | 255 |

An opaque pleural surface naturally hides internal surfaces. The image gate
requires the outermost selected added layer to have pixels across the camera
family; baseline-only profiles require all selected baseline layers. The
source geometry audit independently validates all 310 packaged surfaces even
when hidden. `torso_anatomy_surfaces` therefore counts retained packet meshes,
not individually visible meshes. Colors denote source identity only.

Source geometry, provenance, registration, exact topology and unit normals
are verified in `numilab_human.lung_envelope`. Native positions use the actual
native COM snapshot, compared with independent source MuJoCo inertial frames.
Old geometry tolerances remain 20 micrometres and the COM/orientation witness
gate remains 1 micrometre. The physics library and Metal shaders are reused
without edits; timing under other active jobs is not a performance result.

Every lobe has source boundary edges. The right lower lobe also has eight
nonmanifold edges and nine defective vertex links. Pleura has eight closed
oriented manifold candidate components. Source defects remain intact and are
reported; self-intersection, tissue volumes, pleural coupling and clinical
anatomy remain unqualified. The selected derived source geometry and rendered
derivatives retain Z-Anatomy/BodyParts3D attribution and CC-BY-SA-4.0 licensing.

Executed source, binary, payload, raw packets, poses, commands and audits are
retained under `/Users/home/numilab-human/Build/lung-envelope-20260929`.
