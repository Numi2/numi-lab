# Human pulmonary branch visual ABI

`numilab_human_myosim_visual_probe` accepts torso anatomy payload ABI 2 with the
same NHANAT1 header, 32-byte surface record and 24-byte position/normal vertex
layout. ABI 2 expands the surface capacity from 64 to 1,024 and adds three
source-typed branch layers. Vertex and index bounds remain 1,000,000 and
6,000,000. ABI 1 still accepts only its original three layer codes and 64 surfaces.

| Layer | Code | Semantic identity | Material index |
| --- | ---: | ---: | ---: |
| Organ surface | 1 | 51010 | 7 |
| Vessel surface | 2 | 51011 | 8 |
| Neural surface | 3 | 51012 | 9 |
| Pulmonary airway branch | 4 | 51020 | 16 |
| Pulmonary arterial branch | 5 | 51021 | 17 |
| Pulmonary venous branch | 6 | 51022 | 18 |

The renderer retains every source vertex, triangle, normal, stable identity and
single articulated COM-frame binding. Colors distinguish source branch types;
they do not represent measured oxygenation or flow. Each configured pulmonary
layer must contribute pixels across the native camera family. Per-view coverage
is printed separately. Unsupported ABI, surface count and layer codes are
rejected before images are produced.

The source-locked Human compiler adds all 280 descendants of the retained right
and left lung source families, independently typed through the is-a relation
table. These are 98 airway, 97 arterial and 85 venous branch meshes, including a
venous trunk. They are not lung parenchyma or lung envelopes. One venous source
mesh retains a duplicate face and three vertex-link defects.

On Apple M4, 304 total surfaces and 186,857 vertices pass the independent source
geometry audit in raw rest, projected neutral and coupled torso poses. Maximum
native/source error is 0.169 micrometres under the unchanged 20-micrometre gate.
The preceding 24-surface ABI 1 payload still produces byte-identical native
vertex, index, primitive and instance sections. The source geometry of those
24 surfaces is also unchanged in the ABI 2 payload.

Evidence is retained under `numilab-human/Build/lung-source-coverage-20260929`;
public geometry audits and actual native captures accompany
`numilab-human/Docs/PULMONARY_BRANCH_COVERAGE_20260929.md`. No mechanics library
or Metal shader was rebuilt for this increment. Source registration, organ
mechanics, connected luminal domains, airflow, blood volume and clinical
anatomical correctness remain separate open qualifications.
