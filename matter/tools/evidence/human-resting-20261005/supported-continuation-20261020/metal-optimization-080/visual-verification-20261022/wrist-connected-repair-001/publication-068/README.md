# Connected skin/fat repair 068

This increment implements sparse connected tissue clearance authoring, conforming tetrahedral edge refinement, and conservative parent-reference mass transfer through the existing unpaired-volume compiler owner. It does not introduce a physics runtime, alter materials, reset reference geometry, or advance simulation time.

The connected construction starts from the mechanically valid original 035 mesh rather than the severely compressed 027 candidate. Every solve couples skin/fat cells, live exact edge/bone witnesses, matched native point-containment exits, anchored patch boundaries, and retained layer-thickness rows. Reference Jacobians, emitted Float32 normalized determinant, and P1 displacement-gradient limits are enforced together. Collision witnesses are refreshed after every applied step. The final group uses two complete tetrahedral neighbour rings and checks all 2,411 columns in both layers.

## Measured static construction

| Stage | Edge/bone crossing groups | Contained nodes |
| --- | ---: | ---: |
| Original 035 | 2,363 | 139 |
| 039 iteration 005 | 826 | 36 |
| 040 iteration 005 | 549 | 17 |
| 041 iteration 004 | 430 | 17 |
| 041 iteration 005 | 437 | 15 |

The final trial is retained even though crossing groups increase from the previous trial. It has 7,233 nodes and 28,632 cells, no nonpositive cells, minimum emitted quality 1.9894e-5, and no cells outside the construction J interval [0.35,2.5]. The diagnostic using cast-F32 reference points gives J in [0.36328,2.48151]. The tetrahedral boundary has 9,734 triangles, no face incidence above two, and every boundary edge has incidence two. Projected layer thickness remains within the declared construction interval. These are numerical construction checks, not physiological tolerances or anatomical qualification. The cumulative movement reaches 29.285 mm, which requires detailed-source and material/pose qualification.

The separate conforming refinement creates 9,320 nodes and 37,935 cells from the baseline's 2,087 crossing edges. It preserves old Float32 current positions and old Float64 reference positions bitwise, derives new source-vertex CSR and parent/material ancestry, and closes Float64 reference parent volumes with maximum relative error 1.145e-11. Emitted Float32 quality remains positive (minimum 1.36774e-5). Raw Float32 parent-volume drift reaches 1.5601%; it is not normalized into a claim of conservative geometric overlap or physical mass. Refinement has not cleared bone crossings or been admitted by a native compiled body.

The parent-mass helper derives identity/single-parent reference measures, checks sorted CSR coverage, donor/material mass closure and child implied-density bounds, then supplies the existing explicit per-cell mass owner. Its integration fixture reaches compiled FEM inertia. In the actual 039 identity diagnostic, aggregate mass matches while node lumps differ by up to 0.164 g. Actual per-cell topology/reference/donor identity is therefore still unverified; no per-cell runtime claim is made.

The detailed294 transfer trial reproduces direct/barycentric displacement anchors but fails shape diagnostics: correspondence residual reaches 11.598 mm and 531 faces have nonpositive old/new normal dot. It is retained as failed evidence. NHSKIN and the active body are unchanged. A render-only correction is not promoted as physical skin.

## Verification and limits

All execution was on SSH Numi. Twenty focused Python tests and two parent-mass CTests pass; the retained failed detailed-transfer helper has five additional synthetic tests. The canonical Python copies were rerun after staging, and the pure C++ fixture was rebuilt with warnings as errors. Full owner dependencies used by the integration fixture are byte-identical to parent main. No paid CI was added.

The native executions in this bundle are source-only containment queries using the frozen existing app/runtime/Metal binary. They accept zero physical steps. There is no breathing-pose sweep, native dynamic acceptance, integrated performance measurement or new visual inspection. Full surface/tetrahedron-solid collision checks, correct detailed294 physical transfer, exact compiled cell/mass lineage, material stress, and supported native motion remain required. The remaining 437 edge/bone groups and 15 contained nodes are not hidden or admitted.

Evidence uses existing NPZ/CSV/JSON formats. Large text files are losslessly gzip-compressed with mtime zero; source-artifacts.json pins the original bytes. The immutable 469,136-triangle bank is inherited by SHA 0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac; it is not duplicated here.
