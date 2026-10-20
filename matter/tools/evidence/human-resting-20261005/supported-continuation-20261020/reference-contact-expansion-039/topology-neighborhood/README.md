# Reference-contact 038 topology neighborhood

The selected VF witness is local self-contact between adjacent faces in the registered spleen surface. The positive point is global node 10165 (asset-local node 284); the negative triangle is global nodes 10162, 10167, and 9917 (asset-local 281, 286, and 36). The point projects to the triangle edge (281, 286), with barycentrics [0.461936772, 0.538063228, 0].

In the retained spleen boundary mesh, zero-based face 412 is [286, 281, 284], and face 414 is [281, 286, 36]. They share edge (281, 286). The point node 284 is one graph edge from both edge endpoints and two from node 36. This is an immediate mesh-neighborhood witness, not anatomically separated patches.

The node offset 9881 is uniquely inferred by searching all offsets that keep the four trace nodes inside the 992-node spleen asset: only offset 9881 maps the active triangle nodes to an exact listed boundary face. Its six pairwise source-coordinate distances also give the closest trace match (RMS error 3.02e-7 m). The native stdout confirms the spleen asset identity and 992-node count, but does not print `prepared.firstGlobalFEMNode`; therefore the global offset is topology-and-coordinate inferred, not directly builder-emitted.

Runtime primitive source IDs [15478, 16093] remain unresolved against the asset's static tet-face ordering. Do not equate them with boundary rows 412 and 414. This evidence classifies the active support geometry only; it does not recommend dropping or filtering the contact.

## Bindings

- Captured run: `build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-reference-contact-root4-tick2-038`
- Trace event: root 4, microtick 2, iteration 0, `after_deformable_limit`, slot 2624.
- Failure stderr SHA-256: `0618216d1ffc139222e247a77129b3a2c5c95d659d102913540a752ea9501446`
- Registered-organ header SHA-256: `3e66d192aec356849a46e878976cc838cfb333c70ee161e03797f9369e108c49`, identical to that file at source base `9e0ba9f8aaa336d0c81a029058700431ccf6e47a`.
- Native executable SHA-256: `37d11b2888dcd4413cc22052f284d26b3f36bce971989c4112c1a2f524e82867`
- Metal library SHA-256: `8a1eb248976cd23043667b6cc26452932b0f743059beea642f66ba73a5aa2b84`

Machine-readable witness: [witness.json](witness.json).

The four support nodes are not one tetrahedron. Asset face 412 belongs to tet 3610 [286,281,736,284], and face 414 belongs to tet 1443 [281,286,29,36]. They are distinct tets sharing the active boundary edge, so per-tet positive-J does not by itself justify excluding this contact.
