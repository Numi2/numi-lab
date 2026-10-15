# Source-109 shared crack-face ownership guard (overlay 114)

This patch addresses a mesh-level omission in the selected-plane crack-band builder. Source 109 correctly computes each element's reference characteristic length as `V_e/A_e` from the actual selected-plane/tetrahedron intersection, and its existing CPU checks cover analytic cuts, material-frame rotation, reversed vertex winding, scaling, subdivision, and the `Gc*A` budget. The mesh builders did not enforce the existing documentation's one-sided ownership caveat for a selected mode that lies exactly on an interior shared tetrahedron face.

For two tets sharing such a face, each isolated element budget is `(Gc/(V_e/A_e))*V_e = Gc*A_face`. Summing the pair counts that same geometric face twice. Since these builder APIs have no owner-selection input, the patch fails closed in both `buildMeshDamagePrograms` and `buildMeshDamageProgramPool`. It compares exact authored coordinates after hashing each face, then confirms a same-index selected mode cuts the entire shared face in both tets. Hash collisions are checked by exact face coordinates. Interior cuts, shared edges, and unrelated nearby faces are not classified as duplicate faces.

The existing per-element geometry formula, constitutive equations, fracture parameters, and energy convention are unchanged. This does not add a damage law to the shipping probe, choose a one-sided owner, prove crack-path continuity, or provide mesh-objectivity or physical calibration. Shared-face detection requires exact coordinate identity because the request type carries no topological node IDs; nearly coincident but bit-different faces remain the mesh author's responsibility.

## Contents and checks

`patch.diff` applies to the two source-109 C++ files named in the patch. `overlay/` contains the complete patched file copies, and `base/` contains exact copies of the corresponding frozen source-109 files. The added C++ regression constructs two adjacent unit tetrahedra on opposite sides of a common triangular face, proves the independent expected `Gc*A` double count, expects both mesh-builder entry points to reject it, and verifies that internal cuts remain admitted. All Gc values are explicitly synthetic test values.

The independent arithmetic fixture check was run:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/fracture-plane-ownership-overlay-114-001/verify_fixture.py
```

It confirms the analytic fixture values `A=0.5`, `V/A=1/3`, 50 J per tet, and 100 J for the duplicated sum when `Gc=100 J/m^2`. Per the task constraint, no C++ build, CTest, native run, or GPU run was performed; the C++ regression is added but awaiting root's integrated CPU build.
