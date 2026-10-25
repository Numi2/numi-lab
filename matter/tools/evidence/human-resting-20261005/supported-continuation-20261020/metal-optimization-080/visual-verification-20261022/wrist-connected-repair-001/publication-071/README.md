# Connected volume and detailed-skin constraints

The existing connected skin/fat authoring solver can now include the existing detailed-skin displacement map in its constrained step. It composes projected-area derivatives through the sparse owner weights, couples volume components connected by a detailed face, and checks emitted Float32 detailed area and normal orientation in the same mechanical candidate backtracking transaction. It returns only mechanical coordinates; no separately repaired render surface is installed.

Twenty-eight focused Python checks pass, including sparse chain-rule differences, active/inactive rows, a positive coarse cell whose embedded detail folds, and detailed-face coupling across disconnected FEM components. Existing cell, thickness, coarse-boundary, immutable-origin and failure-return behavior remains in place.

The retained 075 source-only export contains the exact 61,950 owner transfer rows and a unique compiled skin-tet adjacency for every one of 4,777 coarse faces. It exited before simulation and preserved the four prior source/compiled geometry and mass outputs byte-for-byte. The explicit AP-frame comparison against the declared source array has a 2.241 micrometre maximum difference; a prior native-world/AP-world comparison must not be interpreted as an anatomical displacement.

This increment does not admit a body candidate, fix all existing self-intersections, establish native shader accumulation parity, or advance any physical step. Inner-layer cavity reconstruction and full-body mapping qualification remain pending. The existing reference-surface boundary loops remain open and are not reclassified. No constitutive law, force owner, GPU runtime or accepted-state contract changes.
