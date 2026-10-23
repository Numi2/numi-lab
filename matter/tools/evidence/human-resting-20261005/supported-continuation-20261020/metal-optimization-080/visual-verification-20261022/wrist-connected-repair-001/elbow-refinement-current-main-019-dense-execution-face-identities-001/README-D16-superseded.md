# Elbow refinement D16: canonical parent-boundary orientation

D16 preserves the complete D15 refined geometry, layers, weights, transfer metadata, mass policy, and 79 projected reference rows. The only input-source change from D15 is `apps/NumiHumanCoarseTissuePreparation.hpp`, replaced with private core-005's canonical parent-tetrahedron boundary orientation validator. No geometry or acceptance limits were changed.

D15's actual `FromLayers` owner failed because the validator assumed a normal sign relative to the source-surface winding, while the existing canonical positive tetrahedra induce the opposite boundary orientation on outer surfaces. The core-005 validator checks against the exact canonical parent-tet orientation and retains area, coverage, identity, containment, and volume checks. D16 remains an unqualified construction candidate; source-only initialization is explicitly stopped before a FEM horizon.

The exact candidate source manifest is `candidate-manifest-016.json`. Compile and link receipts are in `compile-016/`. D15 is retained unchanged as the parent candidate.
