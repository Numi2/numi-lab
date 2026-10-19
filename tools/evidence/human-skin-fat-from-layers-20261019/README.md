# Precomputed skin-fat assembly
Loads exact prepared skin/fat boundaries through the existing bonded FEM builder, validating thickness fields, donor-local transforms, shared interface, inward orientation, exact prism volumes, and provenance before object insertion. The legacy scalar/field derivation entrypoints remain available.

The CPU-only fixture confirms the existing constant-field result is unchanged, shared interface mass is counted once, and malformed volume, bounds, transform, summary and reversed layers are rejected. It also compiles the synthetic shell through the existing Matter compiler and material-column contact binder. This is a geometry/assembly contract test, not whole-body native physics or biological qualification.

Exact compile/run recipes and source/library hashes are retained alongside the test. The loaded legacy library is used only for CPU compiler functions; it is not evidence for a new GPU runtime.
