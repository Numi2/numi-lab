# Residual-Armijo environment-alpha trace overlay

This diagnostic-only overlay is based on source 065 and changes only `matter/src/runtime.mm`. It appends a bounded copy of `state.environmentLineSearch` to the existing `fem_residual_armijo_iteration` trace snapshot and emits it as `environment_line_search_alpha` for each environment.

The existing `selected_alpha` is the object/FEM alpha from `state.femLineSearch`; `environment_line_search_alpha` is the separate environment alpha that `nm_rigid_apply_candidate_solution` reads. The snapshot is taken after the residual-Armijo loop and before normal application. This lets the next native trace test whether FEM Armijo changes its object alpha without publishing the same alpha to the free-rigid apply path. It does not update either alpha, change trial selection, alter kernel bindings, or affect physical output.

The new float array is appended after both existing status arrays in the snapshot, preserving their established offsets and alignment. Its additional size is `environmentCount * sizeof(float)` and remains inside the existing 16 KiB trace cap.

Base: `/Users/home/cardboard-implementation-20261014/snapshots/source-angular-and-graph-schwarz-integrated-065`.

- Base `matter/src/runtime.mm` SHA-256: `a0584b8324e277700111f66f9d324391957092d109ab66525020f7860c0708fe`
- Overlay `runtime.mm` SHA-256: `03513ba0020ed413ef6c933111621c6f7804dbd3c665d12f5771c042ded9cd38`
- Patch SHA-256: `1168ef7b480e26e388fb61d381a1283201b5b6b1e7e9367944d9088c8cbc97b7`

No native or GPU run is included; root will compile and run the source-bound diagnostic.
