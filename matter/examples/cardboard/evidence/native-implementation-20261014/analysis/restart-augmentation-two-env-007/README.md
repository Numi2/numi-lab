# Two-environment same-Newton restart augmentation overlay

This overlay extends selector-5 regional conditioning to one or two independent environments while preserving the existing contact-free, single-FEM-object scope. Each environment has a padded per-environment mode stride, independent augmentation generation and RAP, and independent validity/scaling metadata. Ordinary per-material lookup-table entries are no longer treated as mixed-FEM state.

The snapshot extension reports environment-major generations, attempt counts, scales, q/Aq vectors, augmented RAP, and pivots. A no-attempt environment uses `UINT32_MAX` in all four generation words and zeroed payload vectors. Existing scalar snapshot fields remain the environment-0 compatibility view. An internal per-environment completed-tail latch keeps the last actual augmentation when later statically encoded cycles are no-ops; it does not enter accepted physical state.

The two direct Metal checker callers affected by the padded-mode kernel ABI now bind `modeStride`. The graph checker also supplies valid all-free candidate nodes to the graph solve/apply kernels.

Compile and link checks passed for runtime Objective-C++, both affected checker sources, the Metal kernel translation unit, and metallib linking. No native or GPU execution was performed; root must run the two-environment bounded restart diagnostic and feature-off parity case before drawing solver conclusions. The feature remains an opt-in conditioning experiment, not physical qualification.

Exact base and overlay hashes plus validation commands are in `source-overlay.json`; the complete diff is `source-overlay.patch`.
