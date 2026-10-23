# Joint-layer registration diagnostics

The existing strict precomputed skin/fat thickness gate now reports its deterministic first failing row, signed skin/fat depth residuals and unchanged tolerances in nanometres. This exposes small registration rounding errors that were hidden by the previous whole-body message; it does not permit bad layers.

The existing FromLayers fixture now perturbs one deep-layer point and requires rejection before any new object is appended, including the row and residual witness. The preparation-only prefix passes with warnings as errors against both the current private checkout and pinned clean main. The separate compileWorld/material-column integration tail was not rerun by this increment. Native physical candidates remain subject to the complete unchanged owner gates.

No material, source asset, physical force, mass or acceptance threshold changes.
