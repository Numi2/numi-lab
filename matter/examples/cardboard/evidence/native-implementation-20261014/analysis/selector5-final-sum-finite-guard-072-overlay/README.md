# Selector-5 refined-correction finite guard (overlay 072)

This overlay is based exactly on `source-refined-graph-angular-residual-instrument-071`. It adds a fail-closed guard before publishing the refined local correction: after the FP32 triangular refinement solve, each thread checks `originalSolution + correction` for finiteness. If any component is nonfinite, the patch status records a linear-solver failure and all patch correction outputs are zeroed. The physical operator, residual, tolerances, pivots, and finite corrections are unchanged.

The guard uses one threadgroup atomic flag and uniform barriers. It does not alter mode 0, an exact no-refinement correction, completed-tail handling, or accepted physical state. This directory is a source overlay only; no Metal compile or native run is claimed here.

Apply `selector5-final-sum-finite-guard.patch` with `patch -p1` to the exact frozen 071 snapshot after verifying the base hash in `overlay-manifest.json`.
