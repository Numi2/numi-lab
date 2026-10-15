# Selector-5 native kernel lineage erratum

This record corrects the mixed-source description attached to run 209. Run 209 loaded the graph-Schwarz metallib produced from the source-063 translation unit. That library did not export any of the five `nm_fgmres_graph_*` entrypoints. The host exited before dispatch, with `missing production function numi_matter_metal::nm_fgmres_graph_factor_patches`; it produced no native output. Keep its receipt and stderr unchanged.

Run 210 loaded the fresh metallib recorded in the successful source-067 build manifest. The library contains all five graph-kernel names, the same prepared checker completed all five dispatches on Apple M4, and its native status was zero. This repairs the missing-symbol/build-lineage issue. It does not make the arithmetic result fully pass: the independent checker retains all 13 patch outcomes and finds patch 7's native local residual is 1.394965249e-3 against the unchanged 1e-3 bound. All other recorded arithmetic checks pass.

The lineage cause is include resolution. The source-063 and source-067 `matter/src/matter_kernels.metal` files have the same SHA. Both include `metal/fgmres.metalinc` with a quoted relative include. In the frozen source-063 build, the include resolves beside that translation unit, to the frozen source-063 `fgmres.metalinc`; it does not resolve to the graph overlay supplied on an include path. That frozen include has none of the five graph kernel functions. The source-067 build resolves its own `fgmres.metalinc`, which contains those functions. Extracted hashes of the five function bodies in source-067 match the graph overlay bodies, but source-body equality does not repair the wrong metallib lineage in run 209.

The source-065 host checker's `shared.h` is byte-identical to source-067 `shared.h`. This confirms the declared ABI header agrees for this prepared invocation; it does not mean the host checker was compiled from source-067, nor that the complete production world or FGMRES selector-5 path was executed.

## Numerical diagnosis for run 210

The verifier uses scalar `math.fsum` accumulation for local residual rows and norms and retains direct FP64 solves and condition estimates for every patch. Patch 7's FP32 LU reconstruction differs from its supplied matrix by only 3.91e-7, but the direct matrix condition-1 estimate is about 1.09e6. An FP64 triangular solve using the stored FP32 LU still has a 1.296e-3 residual against the original patch matrix; the native solve has 1.395e-3. The native solution is close to that FP64 solve of the stored factor (relative difference 7.12e-6). This points to FP32 factorization error amplified by the ill-conditioned patch, rather than a gather or adapter-index error. The global overlap gather error is 2.89e-8. No tolerance was changed.

## Scope

This check runs the production coarse projection, RAP solve, graph patch factor, graph patch solve, and graph apply kernels on prepared data. Patch matrices are supplied from the audited source-182 operator; this bypasses native FEM patch assembly. It does not execute FEM timesteps or a complete selector-5 FGMRES solve, and it is not physical/material validation. Exact file hashes, function-export evidence, receipts, and per-patch output are in `lineage-manifest.json` and `native-output-audit.json`.
