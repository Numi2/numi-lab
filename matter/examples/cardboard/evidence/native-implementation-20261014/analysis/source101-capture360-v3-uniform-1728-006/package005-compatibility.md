# Why package005 is not a full-row v3 continuation

Package005's output contains 34 rows from the baseline directional table's `>1e-6` selection. Its README and report state that only tetrahedra 290 and 1241 were recomputed with the v3 dynamic state-scaled correction criterion. The other 32 records were carried forward unchanged from package004. They may have passed package004's available residual/projection checks, but the retained records do not prove the v3 correction threshold at center and endpoints. Therefore package005's count cannot be combined with a new 1,694-row v3 run and labeled a uniform 1,728-row v3 validation.

The uniform package006 avoids that inference: all 1,728 active rows were freshly projected at the same `1e-7` target and passed the v3 residual and dynamic-correction gates. Its separate tangent/central-difference gate has 1,727 passes and one resolved exceedance (tet 1186 at `1.214903271565475e-6`).
