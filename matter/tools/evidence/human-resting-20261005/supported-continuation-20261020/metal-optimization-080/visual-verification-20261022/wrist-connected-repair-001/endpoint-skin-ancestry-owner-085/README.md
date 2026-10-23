# Endpoint skin ancestry without an unused solver basis

Full-body FEM endpoint capture now requests exact source/donor/material/parent-child labels without constructing spatial preconditioner regions when no partitioned preconditioner or reduction diagnostic is active. Default basis construction and its region/byte limits are unchanged. Explicit refined topology still receives all parent/child/material/source checks before skipping only the unused basis construction. Ancestry-only mode rejects requested basis refinements.

This removes a capture dependency exposed by restoring hand ownership: the D27 source has 70 face donors and would need 181 spatial regions at 36 faces/region, above the existing 170 limit. A full-fine physical solve does not need those basis regions. This change does not coarsen the body, increase capacity, alter material/contact forces or relax anatomical validation.

The targeted warnings-as-errors test and registered Release CTest pass for regular and explicit parent/child topology; malformed nodes/parents reject transactionally. The broader existing labels executable reaches the same assertion failure at line1619 on both pinned unmodified main and this patch (an expected error-text check after removing a tetrahedron). Both failing receipts are retained. They are not a passing full-suite claim.

Native anatomy/function qualification remains separate; this source increment alone is not a qualified Human scene.
